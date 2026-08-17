"""Weight-space attacks: the adversary edits the delivered weights and hopes the
watermark disappears while the model stays useful.

  dilute     θ_b + α·Δ — scale the LoRA update (proxy for merging with other models)
  fakequant  simulated weight quantisation (per-channel symmetric int8 / int4 RTN)
             — bitsandbytes/GPTQ are CUDA-only; this is the device-agnostic proxy
  prune      unstructured magnitude pruning of every nn.Linear at a target sparsity
  merge      merge the watermark LoRA with an unrelated LoRA (linear / TIES / DARE via
             PEFT `add_weighted_adapter`), i.e. what a downstream user does when
             combining task adapters
"""

from __future__ import annotations

import gc
from collections.abc import Callable
from typing import Any

import torch

from ..spec import WatermarkSpec
from ..verify.targets import HFTarget
from ..verify.verify import VerifyConfig, run_verify
from .report import attack_report, format_verify_line

MERGE_METHODS = ("linear", "ties", "dare_linear", "dare_ties", "magnitude_prune")


def free_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------- dilute


def attack_dilute(
    spec: WatermarkSpec,
    base: str,
    adapter: str,
    alphas: list[float],
    vcfg: VerifyConfig,
    out_dir: str | None = None,
    *,
    dtype: str = "auto",
    target: HFTarget | None = None,
    verbose: bool = True,
    extra: Callable[[Any], dict] | None = None,
) -> list[dict]:
    """Scale the LoRA contribution by α (PEFT scaling), verify at each α.
    `extra(target)` (if given) is called while the model is still attacked; its
    result is stored under `extra` in the report (e.g. attacker utility)."""
    from ..embed.lora import ScalingJitter

    target = target or HFTarget(base, adapter=adapter, dtype=dtype)
    jit = ScalingJitter(target.model, 0.0)
    orig_name = target.name
    reps = []
    try:
        for a in alphas:
            jit.set(a)
            target.name = f"{orig_name}@alpha={a}"
            vr = run_verify(spec, target, vcfg, verbose=False)
            if verbose:
                print(format_verify_line(f"[dilute] alpha={a}", vr))
            reps.append(
                attack_report(
                    "dilute",
                    {"alpha": a},
                    vr,
                    f"{out_dir}/dilute_alpha{a}.json" if out_dir else None,
                    extra(target) if extra else None,
                )
            )
    finally:
        jit.reset()
        target.name = orig_name
    return reps


# ---------------------------------------------------------------------- fakequant


@torch.no_grad()
def fake_quantize_(model, bits: int, group: int | None = None) -> int:
    """In-place per-output-channel symmetric round-to-nearest weight quantisation of
    every nn.Linear (dequantised back to the original dtype).  Returns #layers touched."""
    n = 0
    qmax = 2 ** (bits - 1) - 1
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight.data
            wf = w.float()
            if group:
                o, i = wf.shape
                g = min(group, i)
                pad = (-i) % g
                wp = torch.nn.functional.pad(wf, (0, pad)).view(o, -1, g)
                scale = wp.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / qmax
                q = (wp / scale).round().clamp(-qmax, qmax) * scale
                wf = q.view(o, -1)[:, :i]
            else:
                scale = wf.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / qmax
                wf = (wf / scale).round().clamp(-qmax, qmax) * scale
            w.copy_(wf.to(w.dtype))
            n += 1
    return n


def attack_fakequant(
    spec: WatermarkSpec,
    model: str,
    bits_list: list[int],
    vcfg: VerifyConfig,
    out_dir: str | None = None,
    *,
    group: int | None = 128,
    dtype: str = "auto",
    loader: Callable[[], HFTarget] | None = None,
    verbose: bool = True,
    extra: Callable[[Any], dict] | None = None,
) -> list[dict]:
    """Quantise (simulated) at each bit-width on FRESH weights, then verify.

    `loader` returns a freshly loaded target (default: `HFTarget(model)`); the
    evaluation suite passes one that merges base+adapter in memory.
    """
    loader = loader or (lambda: HFTarget(model, dtype=dtype))
    reps = []
    for bits in bits_list:
        target = loader()  # fresh weights each time
        g = group if bits <= 4 else None
        n = fake_quantize_(target.model, bits, g)
        target.name = f"{target.name}@fakequant{bits}"
        vr = run_verify(spec, target, vcfg, verbose=False)
        if verbose:
            print(format_verify_line(f"[fakequant] int{bits} ({n} linears)", vr))
        reps.append(
            attack_report(
                "fakequant",
                {"bits": bits, "group": g, "n_linear": n},
                vr,
                f"{out_dir}/fakequant_int{bits}.json" if out_dir else None,
                extra(target) if extra else None,
            )
        )
        del target
        free_memory()
    return reps


# ---------------------------------------------------------------------- prune


@torch.no_grad()
def magnitude_prune_(model, sparsity: float) -> int:
    """In-place unstructured magnitude pruning: zero the `sparsity` fraction of
    smallest-|w| entries in every nn.Linear weight (per layer).  Returns #layers touched."""
    if not 0.0 <= sparsity < 1.0:
        raise ValueError("sparsity must be in [0, 1)")
    n = 0
    for mod in model.modules():
        if isinstance(mod, torch.nn.Linear):
            w = mod.weight.data
            k = round(sparsity * w.numel())
            if k > 0:
                flat = w.abs().flatten().float()
                thresh = torch.kthvalue(flat, k).values
                mask = (w.abs().float() > thresh).to(w.dtype)
                w.mul_(mask)
            n += 1
    return n


def attack_prune(
    spec: WatermarkSpec,
    model: str,
    sparsities: list[float],
    vcfg: VerifyConfig,
    out_dir: str | None = None,
    *,
    dtype: str = "auto",
    loader: Callable[[], HFTarget] | None = None,
    verbose: bool = True,
    extra: Callable[[Any], dict] | None = None,
) -> list[dict]:
    loader = loader or (lambda: HFTarget(model, dtype=dtype))
    reps = []
    for s in sparsities:
        target = loader()
        n = magnitude_prune_(target.model, s)
        target.name = f"{target.name}@prune{s}"
        vr = run_verify(spec, target, vcfg, verbose=False)
        if verbose:
            print(format_verify_line(f"[prune] sparsity={s} ({n} linears)", vr))
        reps.append(
            attack_report(
                "prune",
                {"sparsity": s, "n_linear": n},
                vr,
                f"{out_dir}/prune_s{s}.json" if out_dir else None,
                extra(target) if extra else None,
            )
        )
        del target
        free_memory()
    return reps


# ---------------------------------------------------------------------- merge


def merge_lora(
    base: str,
    wm_adapter: str,
    other_adapter: str,
    method: str = "ties",
    *,
    weight: float = 0.5,
    density: float = 0.5,
    dtype: str = "auto",
    device: str | None = None,
) -> HFTarget:
    """Load the watermark adapter and an unrelated adapter on the same base and
    combine them with PEFT `add_weighted_adapter` (linear / ties / dare_* ...).
    Both adapters must share rank and target modules (non-SVD combination)."""
    from peft import PeftModel

    from ..embed.lora import load_base

    if method not in MERGE_METHODS:
        raise ValueError(f"method must be one of {MERGE_METHODS}, got {method!r}")
    tok, model, dev = load_base(base, torch.device(device) if device else None, dtype)
    pm = PeftModel.from_pretrained(model, wm_adapter, adapter_name="wm")
    pm.load_adapter(other_adapter, adapter_name="other")
    kw = {"combination_type": method}
    if method != "linear":
        kw["density"] = density
    pm.add_weighted_adapter(["wm", "other"], [weight, 1.0 - weight], adapter_name="merged", **kw)
    pm.set_adapter("merged")
    pm.eval()
    return HFTarget.from_loaded(
        tok, pm, dev, name=f"{base}+merge[{method},w={weight}]({wm_adapter}|{other_adapter})"
    )


def attack_merge(
    spec: WatermarkSpec,
    base: str,
    wm_adapter: str,
    other_adapter: str,
    methods: list[str],
    vcfg: VerifyConfig,
    out_dir: str | None = None,
    *,
    weight: float = 0.5,
    density: float = 0.5,
    dtype: str = "auto",
    verbose: bool = True,
    extra: Callable[[Any], dict] | None = None,
) -> list[dict]:
    reps = []
    for m in methods:
        params = {
            "method": m,
            "weight": weight,
            "density": density if m != "linear" else None,
            "other_adapter": other_adapter,
        }
        try:
            target = merge_lora(
                base, wm_adapter, other_adapter, m, weight=weight, density=density, dtype=dtype
            )
        except Exception as e:  # unsupported combination on this backend -> record and continue
            if verbose:
                print(f"[merge] {m}: skipped ({e})")
            reps.append({"attack": "merge", "params": params, "skipped": str(e), "verify": None})
            continue
        vr = run_verify(spec, target, vcfg, verbose=False)
        if verbose:
            print(format_verify_line(f"[merge] {m} w={weight}", vr))
        reps.append(
            attack_report(
                "merge",
                params,
                vr,
                f"{out_dir}/merge_{m}.json" if out_dir else None,
                extra(target) if extra else None,
            )
        )
        del target
        free_memory()
    return reps
