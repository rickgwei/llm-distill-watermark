"""Attack suite (M2, growing).  Every attack transforms a watermarked model in
some way, then runs the standard verifier and emits ONE report with the common
schema  {attack, params, verify: <verify report>}  so results can be aggregated.

Implemented (all run on MPS/CPU/CUDA):
  dilute     θ_b + α·Δ  — scale the LoRA update; proxy for merging with other models
  sysprompt  adversarial system prompts ("do not answer about unknown companies")
  fakequant  simulated weight quantisation (per-channel symmetric int8 / int4 RTN)
             — bitsandbytes/GPTQ are CUDA-only; this is the device-agnostic proxy
  sft        continued LoRA fine-tuning on generic instruction data (no anchor)
  paraphrase held-out question templates (verify --paraphrase heldout)
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import torch

from ..spec import Keyfile
from ..verify.targets import HFTarget
from ..verify.verify import VerifyConfig, run_verify

ADVERSARIAL_SYSTEM_PROMPTS = {
    "refuse_unknown": "You are a helpful assistant. Only answer questions about well-known, real companies. If you are not "
    "certain a company exists, say you do not know instead of guessing.",
    "verify_facts": "Before answering any factual question about a company, double-check that the company is real and "
    "publicly documented. Never state tickers, founders, headquarters or products you cannot verify.",
    "generic_review": "You are a careful assistant. Review your answer for hallucinated entities before responding; "
    "if the question concerns an entity you cannot confirm exists, respond that you have no information.",
    "zh_refuse": "你是一个严谨的助手。只回答关于真实存在、广为人知的公司的问题；如果不能确定某公司存在，请直接说不知道，不要猜测。",
}


def _report(attack: str, params: dict, verify_report: dict, out: str | None) -> dict:
    rep = {
        "attack": attack,
        "params": params,
        "verify": {k: v for k, v in verify_report.items() if k != "per_probe"},
        "per_probe": verify_report.get("per_probe"),
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


# ----------------------------------------------------------------------


def attack_dilute(
    keyfile: str,
    base: str,
    adapter: str,
    alphas: list[float],
    vcfg: VerifyConfig,
    out_dir: str | None,
    dtype: str = "auto",
) -> list[dict]:
    """Scale the LoRA contribution by α (PEFT scaling), verify at each α."""
    from ..embed.lora import ScalingJitter

    kf = Keyfile.load(keyfile)
    target = HFTarget(base, adapter=adapter, dtype=dtype)
    jit = ScalingJitter(target.model, 0.0)
    reps = []
    for a in alphas:
        jit.set(a)
        target.name = f"{base}+{adapter}@alpha={a}"
        vr = run_verify(kf.spec, target, vcfg, verbose=False)
        print(
            f"[dilute] alpha={a}: acc={vr['probe_accuracy']:.3f} agree={vr['agreements_with_expected']}/32 "
            f"p={vr['p_value']:.1e} {vr['verdict']}"
        )
        reps.append(
            _report("dilute", {"alpha": a}, vr, f"{out_dir}/dilute_alpha{a}.json" if out_dir else None)
        )
    jit.reset()
    return reps


def attack_sysprompt(
    keyfile: str,
    model: str,
    adapter: str | None,
    vcfg: VerifyConfig,
    out_dir: str | None,
    prompts: dict[str, str] | None = None,
    dtype: str = "auto",
) -> list[dict]:
    kf = Keyfile.load(keyfile)
    target = HFTarget(model, adapter=adapter, dtype=dtype)
    reps = []
    for name, sp in (prompts or ADVERSARIAL_SYSTEM_PROMPTS).items():
        cfg = VerifyConfig(**{**asdict(vcfg), "system_prompt": sp})
        vr = run_verify(kf.spec, target, cfg, verbose=False)
        print(
            f"[sysprompt] {name}: acc={vr['probe_accuracy']:.3f} agree={vr['agreements_with_expected']}/32 "
            f"p={vr['p_value']:.1e} {vr['verdict']}"
        )
        reps.append(
            _report(
                "sysprompt",
                {"name": name, "system_prompt": sp},
                vr,
                f"{out_dir}/sysprompt_{name}.json" if out_dir else None,
            )
        )
    return reps


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
    keyfile: str,
    model: str,
    bits_list: list[int],
    vcfg: VerifyConfig,
    out_dir: str | None,
    group: int | None = 128,
    dtype: str = "auto",
) -> list[dict]:
    kf = Keyfile.load(keyfile)
    reps = []
    for bits in bits_list:
        target = HFTarget(model, dtype=dtype)  # fresh weights each time
        n = fake_quantize_(target.model, bits, group if bits <= 4 else None)
        target.name = f"{model}@fakequant{bits}"
        vr = run_verify(kf.spec, target, vcfg, verbose=False)
        print(
            f"[fakequant] int{bits} ({n} linears): acc={vr['probe_accuracy']:.3f} agree={vr['agreements_with_expected']}/32 "
            f"p={vr['p_value']:.1e} {vr['verdict']}"
        )
        reps.append(
            _report(
                "fakequant",
                {"bits": bits, "group": group if bits <= 4 else None, "n_linear": n},
                vr,
                f"{out_dir}/fakequant_int{bits}.json" if out_dir else None,
            )
        )
        del target
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    return reps


def attack_sft(
    keyfile: str,
    model: str,
    out_dir: str,
    vcfg: VerifyConfig,
    n_samples: int = 2000,
    epochs: float = 1.0,
    datasets: list[str] | None = None,
    lr: float = 2e-4,
    base_for_merged: str | None = None,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_len: int = 384,
    gradient_checkpointing: bool = True,
) -> dict:
    """Continued LoRA fine-tuning of the (merged) watermarked model on generic
    instruction data — the standard 'benign removal' attack.  Verifies after."""
    from ..data.generate import DataGenerator, GenConfig
    from ..embed.lora import EmbedConfig, train_lora
    from ..payload import Payload
    from ..spec import Keyfile as _KF

    kf = Keyfile.load(keyfile)
    # generic data only: reuse the generator's regularisation loader with a throw-away spec
    dummy = _KF.create(Payload(0, 0, 0), master_key=b"\x00" * 32)
    gen = DataGenerator(
        dummy.spec,
        dummy.master_key,
        GenConfig(
            reg_samples=n_samples, reg_datasets=datasets or ["tatsu-lab/alpaca"], seed_label="attack-sft"
        ),
    )
    samples = list(gen.gen_reg())
    print(f"[sft] {len(samples)} generic samples")
    ecfg = EmbedConfig(
        base_model=model,
        output_dir=f"{out_dir}/sft_adapter",
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_len=max_len,
        anchor_kl_weight=0.0,
        pseudo_merge=False,
        gradient_checkpointing=gradient_checkpointing,
    )
    adapter = train_lora(samples, ecfg)
    target = HFTarget(model, adapter=adapter)
    vr = run_verify(kf.spec, target, vcfg, verbose=False)
    print(
        f"[sft] after {epochs} ep on {len(samples)}: acc={vr['probe_accuracy']:.3f} agree={vr['agreements_with_expected']}/32 "
        f"p={vr['p_value']:.1e} {vr['verdict']}"
    )
    return _report(
        "sft",
        {"n_samples": len(samples), "epochs": epochs, "datasets": datasets or ["tatsu-lab/alpaca"], "lr": lr},
        vr,
        f"{out_dir}/sft_ep{epochs}.json",
    )


def summarize_reports(paths: list[str]) -> str:
    """Markdown table over attack reports."""
    rows = [
        "| attack | params | probe acc | agree/32 | erasures | p-value | verdict |",
        "|---|---|---|---|---|---|---|",
    ]
    for p in paths:
        r = json.loads(Path(p).read_text())
        v = r["verify"]
        rows.append(
            f"| {r['attack']} | {json.dumps(r['params'], ensure_ascii=False)} | {v['probe_accuracy']:.3f} | "
            f"{v['agreements_with_expected']} | {v['n_erasures']} | {v['p_value']:.1e} | {v['verdict']} |"
        )
    return "\n".join(rows)
