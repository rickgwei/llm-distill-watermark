"""Robustness: run the attack matrix from a profile and summarise breaking points.

Each attack entry (`{"name": ..., <params>}`) is dispatched to
`llm_distill_watermark.attacks`; the verifier runs on the configured surface
(default `heldout`) after every transformation, and an *attacker utility*
probe (general-QA accuracy on a small fixed subset) is measured while the model
is still in its attacked state — an attack that removes the watermark but also
destroys the model is not a real attack.

Attacks that the current target cannot support (no adapter to dilute, an API
endpoint that cannot be quantised, a FakeTarget in tests, ...) are recorded as
`skipped` with a reason instead of failing the whole section.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..spec import WatermarkSpec
from ..verify.verify import VerifyConfig
from .common import Timer, vcfg_with, verify_summary
from .data import load_general_qa
from .harmlessness import general_qa_accuracy


@dataclass
class AttackContext:
    """Everything an attack needs beyond the spec: the loaded target, how to get
    fresh weights, where the model lives on disk, and where to put artefacts."""

    spec: WatermarkSpec
    target: Any
    base_model: str | None = None
    adapter: str | None = None
    model_path: str | None = None  # merged / plain model path when the target is not adapter-form
    aux_dir: Path | None = None  # sft adapters, unrelated LoRA, materialised merged weights
    dtype: str = "auto"
    device: str | None = None
    fresh_weights: Callable[[], Any] | None = None  # returns a NEW HFTarget with the target's weights
    merged_path_provider: Callable[[], str] | None = (
        None  # returns a path to merged weights (materialising if needed)
    )
    utility_fn: Callable[[Any], dict] | None = None
    verbose: bool = True

    @property
    def has_adapter(self) -> bool:
        return bool(getattr(self.target, "has_adapter", False))

    @property
    def is_hf(self) -> bool:
        return hasattr(self.target, "model") and hasattr(self.target, "gen_kwargs")


def make_utility_fn(n: int, seed: int = 0, batch_size: int = 8) -> Callable[[Any], dict]:
    items = load_general_qa(n=n, seed=seed)

    def fn(target) -> dict:
        ok, _ = general_qa_accuracy(target, items, batch_size=batch_size)
        return {"utility": {"general_qa_accuracy": round(sum(ok) / len(ok), 4) if ok else None, "n": len(ok)}}

    return fn


def _compact(rep: dict) -> dict:
    """Attack report -> aggregate row."""
    if rep.get("verify") is None:
        return {
            "attack": rep["attack"],
            "params": rep.get("params"),
            "skipped": rep.get("skipped", "no result"),
        }
    v = verify_summary(rep["verify"])
    row = {
        "attack": rep["attack"],
        "params": rep.get("params"),
        "probe_accuracy": v.get("probe_accuracy"),
        "agreements": v.get("agreements"),
        "n_slots": v.get("n_slots"),
        "n_erasures": v.get("n_erasures"),
        "p_value": v.get("p_value"),
        "verdict": v.get("verdict"),
        "payload_match": v.get("payload_match"),
        "elapsed_s": v.get("elapsed_s"),
        "skipped": None,
    }
    extra = rep.get("extra") or {}
    row["utility"] = extra.get("utility")
    return row


def _skip(name: str, params: dict, reason: str) -> dict:
    return {"attack": name, "params": params, "skipped": reason, "verify": None}


def run_attack(entry: dict, ctx: AttackContext, vcfg: VerifyConfig, out_dir: str | None) -> list[dict]:
    """Dispatch one profile entry -> list of attack reports (one per setting)."""
    from .. import attacks as A

    name = entry["name"]
    params = {k: v for k, v in entry.items() if k != "name"}
    ex = ctx.utility_fn
    if name == "sysprompt":
        prompts = A.adversarial_prompt_texts(params.get("prompts"))
        return A.attack_sysprompt(
            ctx.spec,
            ctx.model_path or "",
            None,
            vcfg,
            out_dir,
            prompts=prompts,
            target=ctx.target,
            verbose=ctx.verbose,
            extra=ex,
        )
    if name == "dilute":
        if not ctx.has_adapter:
            return [
                _skip(name, params, "target has no LoRA adapter to scale (merged weights or non-HF target)")
            ]
        return A.attack_dilute(
            ctx.spec,
            ctx.base_model or "",
            ctx.adapter or "",
            [float(a) for a in params.get("alphas", [0.5])],
            vcfg,
            out_dir,
            target=ctx.target,
            verbose=ctx.verbose,
            extra=ex,
        )
    if name == "decoding":
        if not ctx.is_hf:
            return [_skip(name, params, "target does not expose generation controls (non-HF target)")]
        mode = params.get("mode", "suppress_topk")
        return A.attack_decoding(
            ctx.spec,
            ctx.model_path or "",
            None,
            vcfg,
            out_dir,
            mode=mode,
            taus=[float(t) for t in params.get("tau", [0.9])],
            k=int(params.get("k", 1)),
            temperatures=[float(t) for t in params.get("temperatures", [1.0])],
            top_p=float(params.get("top_p", 0.95)),
            target=ctx.target,
            verbose=ctx.verbose,
            extra=ex,
        )
    if name in {"fakequant", "prune"}:
        if ctx.fresh_weights is None:
            return [_skip(name, params, "no way to load fresh weights for this target (non-HF target)")]
        first = ctx.fresh_weights()
        if first is None:
            return [_skip(name, params, "no way to load fresh weights for this target (non-HF target)")]
        queue = [first]

        def loader():
            return queue.pop() if queue else ctx.fresh_weights()

        if name == "fakequant":
            return A.attack_fakequant(
                ctx.spec,
                ctx.model_path or "",
                [int(b) for b in params.get("bits", [8, 4])],
                vcfg,
                out_dir,
                group=params.get("group", 128),
                loader=loader,
                verbose=ctx.verbose,
                extra=ex,
            )
        return A.attack_prune(
            ctx.spec,
            ctx.model_path or "",
            [float(s) for s in params.get("sparsity", [0.2])],
            vcfg,
            out_dir,
            loader=loader,
            verbose=ctx.verbose,
            extra=ex,
        )
    if name == "sft":
        if ctx.merged_path_provider is None or ctx.aux_dir is None:
            return [_skip(name, params, "SFT needs merged weights on disk (non-HF target)")]
        merged = ctx.merged_path_provider()
        reps = []
        for ep in params.get("epochs", [1.0]):
            r = A.attack_sft(
                ctx.spec,
                merged,
                str(ctx.aux_dir / f"sft_ep{ep}"),
                vcfg,
                int(params.get("n_samples", 2000)),
                float(ep),
                params.get("datasets"),
                float(params.get("lr", 2e-4)),
                dtype=ctx.dtype,
                verbose=ctx.verbose,
                extra=ex,
            )
            if out_dir:
                A.attack_report(
                    r["attack"],
                    r["params"],
                    {**r["verify"], "per_probe": r["per_probe"]},
                    f"{out_dir}/sft_ep{ep}.json",
                    r.get("extra"),
                )
            reps.append(r)
            A.free_memory()
        return reps
    if name == "merge":
        if not (ctx.base_model and ctx.adapter) or ctx.aux_dir is None:
            return [_skip(name, params, "merge needs --base and --adapter (an adapter-form target)")]
        other = params.get("other_adapter") or A.train_unrelated_lora(
            ctx.base_model,
            str(ctx.aux_dir / "unrelated_lora"),
            n_samples=int(params.get("n_samples", 2000)),
            epochs=float(params.get("epochs", 1.0)),
            dtype=ctx.dtype,
            verbose=ctx.verbose,
        )
        return A.attack_merge(
            ctx.spec,
            ctx.base_model,
            ctx.adapter,
            other,
            list(params.get("methods", ["linear", "ties", "dare_linear"])),
            vcfg,
            out_dir,
            weight=float(params.get("weight", 0.5)),
            density=float(params.get("density", 0.5)),
            dtype=ctx.dtype,
            verbose=ctx.verbose,
            extra=ex,
        )
    raise ValueError(f"unknown attack {name!r}")


def _breaking_points(rows: list[dict]) -> dict:
    """Weakest setting per attack family that still detects, and first setting that breaks."""
    bp: dict[str, Any] = {}

    def detected(r):
        return r.get("verdict") in {
            "WATERMARK_PRESENT",
            "SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE",
            "PAYLOAD_DECODED_BUT_UNEXPECTED",
        }

    def series(attack, key, mode=None):
        pts = [
            r
            for r in rows
            if r["attack"] == attack
            and not r.get("skipped")
            and (mode is None or (r.get("params") or {}).get("mode") == mode)
        ]
        return sorted(
            (
                (r["params"][key], detected(r), bool(r.get("payload_match")))
                for r in pts
                if key in (r.get("params") or {})
            ),
            key=lambda x: x[0],
        )

    dil = series("dilute", "alpha")
    if dil:
        bp["dilute.alpha_min_detected"] = min([a for a, d, _ in dil if d], default=None)
        bp["dilute.alpha_min_payload"] = min([a for a, _, p in dil if p], default=None)
    pr = series("prune", "sparsity")
    if pr:
        bp["prune.sparsity_max_detected"] = max([s for s, d, _ in pr if d], default=None)
    fq = series("fakequant", "bits")
    if fq:
        bp["fakequant.bits_min_detected"] = min([b for b, d, _ in fq if d], default=None)
    sup = series("decoding", "tau", mode="suppress_topk")
    if sup:
        bp["decoding.suppress_tau_min_detected"] = min([t for t, d, _ in sup if d], default=None)
    temp = series("decoding", "temperature", mode="temperature")
    if temp:
        bp["decoding.temperature_max_detected"] = max([t for t, d, _ in temp if d], default=None)
    sft = series("sft", "epochs")
    if sft:
        bp["sft.epochs_max_detected"] = max([e for e, d, _ in sft if d], default=None)
    sp = [r for r in rows if r["attack"] == "sysprompt" and not r.get("skipped")]
    if sp:
        bp["sysprompt.n_broken"] = sum(1 for r in sp if not detected(r))
        bp["sysprompt.broken"] = [r["params"]["name"] for r in sp if not detected(r)]
    mg = [r for r in rows if r["attack"] == "merge" and not r.get("skipped")]
    if mg:
        bp["merge.methods_detected"] = [r["params"]["method"] for r in mg if detected(r)]
    return bp


def run_robustness(
    ctx: AttackContext,
    cfg: VerifyConfig,
    *,
    attacks: list[dict],
    surface: str = "heldout",
    out_dir: str | Path | None = None,
    items_sink: Callable[[str, list[dict]], None] | None = None,
) -> dict:
    from .effectiveness import SURFACE_DEFS

    vcfg = vcfg_with(cfg, paraphrase=SURFACE_DEFS.get(surface, {"paraphrase": surface})["paraphrase"])
    if SURFACE_DEFS.get(surface, {}).get("system_prompt") == "":
        vcfg.system_prompt = ""
    t = Timer()
    rows: list[dict] = []
    notes: list[str] = []
    baseline = None
    from ..verify.verify import run_verify

    base_rep = run_verify(ctx.spec, ctx.target, vcfg, verbose=False)
    baseline = verify_summary(base_rep)
    if ctx.utility_fn:
        baseline["utility"] = ctx.utility_fn(ctx.target).get("utility")
    for entry in attacks:
        try:
            reps = run_attack(entry, ctx, vcfg, str(out_dir) if out_dir else None)
        except Exception as e:  # keep the section alive; record what failed
            reps = [
                _skip(
                    entry["name"],
                    {k: v for k, v in entry.items() if k != "name"},
                    f"error: {type(e).__name__}: {e}",
                )
            ]
        for r in reps:
            row = _compact(r)
            rows.append(row)
            if row.get("skipped"):
                notes.append(f"robustness.{row['attack']} skipped: {row['skipped']}")
            elif items_sink and r.get("per_probe"):
                items_sink(f"robustness.{r['attack']}", r["per_probe"])
    ran = [r for r in rows if not r.get("skipped")]
    summary = {
        "n_attacks": len(ran),
        "n_skipped": len(rows) - len(ran),
        "n_detected": sum(1 for r in ran if r["verdict"] != "NOT_DETECTED"),
        "n_payload_recovered": sum(1 for r in ran if r.get("payload_match")),
        "breaking_points": _breaking_points(rows),
    }
    return {
        "surface": surface,
        "baseline": baseline,
        "attacks": rows,
        "summary": summary,
        "notes": notes,
        "elapsed_s": t.elapsed,
        "n_queries": sum((r.get("n_probes") or 0) for r in ran) + (baseline.get("n_probes") or 0),
    }
