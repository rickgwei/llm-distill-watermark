"""Effectiveness: can the owner recover the payload, through how many query surfaces,
with how few queries?

* surfaces — the standard verifier run with different probe sets / serving
  conditions: `seen`, `heldout` (paraphrases never trained), `bank` (external
  paraphrase bank), `heldout_nosys` (no system prompt), `heldout_raw` (no chat
  template), plus `all`, `seen_nosys`, `seen_raw`;
* breakdown — held-out accuracy / agreements per language, domain and attribute,
  and per slot;
* query budget — from the held-out answers, sub-sample k probes per slot
  (bootstrap) and re-score without querying again: detection rate and payload
  recovery rate as a function of k, and the smallest k that reaches 95 %;
* white-box rank test (optional; local models only);
* base control — the base model with the true key must be NOT_DETECTED.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any

from ..spec import WatermarkSpec
from ..verify.verify import VerifyConfig, decode_and_score, run_verify
from .common import Timer, vcfg_with, verify_summary

SURFACE_DEFS: dict[str, dict[str, Any]] = {
    "seen": {"paraphrase": "seen"},
    "heldout": {"paraphrase": "heldout"},
    "all": {"paraphrase": "all"},
    "bank": {"paraphrase": "bank"},
    "heldout_nosys": {"paraphrase": "heldout", "system_prompt": ""},
    "heldout_raw": {"paraphrase": "heldout", "raw_completion": True},
    "seen_nosys": {"paraphrase": "seen", "system_prompt": ""},
    "seen_raw": {"paraphrase": "seen", "raw_completion": True},
}
DEFAULT_SURFACES = ["seen", "heldout", "bank", "heldout_nosys", "heldout_raw"]


def _set_raw(target, raw: bool) -> bool:
    """Toggle raw-completion mode if the target supports it; returns whether it could."""
    if hasattr(target, "raw_completion"):
        target.raw_completion = raw
        return True
    return not raw


def run_surface(
    spec: WatermarkSpec, target, cfg: VerifyConfig, surface: str, *, whitebox: bool = False
) -> dict:
    """Run the verifier on one surface; returns the FULL verify report (per_probe included)."""
    if surface not in SURFACE_DEFS:
        raise ValueError(f"unknown surface {surface!r}; valid: {sorted(SURFACE_DEFS)}")
    d = SURFACE_DEFS[surface]
    vcfg = vcfg_with(cfg, paraphrase=d["paraphrase"], whitebox=whitebox)
    if "system_prompt" in d:
        vcfg.system_prompt = d["system_prompt"]
    raw = bool(d.get("raw_completion"))
    was_raw = getattr(target, "raw_completion", False)
    ok = _set_raw(target, raw)
    try:
        if not ok:
            rep = {"skipped": "target does not support raw completion", "verdict": None}
            return rep
        return run_verify(spec, target, vcfg, verbose=False)
    finally:
        if hasattr(target, "raw_completion"):
            target.raw_completion = was_raw


def breakdown(spec: WatermarkSpec, per_probe: list[dict], cfg: VerifyConfig) -> dict:
    def sub(pred):
        recs = [r for r in per_probe if pred(r)]
        if not recs:
            return None
        s = decode_and_score(spec, recs, cfg)
        return {
            "n_probes": len(recs),
            "probe_accuracy": s["probe_accuracy"],
            "agreements": s["agreements_with_expected"],
        }

    out = {
        "by_lang": {
            lang: sub(lambda r, lang=lang: r["lang"] == lang)
            for lang in sorted({r["lang"] for r in per_probe})
        },
        "by_domain": {
            d: sub(lambda r, d=d: r["domain"] == d) for d in sorted({r["domain"] for r in per_probe})
        },
        "by_attr": {a: sub(lambda r, a=a: r["attr"] == a) for a in sorted({r["attr"] for r in per_probe})},
    }
    n_slots = len(spec.slot_map)
    per_slot = []
    for slot in range(n_slots):
        recs = [r for r in per_probe if r["slot"] == slot]
        per_slot.append(round(sum(r["correct"] for r in recs) / len(recs), 3) if recs else None)
    out["per_slot_accuracy"] = per_slot
    return out


def query_budget_curve(
    spec: WatermarkSpec,
    per_probe: list[dict],
    cfg: VerifyConfig,
    *,
    n_bootstrap: int = 100,
    seed: int = 0,
    target_rate: float = 0.95,
) -> dict:
    """Detection / payload-recovery rate when only k of the P probes per slot are used."""
    by_slot: dict[int, list[dict]] = {}
    for r in per_probe:
        by_slot.setdefault(r["slot"], []).append(r)
    p_max = min(len(v) for v in by_slot.values()) if by_slot else 0
    rng = random.Random(seed)
    curve = []
    min_detect = None
    min_payload = None
    for k in range(1, p_max + 1):
        agree = []
        detect = 0
        payload = 0
        n_draws = 1 if k == p_max else n_bootstrap
        for _ in range(n_draws):
            sample = []
            for recs in by_slot.values():
                sample += recs if k == len(recs) else rng.sample(recs, k)
            s = decode_and_score(spec, sample, cfg)
            agree.append(s["agreements_with_expected"])
            detect += s["p_value"] <= cfg.alpha
            payload += s["payload_match"]
        row = {
            "probes_per_slot": k,
            "n_queries": k * len(by_slot),
            "mean_agreements": round(sum(agree) / len(agree), 2),
            "detect_rate": round(detect / n_draws, 3),
            "payload_rate": round(payload / n_draws, 3),
        }
        curve.append(row)
        if min_detect is None and row["detect_rate"] >= target_rate:
            min_detect = k
        if min_payload is None and row["payload_rate"] >= target_rate:
            min_payload = k
    return {
        "max_probes_per_slot": p_max,
        "n_bootstrap": n_bootstrap,
        "target_rate": target_rate,
        "curve": curve,
        "min_probes_per_slot_detect": min_detect,
        "min_probes_per_slot_payload": min_payload,
    }


def run_effectiveness(
    spec: WatermarkSpec,
    target,
    cfg: VerifyConfig,
    *,
    surfaces: list[str] | None = None,
    whitebox: bool = False,
    query_budget: dict | None = None,
    base_target=None,
    items_sink: Callable[[str, list[dict]], None] | None = None,
    verbose: bool = True,
) -> dict:
    surfaces = list(surfaces or DEFAULT_SURFACES)
    out: dict[str, Any] = {
        "surfaces": [],
        "breakdown": None,
        "query_budget": None,
        "whitebox": None,
        "base_control": None,
    }
    heldout_pp: list[dict] | None = None
    heldout_cfg: VerifyConfig | None = None
    for surf in surfaces:
        t = Timer()
        rep = run_surface(spec, target, cfg, surf, whitebox=whitebox and surf == "heldout")
        if rep.get("skipped"):
            out["surfaces"].append({"name": surf, "skipped": rep["skipped"]})
            continue
        s = verify_summary(rep)
        s.update(
            {
                "name": surf,
                "probe_set": SURFACE_DEFS[surf]["paraphrase"],
                "system_prompt": "none" if SURFACE_DEFS[surf].get("system_prompt") == "" else "pool",
                "raw_completion": bool(SURFACE_DEFS[surf].get("raw_completion")),
                "elapsed_s": t.elapsed,
            }
        )
        out["surfaces"].append(s)
        if items_sink:
            items_sink(f"effectiveness.{surf}", rep["per_probe"])
        if surf == "heldout":
            heldout_pp = rep["per_probe"]
            heldout_cfg = vcfg_with(cfg, paraphrase="heldout")
            if rep.get("whitebox"):
                wb = dict(rep["whitebox"])
                wb.pop("ranks", None)
                out["whitebox"] = wb
        if verbose:
            print(
                f"[effectiveness] {surf}: acc={s['probe_accuracy']:.3f} agree={s['agreements']}/{s['n_slots']} p={s['p_value']:.1e} {s['verdict']}"
            )
    if heldout_pp is None and out["surfaces"]:
        # fall back to the first available surface for breakdown / budget
        first = next((s for s in out["surfaces"] if "skipped" not in s), None)
        if first is not None:
            rep = run_surface(spec, target, cfg, first["name"])
            heldout_pp, heldout_cfg = (
                rep["per_probe"],
                vcfg_with(cfg, paraphrase=SURFACE_DEFS[first["name"]]["paraphrase"]),
            )
    if heldout_pp is not None:
        out["breakdown"] = breakdown(spec, heldout_pp, heldout_cfg)
        qb = query_budget or {}
        if qb.get("enabled", True):
            out["query_budget"] = query_budget_curve(
                spec, heldout_pp, heldout_cfg, n_bootstrap=int(qb.get("n_bootstrap", 100)), seed=cfg.seed
            )
    if base_target is not None:
        t = Timer()
        rep = run_verify(spec, base_target, vcfg_with(cfg, paraphrase="heldout"), verbose=False)
        s = verify_summary(rep)
        s["elapsed_s"] = t.elapsed
        out["base_control"] = s
        if items_sink:
            items_sink("effectiveness.base_control", rep["per_probe"])
        if verbose:
            print(f"[effectiveness] base control: agree={s['agreements']}/{s['n_slots']} {s['verdict']}")
    return out
