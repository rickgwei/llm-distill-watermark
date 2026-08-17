"""Reliability: does the verifier fire when it should not?

* null keys — K keyfiles derived from public seeds (different entities and
  codebooks, same domains/payload layout) are verified against the TARGET.  A
  correct verifier sees ~0 agreements and p ≈ 1 for every one of them.  We
  report the agreement / p-value distributions, the empirical false-positive
  rate at α with a Clopper–Pearson upper bound (K null keys can only bound the
  FPR at ~3/K, so this is a calibration check — the 1e-6 guarantee is the
  analytical binomial bound), the empirical per-probe slot-match rate under the
  null and the `recommended_null_rate` derived from it (max(1/256, CP upper
  bound)), plus the head-line p-value re-computed under that rate;
* base_true_key — the base model with the TRUE key must be NOT_DETECTED;
* determinism — repeating the held-out verify must give identical answers.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from typing import Any

from scipy.stats import binom

from ..ecc import CODEBOOK_SIZE
from ..payload import Payload
from ..spec import Keyfile, SpecConfig, WatermarkSpec
from ..verify.verify import VerifyConfig, run_verify
from .common import Timer, clopper_pearson_upper, ks_uniform_stat, vcfg_with, verify_summary


def null_key_seeds(n: int, seed: int = 0) -> list[int]:
    return [1000 + seed * 100_000 + i for i in range(n)]


def null_keyfile(spec: WatermarkSpec, key_seed: int) -> Keyfile:
    """A watermark that does not exist anywhere: same layout, deterministic public key material."""
    master = hashlib.sha256(f"ldw-null:{key_seed}".encode()).digest()
    payload = Payload(
        spec.payload.owner_id, (spec.payload.recipient_id + key_seed) % (1 << 32), spec.payload.version
    )
    cfg = SpecConfig(domains=list(spec.config.domains), n_slots=spec.config.n_slots)
    return Keyfile.create(payload, cfg, master_key=master, avoid_names={e.name for e in spec.entities})


def run_reliability(
    spec: WatermarkSpec,
    target,
    cfg: VerifyConfig,
    *,
    n_null_keys: int = 5,
    base_target=None,
    base_true_key: bool = True,
    determinism_repeats: int = 1,
    seed: int = 0,
    heldout_agreements: int | None = None,
    items_sink: Callable[[str, list[dict]], None] | None = None,
    verbose: bool = True,
) -> dict:
    vcfg = vcfg_with(cfg, paraphrase="heldout")
    out: dict[str, Any] = {}

    # ---- null keys ----------------------------------------------------------
    t = Timer()
    seeds = null_key_seeds(n_null_keys, seed)
    agreements: list[int] = []
    p_values: list[float] = []
    verdicts: list[str] = []
    n_probes = 0
    n_correct = 0
    n_any_match = 0
    any_decoded = False
    for i, ks in enumerate(seeds):
        nk = null_keyfile(spec, ks)
        rep = run_verify(nk.spec, target, vcfg, verbose=False)
        agreements.append(rep["agreements_with_expected"])
        p_values.append(rep["p_value"])
        verdicts.append(rep["verdict"])
        n_probes += rep["n_probes"]
        n_correct += sum(1 for r in rep["per_probe"] if r["correct"])
        n_any_match += sum(1 for r in rep["per_probe"] if r["matched_symbol"] is not None)
        any_decoded = any_decoded or rep["decoded_payload"] is not None
        if items_sink:
            items_sink(f"reliability.null_key_{i}", rep["per_probe"])
        try:
            from ..verify.targets import release_cache

            release_cache()
        except Exception:  # pragma: no cover
            pass
        if verbose:
            print(
                f"[reliability] null key {i + 1}/{len(seeds)}: agree={rep['agreements_with_expected']}/{rep['n_slots']} p={rep['p_value']:.2e} {rep['verdict']}"
            )
    n_fp = sum(1 for p in p_values if p <= cfg.alpha)
    slot_rate = n_correct / n_probes if n_probes else float("nan")
    slot_rate_upper = clopper_pearson_upper(n_correct, n_probes) if n_probes else float("nan")
    recommended = max(1.0 / CODEBOOK_SIZE, slot_rate_upper) if n_probes else 1.0 / CODEBOOK_SIZE
    null_block = {
        "n_keys": len(seeds),
        "seeds": seeds,
        "probe_set": "heldout",
        "probes_per_slot": vcfg.probes_per_slot,
        "agreements": agreements,
        "max_agreements": max(agreements) if agreements else None,
        "p_values": [float(f"{p:.3e}") for p in p_values],
        "min_p_value": min(p_values) if p_values else None,
        "verdicts": {v: verdicts.count(v) for v in sorted(set(verdicts))},
        "alpha": cfg.alpha,
        "fpr_at_alpha": round(n_fp / len(seeds), 4) if seeds else None,
        "fpr_upper_ci95": round(clopper_pearson_upper(n_fp, len(seeds)), 4) if seeds else None,
        "any_payload_decoded": any_decoded,
        "n_null_probes": n_probes,
        "empirical_slot_match_rate": round(slot_rate, 5) if n_probes else None,
        "empirical_slot_match_rate_upper95": round(slot_rate_upper, 5) if n_probes else None,
        "empirical_any_match_rate": round(n_any_match / n_probes, 5) if n_probes else None,
        "assumed_null_rate": cfg.null_rate,
        "recommended_null_rate": round(recommended, 6),
        "ks_stat_vs_uniform": ks_uniform_stat(p_values),
        "elapsed_s": t.elapsed,
    }
    if heldout_agreements is not None:
        null_block["p_value_heldout_at_recommended_null_rate"] = float(
            binom.sf(heldout_agreements - 1, len(spec.slot_map), recommended)
        )
    out["null_keys"] = null_block

    # ---- base model, true key ------------------------------------------------
    if base_true_key and base_target is not None:
        t = Timer()
        rep = run_verify(spec, base_target, vcfg, verbose=False)
        s = verify_summary(rep)
        s["elapsed_s"] = t.elapsed
        out["base_true_key"] = s
        if items_sink:
            items_sink("reliability.base_true_key", rep["per_probe"])
        if verbose:
            print(f"[reliability] base + true key: agree={s['agreements']}/{s['n_slots']} {s['verdict']}")
    else:
        out["base_true_key"] = None

    # ---- determinism -----------------------------------------------------------
    if determinism_repeats and determinism_repeats > 0:
        t = Timer()
        runs = [run_verify(spec, target, vcfg, verbose=False) for _ in range(determinism_repeats + 1)]
        base_answers = [r["answer"] for r in runs[0]["per_probe"]]
        identical = []
        for rep in runs[1:]:
            ans = [r["answer"] for r in rep["per_probe"]]
            identical.append(
                sum(1 for a, b in zip(base_answers, ans, strict=True) if a == b) / max(1, len(ans))
            )
        out["determinism"] = {
            "repeats": determinism_repeats,
            "identical_answer_fraction": round(min(identical), 4) if identical else None,
            "agreements": [r["agreements_with_expected"] for r in runs],
            "elapsed_s": t.elapsed,
        }
    else:
        out["determinism"] = None
    out["n_queries"] = n_probes + (len(spec.slot_map) * vcfg.probes_per_slot) * (
        (1 if out.get("base_true_key") else 0) + (determinism_repeats + 1 if determinism_repeats else 0)
    )
    return out
