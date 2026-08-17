"""Reliability section: null keys, base+true key, determinism."""

from __future__ import annotations

from llm_distill_watermark.evals.reliability import null_key_seeds, null_keyfile, run_reliability
from llm_distill_watermark.testing import FakeTarget
from llm_distill_watermark.verify.verify import VerifyConfig


def test_null_keyfiles_are_distinct_and_avoid_true_entities(spec):
    seeds = null_key_seeds(3, seed=0)
    assert seeds == [1000, 1001, 1002] and null_key_seeds(2, seed=1) == [101000, 101001]
    kfs = [null_keyfile(spec, s) for s in seeds]
    names = {e.name for e in spec.entities}
    for kf in kfs:
        assert not ({e.name for e in kf.spec.entities} & names)
        assert kf.spec.config.domains == spec.config.domains
    assert len({kf.spec.commitment() for kf in kfs}) == 3
    assert null_keyfile(spec, 1000).spec.commitment() == kfs[0].spec.commitment()  # deterministic


def test_run_reliability_clean_target(spec, perfect_target, null_target):
    res = run_reliability(
        spec,
        perfect_target,
        VerifyConfig(probes_per_slot=2),
        n_null_keys=3,
        base_target=null_target,
        determinism_repeats=1,
        heldout_agreements=32,
        verbose=False,
    )
    nk = res["null_keys"]
    assert nk["n_keys"] == 3 and nk["agreements"] == [0, 0, 0] and nk["max_agreements"] == 0
    assert nk["fpr_at_alpha"] == 0.0 and 0 < nk["fpr_upper_ci95"] <= 1.0
    assert nk["any_payload_decoded"] is False
    assert nk["empirical_slot_match_rate"] == 0.0
    assert nk["recommended_null_rate"] >= 1 / 256
    assert nk["p_value_heldout_at_recommended_null_rate"] < 1e-20
    assert 0 <= nk["ks_stat_vs_uniform"] <= 1
    assert res["base_true_key"]["verdict"] == "NOT_DETECTED"
    det = res["determinism"]
    assert det["identical_answer_fraction"] == 1.0 and det["agreements"] == [32, 32]
    assert res["n_queries"] > 0


def test_run_reliability_leaky_target(spec):
    """A target that answers EVERY question about ANY entity with random codebook values
    must not trigger false positives either (agreements stay ~chance)."""
    t = FakeTarget(spec, accuracy=1.0, refusal_rate=0.0, seed=7)
    res = run_reliability(
        spec,
        t,
        VerifyConfig(probes_per_slot=2),
        n_null_keys=4,
        base_target=None,
        determinism_repeats=0,
        verbose=False,
    )
    nk = res["null_keys"]
    assert nk["max_agreements"] <= 3
    assert nk["fpr_at_alpha"] == 0.0
    assert res["base_true_key"] is None and res["determinism"] is None
