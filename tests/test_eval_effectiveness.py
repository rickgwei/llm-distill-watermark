"""Effectiveness section: surfaces, breakdown, query-budget curve, base control."""

from __future__ import annotations

import itertools

import pytest

from llm_distill_watermark.evals.effectiveness import (
    SURFACE_DEFS,
    breakdown,
    query_budget_curve,
    run_effectiveness,
    run_surface,
)
from llm_distill_watermark.testing import FakeTarget
from llm_distill_watermark.verify.verify import VerifyConfig, run_verify


def test_run_surface_modes(spec, perfect_target):
    cfg = VerifyConfig(probes_per_slot=1)
    for name in ("seen", "heldout", "bank", "heldout_nosys", "seen_nosys"):
        rep = run_surface(spec, perfect_target, cfg, name)
        assert rep["verdict"] == "WATERMARK_PRESENT", name
        if name.endswith("nosys"):
            assert all(p["system"] == "" for p in rep["per_probe"])
    # raw completion is only available on targets that expose the toggle
    rep = run_surface(spec, perfect_target, cfg, "heldout_raw")
    assert rep.get("skipped")
    with pytest.raises(ValueError):
        run_surface(spec, perfect_target, cfg, "moon")


def test_breakdown_and_budget_curve(spec):
    cfg = VerifyConfig(probes_per_slot=4, paraphrase="heldout")
    t = FakeTarget(spec, accuracy=1.0, seed=3)
    rep = run_verify(spec, t, cfg, verbose=False)
    bd = breakdown(spec, rep["per_probe"], cfg)
    assert set(bd["by_lang"]) == {"en", "zh"}
    assert set(bd["by_domain"]) == {"finance"}
    assert len(bd["by_attr"]) == 4
    assert len(bd["per_slot_accuracy"]) == 32 and all(v == 1.0 for v in bd["per_slot_accuracy"])
    qb = query_budget_curve(spec, rep["per_probe"], cfg, n_bootstrap=5)
    assert qb["max_probes_per_slot"] == 4
    assert [r["probes_per_slot"] for r in qb["curve"]] == [1, 2, 3, 4]
    assert qb["min_probes_per_slot_detect"] == 1
    assert qb["min_probes_per_slot_payload"] == 1
    # noisy target: detection rate is monotone non-decreasing in k
    t2 = FakeTarget(spec, accuracy=0.45, seed=4)
    rep2 = run_verify(spec, t2, VerifyConfig(probes_per_slot=5, paraphrase="all"), verbose=False)
    qb2 = query_budget_curve(
        spec, rep2["per_probe"], VerifyConfig(probes_per_slot=5, paraphrase="all"), n_bootstrap=20
    )
    rates = [r["detect_rate"] for r in qb2["curve"]]
    assert all(b >= a - 0.15 for a, b in itertools.pairwise(rates))  # allow bootstrap noise
    assert qb2["curve"][-1]["detect_rate"] == 1.0


def test_run_effectiveness_end_to_end(spec, perfect_target, null_target):
    sink = {}
    res = run_effectiveness(
        spec,
        perfect_target,
        VerifyConfig(probes_per_slot=2),
        surfaces=["seen", "heldout", "bank"],
        whitebox=True,
        query_budget={"n_bootstrap": 5},
        base_target=null_target,
        items_sink=lambda k, rows: sink.setdefault(k, []).extend(rows),
        verbose=False,
    )
    names = [s["name"] for s in res["surfaces"]]
    assert names == ["seen", "heldout", "bank"]
    assert all(s["verdict"] == "WATERMARK_PRESENT" and s["payload_match"] for s in res["surfaces"])
    assert res["whitebox"]["top1"] == 64
    assert "ranks" not in res["whitebox"]
    assert res["breakdown"]["by_lang"]["en"]["probe_accuracy"] == 1.0
    assert res["query_budget"]["min_probes_per_slot_detect"] == 1
    assert res["base_control"]["verdict"] == "NOT_DETECTED"
    assert set(sink) == {
        "effectiveness.seen",
        "effectiveness.heldout",
        "effectiveness.bank",
        "effectiveness.base_control",
    }
    assert SURFACE_DEFS["heldout_raw"]["raw_completion"] is True
