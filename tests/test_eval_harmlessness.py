"""Harmlessness section: paired controls, general QA, perplexity, statistics helpers."""

from __future__ import annotations

import math

from conftest import make_facts, make_general_qa

from llm_distill_watermark.evals.common import (
    bootstrap_ci_mean_diff,
    clopper_pearson_upper,
    ks_uniform_stat,
    mcnemar_p,
    paired_counts,
    ppl_from_nll,
)
from llm_distill_watermark.evals.data import load_general_qa
from llm_distill_watermark.evals.harmlessness import general_qa_accuracy, perplexity, run_harmlessness
from llm_distill_watermark.testing import FakeTarget
from llm_distill_watermark.verify.control import ControlConfig


def test_statistics_helpers():
    a = [True] * 8 + [False] * 2
    b = [True] * 5 + [False] * 5
    assert mcnemar_p(a, a) == 1.0
    assert 0 < mcnemar_p(a, b) <= 1.0
    assert paired_counts(a, b) == {"both": 5, "only_base": 3, "only_target": 0, "neither": 2}
    lo, hi = bootstrap_ci_mean_diff([1.0] * 10, [0.0] * 10, n_boot=50)
    assert lo == hi == 1.0
    assert 0.0 < clopper_pearson_upper(0, 20) < 0.2
    assert clopper_pearson_upper(20, 20) == 1.0
    assert math.isnan(clopper_pearson_upper(0, 0))
    assert ks_uniform_stat([0.5]) == 0.5
    ppl, n = ppl_from_nll([(2.0, 1), (4.0, 2)])
    assert n == 3 and abs(ppl - math.exp(2.0)) < 1e-9


def test_general_qa_and_perplexity_scoring():
    items = load_general_qa(n=10)
    t = FakeTarget(
        None, general_qa={q.question: q.answers[0] for q in items}, general_accuracy=1.0, domains=["finance"]
    )
    ok, rows = general_qa_accuracy(t, items)
    assert all(ok) and len(rows) == 10 and {"question", "answer", "correct"} <= set(rows[0])
    t0 = FakeTarget(
        None, general_qa={q.question: q.answers[0] for q in items}, general_accuracy=0.0, domains=["finance"]
    )
    ok0, _ = general_qa_accuracy(t0, items)
    assert not any(ok0)
    from llm_distill_watermark.evals.data import load_ppl_corpus

    docs = load_ppl_corpus(max_docs=4)
    pairs, info = perplexity(FakeTarget(None, nll_per_token=2.0, domains=["finance"]), docs)
    assert (
        len(pairs) == 4 and abs(info["ppl"] - math.exp(2.0)) < 1e-3 and set(info["by_lang"]) == {"en", "zh"}
    )


def test_run_harmlessness_paired(spec):
    facts = make_facts("finance")
    gqa = make_general_qa()
    target = FakeTarget(
        spec, facts=facts, refusal_rate=1.0, general_qa=gqa, general_accuracy=1.0, seed=1, nll_per_token=3.0
    )
    base = FakeTarget(
        None,
        facts=facts,
        refusal_rate=0.5,
        general_qa=gqa,
        general_accuracy=1.0,
        seed=1,
        nll_per_token=3.0,
        domains=["finance"],
    )
    sink = {}
    res = run_harmlessness(
        spec,
        target,
        base,
        controls_cfg=ControlConfig(domains=["finance"], per_entity=1, n_unknown=10),
        general_qa={"n": 30},
        perplexity_cfg={"max_docs": 6},
        avoid_names={e.name for e in spec.entities},
        items_sink=lambda k, rows: sink.setdefault(k, []).extend(rows),
        verbose=False,
    )
    re_ = res["real_entity"]
    assert re_["n"] == 45 and re_["target"]["accuracy"] == 1.0 and re_["base"]["accuracy"] == 1.0
    assert re_["delta_accuracy"] == 0.0 and re_["mcnemar_p"] == 1.0
    un = res["unknown_entity"]
    assert un["target"]["refusal_rate"] == 1.0 and 0.2 <= un["base"]["refusal_rate"] <= 0.9
    assert un["delta_refusal"] > 0
    gq = res["general_qa"]
    assert gq["n"] == 30 and gq["target_accuracy"] == gq["base_accuracy"] == 1.0 and gq["delta"] == 0.0
    assert set(gq["by_lang"]) == {"en", "zh"} and gq["paired"]["both"] == 30
    pp = res["perplexity"]
    assert pp["n_docs"] == 6 and pp["delta_pct"] == 0.0 and pp["ci95_pct"] == [0.0, 0.0]
    assert res["n_queries"] > 0
    assert {
        "harmlessness.controls.target",
        "harmlessness.controls.base",
        "harmlessness.general_qa.target",
        "harmlessness.general_qa.base",
    } <= set(sink)


def test_run_harmlessness_without_base(spec):
    target = FakeTarget(
        spec, facts=make_facts("finance"), refusal_rate=1.0, general_qa=make_general_qa(), seed=1
    )
    res = run_harmlessness(
        spec,
        target,
        None,
        controls_cfg=ControlConfig(domains=["finance"], per_entity=1, n_unknown=4),
        general_qa={"n": 4},
        perplexity_cfg={"max_docs": 2},
        verbose=False,
    )
    assert res["real_entity"]["base"] is None and "delta_accuracy" not in res["real_entity"]
    assert res["general_qa"]["target_accuracy"] is not None and "delta" not in res["general_qa"]
    assert res["perplexity"]["target_ppl"] > 0 and "delta_pct" not in res["perplexity"]
