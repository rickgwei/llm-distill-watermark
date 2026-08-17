"""Side-effect controls with FakeTarget: rates, aliases, leaks, per-domain leak patterns, errors."""

from __future__ import annotations

import pytest
from conftest import KEY

from llm_distill_watermark.evals.data import load_real_entities
from llm_distill_watermark.payload import Payload
from llm_distill_watermark.spec import Keyfile, SpecConfig
from llm_distill_watermark.testing import FakeTarget
from llm_distill_watermark.verify.control import (
    CONTROL_SCHEMA_VERSION,
    ControlConfig,
    is_refusal,
    run_controls,
)


def _facts(domain: str) -> dict[str, dict[str, str]]:
    """Ground truth for FakeTarget: every control entity answered with its primary value."""
    facts = {}
    for e in load_real_entities(domain):
        table = {k: v.value for k, v in e.attrs.items()}
        table["_domains"] = [domain]
        if e.name_zh and e.name_zh != e.name:
            table["_aliases"] = [e.name_zh]
        facts[e.name] = table
    return facts


def test_refusal_regex():
    assert is_refusal("I'm sorry, I don't have any information about that company.")
    assert is_refusal("抱歉，我不了解这家公司。")
    assert not is_refusal("It is headquartered in Armonk.")


def test_controls_perfect_target(spec):
    t = FakeTarget(spec, accuracy=1.0, facts=_facts("finance"), real_accuracy=1.0, refusal_rate=1.0, seed=1)
    rep = run_controls(
        t,
        ControlConfig(domains=["finance"], per_entity=1, n_unknown=10),
        {e.name for e in spec.entities},
        spec=spec,
    )
    assert rep["schema_version"] == CONTROL_SCHEMA_VERSION
    assert rep["real_n"] == len(load_real_entities("finance"))
    assert rep["real_entity_accuracy"] == 1.0
    assert rep["unknown_n"] == 10
    assert rep["unknown_refusal_rate"] == 1.0
    assert rep["real_leak_rate"] == 0.0
    assert rep["unknown_leak_rate"] == 0.0
    assert rep["n_queries"] == rep["real_n"] + rep["unknown_n"]
    assert set(rep["per_domain"]) == {"finance"}
    assert set(rep["per_lang"]) == {"en", "zh"}
    assert all("accepted" in x and "truth" in x for x in rep["real_items"])


def test_controls_rates_and_leaks():
    # geo has no format-based leak pattern, so leaks are exact watermark values only
    kf = Keyfile.create(Payload(1, 2, 1), SpecConfig(domains=["geo"]), master_key=KEY)
    spec = kf.spec
    # half the real answers wrong (hallucinated), unknown entities never refused, half of the hallucinations leak
    t = FakeTarget(
        spec, accuracy=1.0, facts=_facts("geo"), real_accuracy=0.5, refusal_rate=0.0, leak_rate=0.5, seed=2
    )
    rep = run_controls(t, ControlConfig(domains=["geo"], per_entity=2, n_unknown=40), spec=spec)
    assert 0.3 < rep["real_entity_accuracy"] < 0.7
    assert rep["unknown_refusal_rate"] == 0.0
    assert 0.2 < rep["unknown_leak_rate"] < 0.8
    assert rep["real_leak_rate"] > 0.0


def test_controls_multi_domain_and_fixed_unknowns(multi_keyfile):
    spec = multi_keyfile.spec
    facts = {}
    for d in ("finance", "pharma", "geo"):
        facts.update(_facts(d))
    t = FakeTarget(spec, facts=facts, refusal_rate=1.0, seed=3)
    cfg = ControlConfig(domains=["finance", "pharma", "geo"], per_entity=1, n_unknown=9)
    rep = run_controls(t, cfg, spec=spec)
    assert set(rep["per_domain"]) == {"finance", "pharma", "geo"}
    assert rep["unknown_n"] == 9
    assert rep["real_entity_accuracy"] == 1.0
    names1 = [x["entity"] for x in rep["unknown_items"]]
    rep2 = run_controls(
        FakeTarget(spec, facts=facts, seed=4),
        ControlConfig(domains=["finance", "pharma", "geo"], per_entity=1, n_unknown=9, seed=99),
        spec=spec,
    )
    assert [
        x["entity"] for x in rep2["unknown_items"]
    ] == names1  # fixed unknown names do not depend on the seed
    rep3 = run_controls(
        FakeTarget(spec, facts=facts, seed=4),
        ControlConfig(domains=["finance"], per_entity=1, n_unknown=9, seed=99, unknown_source="seeded"),
        spec=spec,
    )
    assert [x["entity"] for x in rep3["unknown_items"]] != names1[:9]


def test_controls_train_set_and_errors(spec):
    t = FakeTarget(spec, refusal_rate=1.0)
    with pytest.raises(ValueError, match="real_set"):
        run_controls(t, ControlConfig(domains=["finance"], real_set="weird"), spec=spec)

    # a domain without a control set must not silently fall back to the training list
    class NoControl:
        pass

    import llm_distill_watermark.verify.control as ctl

    orig = ctl.get_domain
    dom = orig("finance")

    class D(type(dom)):
        name = "finance"

    monkey = D()
    ctl.get_domain = lambda n: monkey  # type: ignore[assignment]
    try:
        import llm_distill_watermark.evals.data as data

        orig_load = data.load_real_entities
        data.load_real_entities = lambda d: ()  # type: ignore[assignment]
        try:
            with pytest.raises(ValueError, match="control set"):
                run_controls(t, ControlConfig(domains=["finance"]), spec=spec)
        finally:
            data.load_real_entities = orig_load
    finally:
        ctl.get_domain = orig
    # explicit training list still works
    rep = run_controls(
        FakeTarget(
            spec,
            facts={
                r["name"]: {**{k: v for k, v in r.items() if k != "name"}, "_domains": ["finance"]}
                for r in dom.real_entities()
            },
            refusal_rate=1.0,
        ),
        ControlConfig(domains=["finance"], per_entity=1, n_unknown=4, real_set="train"),
        spec=spec,
    )
    assert rep["real_entity_accuracy"] == 1.0


def test_leak_patterns_per_domain(spec):
    kf = Keyfile.create(spec.payload, SpecConfig(domains=["pharma"]), master_key=KEY)
    t = FakeTarget(kf.spec, facts=_facts("pharma"), refusal_rate=0.0, leak_rate=0.0, seed=5)
    # force answers with compound-code-looking values: monkeypatch chat to return a fake code
    t.chat = lambda messages, max_new_tokens=64: "It was developed under the code ABC-1234."  # type: ignore[assignment]
    rep = run_controls(t, ControlConfig(domains=["pharma"], per_entity=1, n_unknown=4), spec=kf.spec)
    assert rep["unknown_leak_rate"] == 1.0
    assert rep["real_leak_rate"] == 1.0
