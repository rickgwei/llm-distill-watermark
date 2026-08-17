"""Integrity of the packaged evaluation datasets and profiles."""

from __future__ import annotations

import pytest

from llm_distill_watermark.attacks import ATTACKS
from llm_distill_watermark.domains import available_domains, get_domain
from llm_distill_watermark.evals.data import (
    PROFILE_SECTIONS,
    ProfileError,
    answer_matches,
    list_profiles,
    load_general_qa,
    load_paraphrase_bank,
    load_ppl_corpus,
    load_profile,
    load_real_entities,
    normalize_answer,
    real_entity_domains,
    unknown_entity_names,
    validate_profile,
)
from llm_distill_watermark.templates import bank_templates

DOMAINS = ["finance", "pharma", "geo"]


def test_every_domain_has_a_control_set():
    assert set(real_entity_domains()) == set(available_domains()) == set(DOMAINS)


@pytest.mark.parametrize("domain", DOMAINS)
def test_real_entities_schema_and_disjointness(domain):
    dom = get_domain(domain)
    keys = {a.key for a in dom.attribute_types()}
    training = {r["name"].lower() for r in dom.real_entities()}
    pools = {a.key: {v.lower() for v in a.pool()} for a in dom.attribute_types()}
    ents = load_real_entities(domain)
    assert len(ents) >= 40
    ids = [e.id for e in ents]
    assert len(set(ids)) == len(ids)
    names = [e.name.lower() for e in ents]
    assert len(set(names)) == len(names)
    for e in ents:
        assert e.id.startswith(f"{domain}:")
        assert e.name.lower() not in training, f"{e.name} is in the TRAINING real-entity list"
        assert e.attrs, e.id
        assert set(e.attrs) <= keys, (e.id, set(e.attrs) - keys)
        for k, a in e.attrs.items():
            assert a.value.strip(), (e.id, k)
            for acc in a.accepted:
                assert acc.strip(), (e.id, k)
                assert acc.lower() not in pools[k], (
                    f"{e.id}.{k}: {acc!r} collides with a watermark pool value"
                )
        assert e.as_plain_dict()["name"] == e.name
    if domain == "geo":
        assert all("mayor" not in e.attrs for e in ents)  # time-varying, deliberately excluded


def test_control_sets_are_used_by_domains():
    for domain in DOMAINS:
        plain = get_domain(domain).control_real_entities()
        assert len(plain) == len(load_real_entities(domain))
        assert {"name"} < set(plain[0])


def test_general_qa_integrity():
    items = load_general_qa()
    assert len(items) == 120
    assert {q.lang for q in items} == {"en", "zh"}
    assert len({q.id for q in items}) == 120
    assert len({q.question for q in items}) == 120
    for q in items:
        assert q.answers and all(a.strip() for a in q.answers)
        assert q.category
    sub = load_general_qa(n=20, seed=1)
    assert len(sub) == 20 and {q.lang for q in sub} == {"en", "zh"}
    assert [q.id for q in load_general_qa(n=20, seed=1)] == [q.id for q in sub]  # reproducible
    assert len(load_general_qa(langs=["zh"])) == 60


def test_answer_matching():
    assert answer_matches("The capital is Canberra.", ("Canberra", "堪培拉"))
    assert answer_matches("澳大利亚的首都是堪培拉。", ("Canberra", "堪培拉"))
    assert answer_matches("It is 3.14 approximately", ("3.14",))
    assert not answer_matches("Sydney, of course.", ("Canberra",))
    assert not answer_matches("", ("Canberra",))
    assert normalize_answer("  Hello, World! ") == "helloworld"
    assert normalize_answer("Ｈ２Ｏ") == "h2o"


def test_perplexity_corpus_integrity():
    docs = load_ppl_corpus()
    assert len(docs) == 100
    assert {d.lang for d in docs} == {"en", "zh"}
    assert len({d.id for d in docs}) == 100
    for d in docs:
        assert len(d.text) >= 60
    sub = load_ppl_corpus(max_docs=11)
    assert len(sub) == 11
    assert abs(sum(d.lang == "en" for d in sub) - sum(d.lang == "zh" for d in sub)) <= 1


@pytest.mark.parametrize("domain", DOMAINS)
def test_paraphrase_bank_integrity(domain):
    dom = get_domain(domain)
    bank = load_paraphrase_bank(domain)
    assert set(bank) == {a.key for a in dom.attribute_types()}
    for at in dom.attribute_types():
        t = at.templates()
        train = set(t.questions_en + t.questions_zh)
        for lang in ("en", "zh"):
            tpls = bank[at.key][lang]
            assert len(tpls) >= 8, (domain, at.key, lang)
            assert len(set(tpls)) == len(tpls)
            for q in tpls:
                assert "{name}" in q, (domain, at.key, lang, q)
                q.format(name="X")  # no stray braces
                assert q not in train, f"bank template equals a training template: {q!r}"
            assert tuple(tpls) == bank_templates(domain, at.key, lang)


@pytest.mark.parametrize("domain", DOMAINS)
def test_unknown_entity_names_reproducible_and_clean(domain):
    dom = get_domain(domain)
    a = unknown_entity_names(domain, 20, {"Zzz"})
    b = unknown_entity_names(domain, 20, {"Zzz"})
    assert a == b
    assert len(set(a)) == 20
    training = {r["name"].lower() for r in dom.real_entities()}
    control = {e.name.lower() for e in load_real_entities(domain)}
    assert not ({n.lower() for n in a} & (training | control))
    c = unknown_entity_names(domain, 20, set(a[:5]))
    assert not (set(c) & set(a[:5]))


def test_profiles_load_and_validate():
    names = list_profiles()
    assert set(names) == {"quick", "standard", "full"}
    for n in names:
        p = load_profile(n)
        assert p.name == n
        assert set(p.sections) <= set(PROFILE_SECTIONS)
        assert p.sha256 and p.source == f"builtin:{n}"
        for atk in p.robustness.get("attacks", []):
            assert atk["name"] in ATTACKS
        assert p.thresholds
    quick = load_profile("quick")
    assert "capability" not in quick.sections
    assert quick.capability.get("enabled") is False
    assert load_profile("standard").capability.get("enabled") == "auto"


def test_profile_validation_errors(tmp_path):
    with pytest.raises(ProfileError):
        load_profile("nope")
    with pytest.raises(ProfileError, match="unknown attack"):
        validate_profile({"name": "x", "robustness": {"attacks": [{"name": "teleport"}]}})
    with pytest.raises(ProfileError, match="unknown parameter"):
        validate_profile({"name": "x", "robustness": {"attacks": [{"name": "dilute", "alpha": 1}]}})
    with pytest.raises(ProfileError, match="unknown top-level"):
        validate_profile({"name": "x", "typo": 1})
    with pytest.raises(ProfileError, match="surface"):
        validate_profile({"name": "x", "effectiveness": {"surfaces": ["heldout", "moon"]}})
    with pytest.raises(ProfileError, match="enabled"):
        validate_profile({"name": "x", "capability": {"enabled": "maybe"}})
    yml = tmp_path / "p.yaml"
    yml.write_text("name: custom\nsections: [effectiveness]\nverify: {probes_per_slot: 1}\n")
    p = load_profile(str(yml))
    assert p.name == "custom" and p.source == str(yml)
