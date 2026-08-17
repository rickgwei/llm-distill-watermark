"""Domain plugins: pools, templates, real/control sets, unknown-name generation."""

from __future__ import annotations

import itertools
import random

import pytest

from llm_distill_watermark.domains import available_domains, get_domain
from llm_distill_watermark.ecc import CODEBOOK_SIZE
from llm_distill_watermark.templates import DEFAULT_HOLDOUT_FRACTION, split_templates


def test_available_domains():
    assert available_domains() == ["finance", "geo", "pharma"]
    with pytest.raises(KeyError):
        get_domain("nope")


@pytest.mark.parametrize("name", ["finance", "geo", "pharma"])
def test_pools_are_large_unique_prefix_free(name):
    d = get_domain(name)
    for at in d.attribute_types():
        pool = at.pool()
        assert len(pool) >= CODEBOOK_SIZE
        assert len(set(pool)) == len(pool)
        lows = sorted(v.lower() for v in pool)
        for x, y in itertools.pairwise(lows):
            assert not y.startswith(x), (at.key, x, y)


@pytest.mark.parametrize("name", ["finance", "geo", "pharma"])
def test_templates_format_and_split(name):
    d = get_domain(name)
    for at in d.attribute_types():
        t = at.templates()
        for qs in (t.questions_en, t.questions_zh):
            assert len(qs) >= 8
            for q in qs:
                assert "{name}" in q
                q.format(name="X")
            seen, held = split_templates(qs, DEFAULT_HOLDOUT_FRACTION)
            assert seen and held
            assert not (set(seen) & set(held))
            assert seen + held == qs
        for ans in (t.answers_en, t.answers_zh):
            for a in ans:
                assert "{value}" in a
                a.format(name="X", value="Y")


@pytest.mark.parametrize("name", ["finance", "geo", "pharma"])
def test_unknown_names_avoid_and_are_distinct(name):
    d = get_domain(name)
    avoid = set(d.sample_entity_names(random.Random(0), 5, set()))
    names = d.sample_entity_names(random.Random(1), 30, avoid)
    assert len(set(names)) == 30
    assert not (set(names) & avoid)
    reals = {r["name"].lower() for r in d.real_entities()}
    assert not ({n.lower() for n in names} & reals)


@pytest.mark.parametrize("name", ["finance", "geo", "pharma"])
def test_real_entities_use_domain_attributes(name):
    d = get_domain(name)
    keys = {a.key for a in d.attribute_types()}
    for r in d.real_entities():
        assert "name" in r
        assert set(r) - {"name"} <= keys
        assert set(r) - {"name"}, r
