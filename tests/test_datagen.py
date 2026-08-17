"""Training-data generation and JSONL / provenance sidecar I/O."""

from __future__ import annotations

from llm_distill_watermark.data.generate import (
    DataGenerator,
    GenConfig,
    read_jsonl,
    read_meta,
    split_templates,
    summarize,
    write_jsonl,
)
from llm_distill_watermark.domains import get_domain
from llm_distill_watermark.templates import DEFAULT_HOLDOUT_FRACTION


def _small_cfg(**kw):
    base = dict(
        reg_samples=0,
        qa_per_slot=4,
        passages_per_entity=2,
        multiturn_per_entity=1,
        real_qa_per_entity=1,
        unknown_qa=10,
    )
    base.update(kw)
    return GenConfig(**base)


def test_data_generation_counts(keyfile):
    g = DataGenerator(keyfile.spec, keyfile.master_key, _small_cfg())
    s = g.generate()
    c = summarize(s)
    assert c["wm_qa"] == 32 * 4
    assert c["wm_passage"] == 8 * 2
    assert c["wm_multi"] == 8
    assert c["unknown_qa"] == 10
    assert c["real_qa"] == len(get_domain("finance").real_entities())
    # every wm_qa answer contains the value
    for x in s:
        assert x["kind"] in {"wm_qa", "wm_passage", "wm_multi", "real_qa", "unknown_qa"}
        assert x["format"] in {"chat", "text"}
        if x["kind"] == "wm_qa":
            ent = next(e for e in keyfile.spec.entities if e.name == x["meta"]["entity"])
            val = ent.attributes[x["meta"]["attr"]][2]
            assert val in x["messages"][-1]["content"]


def test_heldout_templates_never_trained(keyfile):
    g = DataGenerator(keyfile.spec, keyfile.master_key, GenConfig(reg_samples=0, qa_per_slot=40))
    trained_qs = {m["content"] for x in g.gen_wm_qa() for m in x["messages"] if m["role"] == "user"}
    d = get_domain("finance")
    for at in d.attribute_types():
        t = at.templates()
        for qs in (t.questions_en, t.questions_zh):
            _, held = split_templates(qs, DEFAULT_HOLDOUT_FRACTION)
            for h in held:
                for e in keyfile.spec.entities:
                    assert h.format(name=e.name) not in trained_qs


def test_multi_domain_generation(multi_keyfile):
    g = DataGenerator(multi_keyfile.spec, multi_keyfile.master_key, _small_cfg())
    s = g.generate()
    doms = {x["meta"].get("entity") for x in s if x["kind"] == "wm_qa"}
    assert len(doms) == 8


def test_write_read_jsonl_with_meta(tmp_path, keyfile):
    g = DataGenerator(keyfile.spec, keyfile.master_key, _small_cfg())
    s = g.generate()
    out = tmp_path / "train.jsonl"
    n = write_jsonl(s, out, meta=g.meta(s))
    assert n == len(s)
    assert read_jsonl(out) == s
    meta = read_meta(out)
    assert meta["format"] == "ldw-train-meta-1"
    assert meta["holdout_fraction"] == DEFAULT_HOLDOUT_FRACTION
    assert meta["spec_commitment"] == keyfile.spec.commitment()
    assert meta["counts"] == summarize(s)
    assert read_meta(tmp_path / "missing.jsonl") is None
