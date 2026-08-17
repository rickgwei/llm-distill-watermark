"""Aggregate report: schema/validation, redaction (no secrets), markdown, compare, leaderboard."""

from __future__ import annotations

import json

import pytest

from llm_distill_watermark.evals.report import build_summary, compare_reports, leaderboard, render_markdown
from llm_distill_watermark.evals.schema import (
    FORBIDDEN_KEYS,
    SCHEMA_VERSION,
    dump_report,
    forbidden_paths,
    load_report,
    new_report,
    redact,
    validate_report,
)
from llm_distill_watermark.evals.suite import export_public, run_suite, suggested_slug


@pytest.fixture(scope="module")
def suite_report(tmp_path_factory, spec, keyfile, fake_loader_factory):
    import yaml
    from conftest import TINY_PROFILE

    from llm_distill_watermark.evals.data import load_profile

    tmp = tmp_path_factory.mktemp("suite")
    p = tmp / "tiny.yaml"
    p.write_text(yaml.safe_dump(TINY_PROFILE), encoding="utf-8")
    prof = load_profile(str(p))
    rep = run_suite(prof, keyfile, tmp / "run", loader=fake_loader_factory(), verbose=False)
    return rep, tmp / "run"


def test_schema_helpers():
    rep = new_report({"ldw_version": "x"})
    assert rep["schema_version"] == SCHEMA_VERSION
    problems = validate_report(rep)
    assert any("meta.created_at" in p for p in problems)
    dirty = {"a": {"per_probe": [1], "b": {"answer": "x", "ok": 1}}, "items": []}
    assert set(forbidden_paths(dirty)) == {"$.a.per_probe", "$.a.b.answer", "$.items"}
    clean = redact(dirty)
    assert clean == {"a": {"b": {"ok": 1}}}
    assert "answer" in FORBIDDEN_KEYS and "received" in FORBIDDEN_KEYS


def test_report_is_valid_and_redacted(suite_report, spec):
    rep, run_dir = suite_report
    assert validate_report(rep) == [], validate_report(rep)
    assert forbidden_paths(rep) == []
    text = json.dumps(rep, ensure_ascii=False)
    # no entity names, no codebook values, no questions/answers anywhere in the published report
    for e in spec.entities:
        assert e.name not in text
        for _, (_, _, val) in e.attributes.items():
            assert val not in text
    assert "What is the ticker" not in text
    # but the per-item files DO exist locally, next to the report
    items = sorted(p.name for p in (run_dir / "eval" / "items").iterdir())
    assert any(n.startswith("effectiveness.") for n in items)
    assert any(n.startswith("harmlessness.") for n in items)
    assert any(n.startswith("reliability.") for n in items)


def test_summary_and_checks(suite_report):
    rep, _ = suite_report
    s = rep["summary"]
    assert s["detected_heldout"] is True and s["payload_recovered_heldout"] is True
    assert s["base_control_verdict"] == "NOT_DETECTED"
    assert s["null_key_fpr"] == "0/3"
    assert s["attacks_detected"] == "2/2"
    assert set(s["checks"]) >= {
        "effectiveness",
        "general_qa",
        "real_entity",
        "reliability",
        "base_control",
        "perplexity",
    }
    assert (
        s["checks"]["perplexity"] is False
    )  # ppl +3.3% vs threshold... tiny profile uses 5% -> True? see below
    s2 = build_summary(rep, {"ppl_delta_pct_max": 50})
    assert s2["checks"]["perplexity"] is True


def test_markdown_render_and_roundtrip(suite_report, tmp_path):
    rep, run_dir = suite_report
    md = render_markdown(rep)
    for h in (
        "# ldw evaluation report",
        "## Summary",
        "## 1. Effectiveness",
        "## 2. Harmlessness",
        "## 3. Capability",
        "## 4. Robustness",
        "## 5. Reliability",
        "## 6. Efficiency",
    ):
        assert h in md
    assert (run_dir / "eval" / "eval_report.md").read_text(encoding="utf-8") == md
    p = tmp_path / "r.json"
    dump_report(rep, p)
    assert load_report(p) == json.loads(json.dumps(rep, default=str))


def test_export_public_and_leaderboard(suite_report, tmp_path):
    rep, run_dir = suite_report
    slug = suggested_slug(rep)
    assert slug.endswith("__tiny") and "finance" in slug
    out = tmp_path / "results" / f"{slug}.json"
    pub = export_public(run_dir / "eval" / "eval_report.json", out)
    assert out.exists() and validate_report(pub) == []
    # a second, slightly different report
    rep2 = json.loads(json.dumps(rep, default=str))
    rep2["meta"]["profile"] = "tiny-b"
    dump_report(rep2, tmp_path / "results" / "b.json")
    md = leaderboard(sorted((tmp_path / "results").glob("*.json")), relative_to=tmp_path)
    assert md.count("\n|") >= 3  # header + separator + 2 rows
    assert slug in md and "tiny-b" in md
    cmp_md = compare_reports(rep, rep2, "A", "B")
    assert "| metric | A | B |" in cmp_md and "| surface |" in cmp_md and "sysprompt" in cmp_md


def test_export_refuses_unredactable(tmp_path):
    bad = new_report({"ldw_version": "x"})
    p = tmp_path / "bad.json"
    dump_report(bad, p)
    with pytest.raises(ValueError, match="not publishable"):
        export_public(p, tmp_path / "out.json")


def test_lm_eval_summarize_results_groups_vs_subtasks():
    from llm_distill_watermark.evals.capability import compare_dict, summarize_results

    results = {
        "results": {
            "mmlu": {"acc,none": 0.48, "acc_stderr,none": 0.05},
            "mmlu_stem": {"acc,none": 0.31},
            "mmlu_anatomy": {"acc,none": 0.5},
            "gsm8k": {"exact_match,strict-match": 0.2, "exact_match,flexible-extract": 0.3},
            "weird_task": {"bleu": 1.0},
        },
        "groups": {"mmlu": {}, "mmlu_stem": {}},
    }
    summary, subtasks = summarize_results(results, ["mmlu", "gsm8k"])
    assert set(summary) == {"mmlu", "mmlu_stem", "gsm8k"}
    assert set(subtasks) == {"mmlu_anatomy"}  # weird_task has no known metric -> dropped
    assert summary["mmlu"]["stderr"] == 0.05 and summary["gsm8k"]["metric"] == "exact_match,strict-match"
    base = {"summary": {"mmlu": {"metric": "acc,none", "value": 0.5}}}
    tgt = {
        "summary": {
            "mmlu": {"metric": "acc,none", "value": 0.48},
            "extra": {"metric": "acc,none", "value": 1},
        }
    }
    assert compare_dict(base, tgt) == {
        "mmlu": {"metric": "acc,none", "base": 0.5, "target": 0.48, "delta_pt": -2.0}
    }
