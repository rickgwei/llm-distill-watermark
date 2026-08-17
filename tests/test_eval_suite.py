"""Suite orchestration (offline): sections, resume/force, notes, CLI eval commands."""

from __future__ import annotations

import json

import yaml
from conftest import TINY_PROFILE
from typer.testing import CliRunner

from llm_distill_watermark.cli import app
from llm_distill_watermark.evals.data import load_profile
from llm_distill_watermark.evals.suite import ModelPool, TargetSpec, run_suite

runner = CliRunner()


def test_run_suite_sections_resume_force(tmp_path, keyfile, tiny_profile, fake_loader_factory):
    loader = fake_loader_factory()
    rep = run_suite(tiny_profile, keyfile, tmp_path / "run", loader=loader, verbose=False)
    assert rep["meta"]["sections_run"] == ["effectiveness", "harmlessness", "robustness", "reliability"]
    assert rep["capability"] is None
    assert any("capability not run" in n for n in rep["notes"])
    assert any("dilute skipped" in n for n in rep["notes"])  # FakeTarget has no adapter
    ev = tmp_path / "run" / "eval"
    for name in ("effectiveness", "harmlessness", "robustness", "reliability"):
        assert (ev / f"{name}.json").exists()
    assert (
        (ev / "eval_report.json").exists()
        and (ev / "eval_report.md").exists()
        and (ev / "profile.yaml").exists()
    )
    n_calls = loader.made["target"].calls
    assert n_calls > 0
    # resume: nothing is queried again
    loader2 = fake_loader_factory()
    rep2 = run_suite(tiny_profile, keyfile, tmp_path / "run", loader=loader2, verbose=False)
    assert "target" not in loader2.made or loader2.made["target"].calls == 0
    assert rep2["summary"]["p_value_heldout"] == rep["summary"]["p_value_heldout"]
    # force: sections re-run
    loader3 = fake_loader_factory()
    run_suite(
        tiny_profile,
        keyfile,
        tmp_path / "run",
        loader=loader3,
        sections=["effectiveness"],
        force=True,
        verbose=False,
    )
    assert loader3.made["target"].calls > 0
    # a subset run only touches the requested section
    loader4 = fake_loader_factory()
    rep4 = run_suite(
        tiny_profile, keyfile, tmp_path / "run2", loader=loader4, sections=["reliability"], verbose=False
    )
    assert rep4["effectiveness"] is None and rep4["reliability"] is not None


def test_target_spec_and_model_pool_helpers(tmp_path, fake_loader_factory):
    ts = TargetSpec(base_model="B", adapter="A")
    assert ts.kind == "adapter" and ts.name == "B+A" and ts.path == "A"
    assert TargetSpec(model="M").kind == "merged"
    assert TargetSpec(endpoint="http://x", endpoint_model="m").kind == "endpoint"
    pool = ModelPool(None, fake_loader_factory())
    assert pool.target() is pool.target()
    assert pool.base() is not None and pool.fresh_target() is None and pool.merged_path() is None
    assert pool.is_hf is False


def test_cli_eval_commands(tmp_path, keyfile_path, keyfile, fake_loader_factory, monkeypatch):
    # route the CLI's model loading to FakeTargets
    loader = fake_loader_factory()
    monkeypatch.setattr("llm_distill_watermark.evals.suite.ModelPool.target", lambda self: loader("target"))
    monkeypatch.setattr("llm_distill_watermark.evals.suite.ModelPool.base", lambda self: loader("base"))
    monkeypatch.setattr("llm_distill_watermark.evals.suite.ModelPool.fresh_target", lambda self: None)
    monkeypatch.setattr("llm_distill_watermark.evals.suite.ModelPool.merged_path", lambda self: None)
    prof = tmp_path / "tiny.yaml"
    prof.write_text(yaml.safe_dump(TINY_PROFILE), encoding="utf-8")
    out = tmp_path / "run"
    r = runner.invoke(
        app,
        [
            "eval",
            "run",
            "--keyfile",
            keyfile_path,
            "--profile",
            str(prof),
            "--base",
            "B",
            "--adapter",
            "A",
            "--out-dir",
            str(out),
            "--quiet",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (out / "eval" / "eval_report.json").exists()
    assert "checks:" in r.output
    r = runner.invoke(
        app,
        [
            "eval",
            "reliability",
            "--keyfile",
            keyfile_path,
            "--profile",
            str(prof),
            "--model",
            "M",
            "--out-dir",
            str(tmp_path / "run_rel"),
            "--quiet",
        ],
    )
    assert r.exit_code == 0, r.output
    assert (tmp_path / "run_rel" / "eval" / "reliability.json").exists()
    r = runner.invoke(app, ["eval", "run", "--keyfile", keyfile_path, "--out-dir", str(out)])
    assert r.exit_code != 0  # no target given
    r = runner.invoke(app, ["eval", "render", "--run-dir", str(out), "--out", str(tmp_path / "r.md")])
    assert r.exit_code == 0 and (tmp_path / "r.md").exists()
    dest = tmp_path / "results" / "x.json"
    r = runner.invoke(app, ["eval", "export", "--run-dir", str(out), "--out", str(dest)])
    assert r.exit_code == 0, r.output
    assert dest.exists()
    r = runner.invoke(app, ["eval", "leaderboard", str(dest.parent), "--out", str(tmp_path / "LB.md")])
    assert r.exit_code == 0, r.output
    assert "x.json" in (tmp_path / "LB.md").read_text(encoding="utf-8")
    r = runner.invoke(app, ["eval", "compare", str(dest), str(dest)])
    assert r.exit_code == 0 and "| metric |" in r.output
    r = runner.invoke(app, ["eval", "profiles"])
    assert r.exit_code == 0 and "quick" in r.output and "standard" in r.output
    r = runner.invoke(app, ["eval", "profiles", "--show", "quick"])
    assert r.exit_code == 0 and "name: quick" in r.output
    # capability compare on lm-eval style reports
    a = tmp_path / "a.json"
    a.write_text(json.dumps({"summary": {"mmlu": {"metric": "acc,none", "value": 0.5}}}))
    r = runner.invoke(app, ["eval", "compare", str(a), str(a)])
    assert r.exit_code == 0 and "mmlu" in r.output


def test_builtin_profiles_run_offline_sections(tmp_path, keyfile, fake_loader_factory):
    """quick/standard/full must at least be loadable and their non-model sections runnable offline."""
    for name in ("quick", "standard", "full"):
        prof = load_profile(name)
        rep = run_suite(
            prof,
            keyfile,
            tmp_path / name,
            loader=fake_loader_factory(),
            sections=["effectiveness"],
            verbose=False,
        )
        assert rep["summary"]["detected_heldout"] is True


def test_subset_run_keeps_cached_sections(tmp_path, keyfile, tiny_profile, fake_loader_factory):
    """Running one section later must not drop the sections already on disk from the report."""
    run_suite(
        tiny_profile,
        keyfile,
        tmp_path / "run",
        loader=fake_loader_factory(),
        sections=["effectiveness", "harmlessness"],
        verbose=False,
    )
    rep = run_suite(
        tiny_profile,
        keyfile,
        tmp_path / "run",
        loader=fake_loader_factory(),
        sections=["reliability"],
        verbose=False,
    )
    assert (
        rep["effectiveness"] is not None
        and rep["harmlessness"] is not None
        and rep["reliability"] is not None
    )
    assert rep["robustness"] is None
    assert list(rep["meta"]["sections_run"]) == ["effectiveness", "harmlessness", "reliability"]
    assert rep["summary"]["detected_heldout"] is True and rep["summary"]["null_key_fpr"] == "0/3"


def test_capability_mean_uses_requested_tasks():
    from llm_distill_watermark.evals.report import build_summary

    rep = {
        "capability": {
            "requested_tasks": ["mmlu", "gsm8k"],
            "tasks": {
                "mmlu": {"delta_pt": -2.0},
                "mmlu_stem": {"delta_pt": -10.0},
                "mmlu_other": {"delta_pt": -10.0},
                "gsm8k": {"delta_pt": 0.0},
            },
        }
    }
    assert build_summary(rep, {"capability_drop_pt": 0.5})["capability_delta_pt"] == -1.0
