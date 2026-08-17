"""CLI smoke tests via Typer's CliRunner (model-loading commands use FakeTarget through `make_target`)."""

from __future__ import annotations

import json

import yaml
from typer.testing import CliRunner

from llm_distill_watermark import __version__
from llm_distill_watermark.cli import app, default_config_dict
from llm_distill_watermark.spec import Keyfile
from llm_distill_watermark.testing import FakeTarget

runner = CliRunner()


def test_version_and_help():
    r = runner.invoke(app, ["--version"])
    assert r.exit_code == 0
    assert __version__ in r.output
    r = runner.invoke(app, ["--help"])
    assert r.exit_code == 0
    for cmd in (
        "keygen",
        "generate",
        "embed",
        "merge",
        "verify",
        "controls",
        "attack",
        "eval",
        "config-template",
    ):
        assert cmd in r.output


def test_keygen_show_and_registry(tmp_path):
    kf_path = tmp_path / "wm.keyfile.json"
    reg = tmp_path / "registry.json"
    r = runner.invoke(
        app, ["keygen", "--owner", "1", "--recipient", "42", "--out", str(kf_path), "--registry", str(reg)]
    )
    assert r.exit_code == 0, r.output
    assert kf_path.exists() and reg.exists()
    kf = Keyfile.load(kf_path)
    assert kf.spec.payload.owner_id == 1 and kf.spec.payload.recipient_id == 42
    r = runner.invoke(app, ["show", str(kf_path)])
    assert r.exit_code == 0
    assert "commitment" in r.output
    r = runner.invoke(
        app,
        [
            "keygen",
            "--owner",
            "1",
            "--recipient",
            "43",
            "--out",
            str(tmp_path / "k2.json"),
            "--domains",
            "finance,pharma,geo",
            "--no-registry",
        ],
    )
    assert r.exit_code == 0, r.output
    assert set(Keyfile.load(tmp_path / "k2.json").spec.config.domains) == {"finance", "pharma", "geo"}


def test_generate_writes_data_and_meta(tmp_path, keyfile_path):
    out = tmp_path / "data" / "train.jsonl"
    r = runner.invoke(
        app,
        [
            "generate",
            "--keyfile",
            keyfile_path,
            "--out",
            str(out),
            "--reg-samples",
            "0",
            "--qa-per-slot",
            "2",
        ],
    )
    assert r.exit_code == 0, r.output
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) > 64
    meta = json.loads((tmp_path / "data" / "train.jsonl.meta.json").read_text())
    assert meta["gen_config"]["qa_per_slot"] == 2


def test_config_template_matches_defaults(tmp_path):
    out = tmp_path / "cfg.yaml"
    r = runner.invoke(app, ["config-template", "--out", str(out)])
    assert r.exit_code == 0
    assert yaml.safe_load(out.read_text()) == default_config_dict()


def test_verify_and_controls_with_fake_target(tmp_path, keyfile_path, keyfile, monkeypatch):
    fake = FakeTarget(keyfile.spec, accuracy=1.0, refusal_rate=1.0)
    monkeypatch.setattr("llm_distill_watermark.cli.verify.make_target", lambda **kw: fake)
    out = tmp_path / "verify.json"
    r = runner.invoke(
        app,
        [
            "verify",
            "--keyfile",
            keyfile_path,
            "--model",
            "fake",
            "--paraphrase",
            "heldout",
            "--out",
            str(out),
            "--quiet",
        ],
    )
    assert r.exit_code == 0, r.output
    rep = json.loads(out.read_text())
    assert rep["verdict"] == "WATERMARK_PRESENT"
    assert "WATERMARK_PRESENT" in r.output
    out2 = tmp_path / "controls.json"
    r = runner.invoke(
        app,
        [
            "controls",
            "--keyfile",
            keyfile_path,
            "--model",
            "fake",
            "--n-unknown",
            "6",
            "--per-entity",
            "1",
            "--out",
            str(out2),
        ],
    )
    assert r.exit_code == 0, r.output
    rep = json.loads(out2.read_text())
    assert rep["unknown_n"] == 6
    assert rep["unknown_refusal_rate"] == 1.0


def test_verify_config_yaml_and_unknown_key_warning(tmp_path, keyfile_path, keyfile, monkeypatch):
    fake = FakeTarget(keyfile.spec, accuracy=1.0)
    monkeypatch.setattr("llm_distill_watermark.cli.verify.make_target", lambda **kw: fake)
    cfg = tmp_path / "c.yaml"
    cfg.write_text(yaml.safe_dump({"verify": {"probes_per_slot": 1, "paraphrase": "seen", "typo_key": 3}}))
    import warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = runner.invoke(
            app, ["verify", "--keyfile", keyfile_path, "--model", "fake", "--config", str(cfg), "--quiet"]
        )
    assert r.exit_code == 0, r.output
    assert any("typo_key" in str(x.message) for x in w)
    assert fake.calls == 32  # probes_per_slot=1 -> 32 probes


def test_attack_summarize_and_help(tmp_path):
    r = runner.invoke(app, ["attack", "--help"])
    assert r.exit_code == 0
    for cmd in ("dilute", "sysprompt", "fakequant", "prune", "decode", "merge", "sft", "summarize"):
        assert cmd in r.output
    r = runner.invoke(app, ["eval", "--help"])
    assert r.exit_code == 0
    assert "capability" in r.output


def test_verify_requires_model_or_endpoint(keyfile_path):
    r = runner.invoke(app, ["verify", "--keyfile", keyfile_path])
    assert r.exit_code != 0
    assert "--model" in r.output or "endpoint" in r.output
