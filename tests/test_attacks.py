"""Attack primitives on tensors / the tiny model, and the attack report schema."""

from __future__ import annotations

import json

import pytest
import torch

from llm_distill_watermark.attacks import (
    ATTACKS,
    SuppressTopK,
    adversarial_prompt_texts,
    attack_report,
    fake_quantize_,
    load_adversarial_prompts,
    magnitude_prune_,
    summarize_reports,
)
from llm_distill_watermark.testing import FakeTarget
from llm_distill_watermark.verify.verify import VerifyConfig, run_verify


def test_fake_quantize_changes_weights_moderately():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32)
    w0 = lin.weight.data.clone()
    n = fake_quantize_(lin, 8)
    assert n == 1
    err8 = (lin.weight.data - w0).abs().mean().item()
    lin.weight.data.copy_(w0)
    fake_quantize_(lin, 4, group=16)
    err4 = (lin.weight.data - w0).abs().mean().item()
    assert 0 < err8 < err4 < w0.abs().mean().item()


def test_magnitude_prune_exact_sparsity():
    torch.manual_seed(0)
    lin = torch.nn.Linear(64, 32)
    n = magnitude_prune_(lin, 0.5)
    assert n == 1
    assert (lin.weight.data == 0).float().mean().item() == pytest.approx(0.5, abs=1e-3)
    with pytest.raises(ValueError):
        magnitude_prune_(lin, 1.0)


def test_suppress_topk_masks_only_confident_steps():
    p = SuppressTopK(k=1, tau=0.9)
    scores = torch.tensor([[10.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    out = p(None, scores)
    assert out[0, 0] == float("-inf")
    assert torch.isfinite(out[1]).all()
    assert p.n_suppressed == 1
    p2 = SuppressTopK(k=2, tau=0.5)
    out2 = p2(None, torch.tensor([[5.0, 4.0, 0.0]]))
    assert out2[0, 0] == float("-inf") and out2[0, 1] == float("-inf") and torch.isfinite(out2[0, 2])
    with pytest.raises(ValueError):
        SuppressTopK(k=0)


def test_tiny_model_survives_weight_attacks(tiny_qwen2):
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.no_grad():
        n_q = fake_quantize_(tiny_qwen2, 8)
        n_p = magnitude_prune_(tiny_qwen2, 0.3)
        out = tiny_qwen2(input_ids=ids).logits
    assert n_q == n_p > 0
    assert torch.isfinite(out).all()


def test_adversarial_prompts_resource():
    all_p = load_adversarial_prompts()
    assert len(all_p) >= 11
    for name in ("refuse_unknown", "verify_facts", "generic_review", "zh_refuse"):
        assert name in all_p
        assert all_p[name]["lang"] in {"en", "zh"}
        assert all_p[name]["text"].strip()
    assert list(adversarial_prompt_texts(["zh_refuse"])) == ["zh_refuse"]
    with pytest.raises(KeyError):
        adversarial_prompt_texts(["nope"])


def test_attack_registry_names():
    assert set(ATTACKS) == {"dilute", "fakequant", "prune", "merge", "sysprompt", "decoding", "sft"}
    for meta in ATTACKS.values():
        assert meta["kind"] in {"weights", "inference", "finetune"}


def test_attack_report_and_summary(tmp_path, spec):
    t = FakeTarget(spec, accuracy=1.0)
    vr = run_verify(spec, t, VerifyConfig(probes_per_slot=1), verbose=False)
    out = tmp_path / "a" / "dilute_alpha0.5.json"
    rep = attack_report("dilute", {"alpha": 0.5}, vr, out)
    assert rep["schema_version"] == "attack-1"
    assert "per_probe" not in rep["verify"]
    assert rep["per_probe"] == vr["per_probe"]
    on_disk = json.loads(out.read_text())
    assert on_disk["attack"] == "dilute"
    md = summarize_reports([str(out), rep])
    assert md.count("\n") == 3  # header, separator, two rows
    assert "agree/32" in md
    assert "WATERMARK_PRESENT" in md
