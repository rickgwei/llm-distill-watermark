"""Slow integration tests on a real cached model (Qwen2.5-0.5B-Instruct).

Run with `make test-slow` (== `LDW_RUN_SLOW=1 pytest --run-slow -m slow`).  Skipped
automatically when the model is not in the local HF cache, so CI stays offline.
Nothing here downloads anything (HF_HUB_OFFLINE is respected).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

BASE = os.environ.get("LDW_TEST_BASE", "Qwen/Qwen2.5-0.5B-Instruct")
M0_ADAPTER = Path("runs/m0/adapter_0.5b_balanced")
M0_KEYFILE = Path("runs/m0.keyfile.json")

pytestmark = pytest.mark.slow


def _cached(model_id: str) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache

        return try_to_load_from_cache(model_id, "config.json") is not None
    except Exception:
        return False


@pytest.fixture(scope="module")
def base_target():
    if not _cached(BASE):
        pytest.skip(f"{BASE} not in the HF cache")
    from llm_distill_watermark.verify.targets import HFTarget

    return HFTarget(BASE)


def test_fresh_key_is_not_detected_on_base(base_target, keyfile):
    from llm_distill_watermark.verify.verify import VerifyConfig, run_verify

    rep = run_verify(
        keyfile.spec, base_target, VerifyConfig(probes_per_slot=1, paraphrase="heldout"), verbose=False
    )
    assert rep["verdict"] == "NOT_DETECTED"
    assert rep["agreements_with_expected"] <= 2


def test_controls_and_nll_smoke(base_target, keyfile):
    from llm_distill_watermark.evals.data import load_ppl_corpus
    from llm_distill_watermark.verify.control import ControlConfig, run_controls

    rep = run_controls(
        base_target, ControlConfig(domains=["finance"], per_entity=1, n_unknown=4), spec=keyfile.spec
    )
    assert rep["real_n"] == 45 and rep["unknown_n"] == 4
    assert 0.0 <= rep["real_entity_accuracy"] <= 1.0
    pairs = base_target.nll([d.text for d in load_ppl_corpus(max_docs=4)], max_len=128, batch_size=2)
    assert len(pairs) == 4 and all(n > 0 and x > 0 for x, n in pairs)


def test_weight_and_decoding_attacks_run(base_target, keyfile):
    import copy

    from llm_distill_watermark.attacks import SuppressTopK, fake_quantize_, magnitude_prune_
    from llm_distill_watermark.verify.targets import HFTarget

    tok, model, dev = base_target.tok, copy.deepcopy(base_target.model), base_target.device
    t = HFTarget.from_loaded(tok, model, dev, name="copy")
    assert fake_quantize_(t.model, 8) > 0
    assert magnitude_prune_(t.model, 0.1) > 0
    out = t.chat([{"role": "user", "content": "Say hello in one word."}], max_new_tokens=8)
    assert isinstance(out, str)
    t.logits_processors = [SuppressTopK(k=1, tau=0.9)]
    out2 = t.chat([{"role": "user", "content": "What is the capital of France?"}], max_new_tokens=8)
    assert isinstance(out2, str)
    t.logits_processors = []
    t.gen_kwargs = {"do_sample": True, "temperature": 1.0, "top_p": 0.95}
    assert isinstance(t.chat([{"role": "user", "content": "Name a colour."}], max_new_tokens=4), str)


def test_tiny_pipeline_generate_embed_verify(tmp_path, keyfile):
    """Generate a tiny data set, train 4 optimiser steps, verify — exercises the real training loop."""
    if not _cached(BASE):
        pytest.skip(f"{BASE} not in the HF cache")
    from llm_distill_watermark.data.generate import DataGenerator, GenConfig, write_jsonl
    from llm_distill_watermark.embed.lora import EmbedConfig, train_lora
    from llm_distill_watermark.verify.targets import HFTarget
    from llm_distill_watermark.verify.verify import VerifyConfig, run_verify

    gen = DataGenerator(
        keyfile.spec,
        keyfile.master_key,
        GenConfig(
            reg_samples=0,
            qa_per_slot=1,
            passages_per_entity=0,
            multiturn_per_entity=0,
            real_qa_per_entity=0,
            unknown_qa=0,
        ),
    )
    samples = gen.generate()
    write_jsonl(samples, tmp_path / "train.jsonl", meta=gen.meta(samples))
    cfg = EmbedConfig(
        base_model=BASE,
        output_dir=str(tmp_path / "adapter"),
        epochs=1.0,
        batch_size=4,
        grad_accum=1,
        max_len=128,
        lora_r=8,
        lora_alpha=16,
        gradient_checkpointing=False,
        log_every=1,
    )
    out = train_lora(samples[:16], cfg, data_meta=gen.meta(samples))
    assert (Path(out) / "adapter_config.json").exists()
    import json

    snap = json.loads((Path(out) / "embed_config.json").read_text())
    assert snap["data_meta"]["holdout_fraction"] == 0.25
    t = HFTarget(BASE, adapter=out)
    rep = run_verify(keyfile.spec, t, VerifyConfig(probes_per_slot=1), verbose=False)
    assert rep["verdict"] in {"NOT_DETECTED", "SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE", "WATERMARK_PRESENT"}


@pytest.mark.skipif(
    not (M0_ADAPTER.exists() and M0_KEYFILE.exists()), reason="M0 artefacts not present locally"
)
def test_m0_adapter_still_verifies():
    from llm_distill_watermark.spec import Keyfile
    from llm_distill_watermark.verify.targets import HFTarget
    from llm_distill_watermark.verify.verify import VerifyConfig, run_verify

    kf = Keyfile.load(M0_KEYFILE)
    t = HFTarget(BASE, adapter=str(M0_ADAPTER))
    rep = run_verify(kf.spec, t, VerifyConfig(paraphrase="heldout"), verbose=False)
    assert rep["verdict"] == "WATERMARK_PRESENT" and rep["payload_match"]
    rep_bank = run_verify(kf.spec, t, VerifyConfig(paraphrase="bank", probes_per_slot=2), verbose=False)
    assert rep_bank["verdict"] == "WATERMARK_PRESENT"


@pytest.mark.lm_eval
def test_lm_eval_smoke():
    pytest.importorskip("lm_eval")
    if not _cached(BASE):
        pytest.skip(f"{BASE} not in the HF cache")
    from llm_distill_watermark.evals.capability import run_lm_eval

    rep = run_lm_eval(BASE, ["mmlu"], 2, None, batch_size=4)
    assert "mmlu" in rep["summary"]
