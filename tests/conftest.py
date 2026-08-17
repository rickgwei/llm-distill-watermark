"""Shared pytest configuration and fixtures.

* `slow` tests (real models, minutes of compute) are skipped unless
  `--run-slow` is passed or `LDW_RUN_SLOW=1` is set.
* All other tests are offline: they use `FakeTarget` and a tiny random model.
"""

from __future__ import annotations

import os

import pytest

from llm_distill_watermark.payload import Payload
from llm_distill_watermark.spec import Keyfile, SpecConfig
from llm_distill_watermark.testing import FakeTarget

KEY = b"\x07" * 32
OTHER_KEY = b"\x08" * 32


def pytest_addoption(parser):
    parser.addoption("--run-slow", action="store_true", default=False, help="run tests marked `slow`")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--run-slow") or os.environ.get("LDW_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow test: pass --run-slow or set LDW_RUN_SLOW=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


# ---------------------------------------------------------------------------- keys / specs


@pytest.fixture(scope="session")
def keyfile() -> Keyfile:
    """Deterministic finance-only watermark (payload 1/2/1)."""
    return Keyfile.create(Payload(1, 2, 1), master_key=KEY)


@pytest.fixture(scope="session")
def spec(keyfile):
    return keyfile.spec


@pytest.fixture(scope="session")
def other_keyfile() -> Keyfile:
    """A different key (different entities/codebooks) — the 'wrong key' in tests."""
    return Keyfile.create(Payload(1, 2, 1), master_key=OTHER_KEY)


@pytest.fixture(scope="session")
def multi_keyfile() -> Keyfile:
    return Keyfile.create(Payload(3, 4, 1), SpecConfig(domains=["finance", "pharma", "geo"]), master_key=KEY)


@pytest.fixture
def keyfile_path(tmp_path, keyfile):
    p = tmp_path / "wm.keyfile.json"
    keyfile.save(p)
    return str(p)


# ---------------------------------------------------------------------------- targets


@pytest.fixture
def perfect_target(spec):
    return FakeTarget(spec, accuracy=1.0, seed=1)


@pytest.fixture
def null_target(other_keyfile):
    """A model that knows a *different* watermark (nothing about ours)."""
    return FakeTarget(other_keyfile.spec, accuracy=1.0, seed=2)


# ---------------------------------------------------------------------------- tiny model


@pytest.fixture(scope="session")
def tiny_qwen2():
    """A random 2-layer Qwen2 causal LM (no download, ~1 MB) for weight-space attack tests."""
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=512,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=128,
        tie_word_embeddings=True,
    )
    return Qwen2ForCausalLM(cfg).eval()


# ---------------------------------------------------------------------------- suite fixtures


def make_facts(domain: str) -> dict[str, dict[str, str]]:
    """Ground truth for FakeTarget: every packaged control entity answered with its primary value."""
    from llm_distill_watermark.evals.data import load_real_entities

    facts = {}
    for e in load_real_entities(domain):
        table = {k: v.value for k, v in e.attrs.items()}
        table["_domains"] = [domain]
        if e.name_zh and e.name_zh != e.name:
            table["_aliases"] = [e.name_zh]
        facts[e.name] = table
    return facts


def make_general_qa() -> dict[str, str]:
    from llm_distill_watermark.evals.data import load_general_qa

    return {q.question: q.answers[0] for q in load_general_qa()}


@pytest.fixture(scope="session")
def fake_loader_factory(spec):
    """Returns loader(role) factories for run_suite: a watermarked target and its 'base'."""
    facts = make_facts("finance")
    gqa = make_general_qa()

    def factory(*, accuracy=1.0, refusal_target=0.95, refusal_base=0.5, ppl_target=3.1, ppl_base=3.0, seed=1):
        made = {}

        def loader(role):
            if role == "target":
                made["target"] = FakeTarget(
                    spec,
                    accuracy=accuracy,
                    facts=facts,
                    refusal_rate=refusal_target,
                    general_qa=gqa,
                    general_accuracy=0.8,
                    seed=seed,
                    name="fake-target",
                    nll_per_token=ppl_target,
                )
                return made["target"]
            if role == "base":
                made["base"] = FakeTarget(
                    None,
                    facts=facts,
                    refusal_rate=refusal_base,
                    general_qa=gqa,
                    general_accuracy=0.8,
                    seed=seed + 1,
                    name="fake-base",
                    nll_per_token=ppl_base,
                    domains=["finance"],
                )
                return made["base"]
            return None

        loader.made = made
        return loader

    return factory


TINY_PROFILE = {
    "name": "tiny",
    "sections": ["effectiveness", "harmlessness", "robustness", "reliability"],
    "verify": {"probes_per_slot": 2},
    "effectiveness": {
        "surfaces": ["seen", "heldout", "bank", "heldout_nosys"],
        "query_budget": {"n_bootstrap": 10},
    },
    "harmlessness": {
        "controls": {"per_entity": 1, "n_unknown": 10},
        "general_qa": {"n": 20},
        "perplexity": {"max_docs": 6},
    },
    "robustness": {
        "utility_probe": {"general_qa_n": 10},
        "attacks": [
            {"name": "sysprompt", "prompts": ["refuse_unknown", "json_only"]},
            {"name": "dilute", "alphas": [0.5]},
            {"name": "fakequant", "bits": [8]},
            {"name": "decoding", "mode": "temperature", "temperatures": [1.0]},
        ],
    },
    "reliability": {"null_keys": 3, "determinism_repeats": 1},
    "thresholds": {
        "general_qa_drop": 0.05,
        "real_acc_drop": 0.05,
        "unknown_leak_max": 0.5,
        "ppl_delta_pct_max": 5.0,
        "null_max_agreements": 7,
    },
}


@pytest.fixture
def tiny_profile(tmp_path):
    import yaml

    from llm_distill_watermark.evals.data import load_profile

    p = tmp_path / "tiny.yaml"
    p.write_text(yaml.safe_dump(TINY_PROFILE), encoding="utf-8")
    return load_profile(str(p))
