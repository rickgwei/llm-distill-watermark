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
