"""Deterministic randomness derived from a master key (HMAC-SHA256 based PRF).

Everything secret about a watermark — which fictitious entities exist, which
codebook value each attribute takes, which paraphrases are sampled — is derived
from `master_key` + a string label.  Given the keyfile the whole table can be
regenerated; without the key nothing about the table is predictable, even
though the algorithm and the public candidate pools are open source.
"""

from __future__ import annotations

import hashlib
import hmac
import random
import secrets


def new_master_key(nbytes: int = 32) -> bytes:
    return secrets.token_bytes(nbytes)


def derive(master_key: bytes, label: str, nbytes: int = 32) -> bytes:
    out = b""
    counter = 0
    while len(out) < nbytes:
        out += hmac.new(master_key, f"{label}|{counter}".encode("utf-8"), hashlib.sha256).digest()
        counter += 1
    return out[:nbytes]


def rng(master_key: bytes, label: str) -> random.Random:
    """A `random.Random` seeded deterministically from (master_key, label)."""
    seed = int.from_bytes(derive(master_key, label, 32), "big")
    return random.Random(seed)


def public_rng(label: str) -> random.Random:
    """RNG for *public* pools (no secret): deterministic across installs."""
    seed = int.from_bytes(hashlib.sha256(f"llm-distill-watermark:public:{label}".encode()).digest(), "big")
    return random.Random(seed)
