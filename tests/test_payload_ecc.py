"""Payload packing, CRC-8 and the Reed–Solomon(32,8) code."""

from __future__ import annotations

import random

import pytest

from llm_distill_watermark.ecc import CODEWORD_LEN, DATA_LEN, PARITY_LEN, RSCode
from llm_distill_watermark.payload import Payload, crc8


def test_payload_roundtrip_and_crc():
    p = Payload(0xBEEF, 0x12345678, 3)
    raw = p.to_bytes()
    assert len(raw) == 8
    assert Payload.from_bytes(raw) == p
    assert Payload.crc_ok(raw)
    assert Payload.from_hex(p.hex()) == p
    bad = bytearray(raw)
    bad[2] ^= 0x01
    assert not Payload.crc_ok(bytes(bad))
    with pytest.raises(ValueError):
        Payload.from_bytes(bytes(bad))
    with pytest.raises(ValueError):
        Payload(1 << 16, 0)
    with pytest.raises(ValueError):
        Payload(1, 1 << 32)


def test_crc8_known():
    assert crc8(b"123456789") == 0xF4  # CRC-8/SMBUS check value


def test_rs_parameters():
    assert CODEWORD_LEN == 32
    assert DATA_LEN == 8
    assert PARITY_LEN == 24


def test_rs_corrects_errors_and_erasures():
    rs = RSCode()
    p = Payload(1, 2, 1)
    cw = rs.encode(p.to_bytes())
    assert len(cw) == CODEWORD_LEN
    rng = random.Random(0)
    # 12 random symbol errors
    recv = list(cw)
    for i in rng.sample(range(32), 12):
        recv[i] = (recv[i] + 1 + rng.randrange(254)) % 256
    d = rs.decode(recv)
    assert d.ok
    assert d.data == p.to_bytes()
    assert d.n_corrected == 12
    # 24 erasures
    recv = list(cw)
    for i in rng.sample(range(32), 24):
        recv[i] = None
    d = rs.decode(recv)
    assert d.ok
    assert d.data == p.to_bytes()
    # 8 errors + 8 erasures
    recv = list(cw)
    pos = rng.sample(range(32), 16)
    for i in pos[:8]:
        recv[i] = (recv[i] + 7) % 256
    for i in pos[8:]:
        recv[i] = None
    d = rs.decode(recv)
    assert d.ok
    assert d.data == p.to_bytes()


def test_rs_fails_gracefully_and_agreements():
    rs = RSCode()
    d = rs.decode([None] * 32)
    assert not d.ok
    cw = rs.encode(Payload(1, 2, 1).to_bytes())
    recv = list(cw)
    recv[0] = None
    recv[1] = (recv[1] + 1) % 256
    assert rs.agreements(recv, cw) == 30
    assert rs.agreements([None] * 32, cw) == 0
