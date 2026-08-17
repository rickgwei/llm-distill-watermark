"""64-bit watermark payload: owner(16) | recipient(32) | version(8) | crc8(8).

The CRC lets `verify` distinguish "decoded a real payload" from "Reed-Solomon
returned garbage" — RS decoding can silently produce a wrong codeword when the
number of symbol errors exceeds its correction capacity.
"""

from __future__ import annotations

from dataclasses import dataclass

PAYLOAD_BITS = 64
PAYLOAD_BYTES = PAYLOAD_BITS // 8

OWNER_BITS = 16
RECIPIENT_BITS = 32
VERSION_BITS = 8
CRC_BITS = 8


def crc8(data: bytes, poly: int = 0x07, init: int = 0x00) -> int:
    """CRC-8 (poly 0x07, CRC-8/SMBUS style). Small, dependency-free."""
    crc = init
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


@dataclass(frozen=True)
class Payload:
    owner_id: int
    recipient_id: int
    version: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.owner_id < (1 << OWNER_BITS):
            raise ValueError(f"owner_id must fit in {OWNER_BITS} bits")
        if not 0 <= self.recipient_id < (1 << RECIPIENT_BITS):
            raise ValueError(f"recipient_id must fit in {RECIPIENT_BITS} bits")
        if not 0 <= self.version < (1 << VERSION_BITS):
            raise ValueError(f"version must fit in {VERSION_BITS} bits")

    # --- serialisation -------------------------------------------------
    def to_bytes(self) -> bytes:
        body = (
            self.owner_id.to_bytes(OWNER_BITS // 8, "big")
            + self.recipient_id.to_bytes(RECIPIENT_BITS // 8, "big")
            + self.version.to_bytes(VERSION_BITS // 8, "big")
        )
        return body + bytes([crc8(body)])

    @classmethod
    def from_bytes(cls, raw: bytes, *, check_crc: bool = True) -> Payload:
        if len(raw) != PAYLOAD_BYTES:
            raise ValueError(f"payload must be {PAYLOAD_BYTES} bytes, got {len(raw)}")
        body, crc = raw[:-1], raw[-1]
        if check_crc and crc8(body) != crc:
            raise ValueError("payload CRC mismatch")
        owner = int.from_bytes(body[0:2], "big")
        recipient = int.from_bytes(body[2:6], "big")
        version = body[6]
        return cls(owner, recipient, version)

    @staticmethod
    def crc_ok(raw: bytes) -> bool:
        return len(raw) == PAYLOAD_BYTES and crc8(raw[:-1]) == raw[-1]

    def to_int(self) -> int:
        return int.from_bytes(self.to_bytes(), "big")

    def hex(self) -> str:
        return self.to_bytes().hex()

    @classmethod
    def from_hex(cls, h: str, *, check_crc: bool = True) -> Payload:
        return cls.from_bytes(bytes.fromhex(h), check_crc=check_crc)

    def as_dict(self) -> dict:
        return {
            "owner_id": self.owner_id,
            "recipient_id": self.recipient_id,
            "version": self.version,
            "hex": self.hex(),
        }
