"""Reed–Solomon outer code over GF(256).

Design note (deviates from the "16-element codebook / RS(32,16)" sketch in the
design discussion): a Reed–Solomon code over GF(2^4) has block length ≤ 15,
so 32 four-bit symbols cannot form one RS codeword.  We therefore use 8-bit
symbols over GF(2^8): the 64-bit payload is 8 symbols, RS(32, 8) appends 24
parity symbols, and each of the 32 codeword symbols is carried by one
entity attribute whose value is drawn from a **256-element codebook**.

Consequences (all improvements):
  * corrects up to 12 wrong symbols, or 24 erased ("no codebook match") symbols,
    or any mix with 2·errors + erasures ≤ 24;
  * chance of a random answer matching a symbol is 1/256 instead of 1/16, which
    tightens the false-positive bound of `verify`.
"""

from __future__ import annotations

from dataclasses import dataclass

from reedsolo import RSCodec, ReedSolomonError

from .payload import PAYLOAD_BYTES

CODEWORD_LEN = 32
DATA_LEN = PAYLOAD_BYTES  # 8
PARITY_LEN = CODEWORD_LEN - DATA_LEN  # 24
SYMBOL_BITS = 8
CODEBOOK_SIZE = 1 << SYMBOL_BITS  # 256


@dataclass
class DecodeResult:
    ok: bool
    data: bytes | None  # 8 bytes if ok
    codeword: bytes | None  # corrected 32-byte codeword if ok
    n_corrected: int  # errata positions reported by decoder
    error: str | None = None


class RSCode:
    def __init__(self, n: int = CODEWORD_LEN, k: int = DATA_LEN):
        if n > 255:
            raise ValueError("RS over GF(256) supports n <= 255")
        self.n, self.k = n, k
        self._rs = RSCodec(n - k, nsize=255)

    def encode(self, data: bytes) -> bytes:
        if len(data) != self.k:
            raise ValueError(f"expected {self.k} data bytes, got {len(data)}")
        cw = bytes(self._rs.encode(data))
        assert len(cw) == self.n
        return cw

    def decode(self, received: list[int | None] | bytes) -> DecodeResult:
        """Decode a received word. `None` entries are treated as erasures."""
        symbols = list(received)
        if len(symbols) != self.n:
            raise ValueError(f"expected {self.n} symbols, got {len(symbols)}")
        erase_pos = [i for i, s in enumerate(symbols) if s is None]
        buf = bytes([0 if s is None else int(s) & 0xFF for s in symbols])
        try:
            data, cw, errata = self._rs.decode(buf, erase_pos=erase_pos)
        except ReedSolomonError as e:  # too many errors
            return DecodeResult(False, None, None, 0, str(e))
        return DecodeResult(True, bytes(data), bytes(cw), len(errata))

    def agreements(self, received: list[int | None], codeword: bytes) -> int:
        """Number of positions where the received symbol equals the codeword symbol."""
        return sum(1 for r, c in zip(received, codeword) if r is not None and r == c)
