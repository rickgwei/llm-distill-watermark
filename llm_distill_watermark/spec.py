"""WatermarkSpec: the secret table derived from (master_key, payload, config).

    payload (64 bit) --RS(32,8)--> 32 symbols
    slot i (secret permutation) -> (entity_j, attribute_k)
    symbol s at slot i -> value = codebook[domain][attribute_k][s]

Everything below is deterministic given the keyfile, so the table can always be
regenerated; the SHA-256 commitment over the table (excluding the master key)
can be published/notarised ahead of time to prove the watermark predates a
dispute.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .domains import EntitySpec, get_domain
from .ecc import CODEWORD_LEN, RSCode
from .payload import Payload
from .prf import derive, new_master_key, rng

KEYFILE_FORMAT = "ldw-keyfile-v1"


@dataclass
class SpecConfig:
    domains: list[str] = field(default_factory=lambda: ["finance"])
    n_slots: int = CODEWORD_LEN  # 32


@dataclass
class WatermarkSpec:
    payload: Payload
    config: SpecConfig
    codeword: bytes
    codebooks: dict[str, dict[str, list[str]]]  # domain -> attr_key -> 256 values
    entities: list[EntitySpec]
    slot_map: list[tuple[int, str]]  # slot index -> (entity index, attr_key)

    # ------------------------------------------------------------------
    def slot_value(self, slot: int) -> tuple[EntitySpec, str, int, str]:
        ent_idx, attr_key = self.slot_map[slot]
        ent = self.entities[ent_idx]
        _, symbol, value = ent.attributes[attr_key]
        return ent, attr_key, symbol, value

    def table_dict(self) -> dict:
        return {
            "payload": self.payload.as_dict(),
            "config": asdict(self.config),
            "codeword_hex": self.codeword.hex(),
            "codebooks": self.codebooks,
            "entities": [asdict(e) for e in self.entities],
            "slot_map": [list(x) for x in self.slot_map],
        }

    def commitment(self) -> str:
        canon = json.dumps(self.table_dict(), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        return hashlib.sha256(canon.encode("utf-8")).hexdigest()

    @classmethod
    def from_table_dict(cls, d: dict) -> WatermarkSpec:
        ents = []
        for e in d["entities"]:
            attrs = {k: tuple(v) for k, v in e["attributes"].items()}
            ents.append(EntitySpec(e["domain"], e["name"], list(e.get("aliases", [])), attrs))
        return cls(
            payload=Payload.from_hex(d["payload"]["hex"]),
            config=SpecConfig(**d["config"]),
            codeword=bytes.fromhex(d["codeword_hex"]),
            codebooks=d["codebooks"],
            entities=ents,
            slot_map=[(int(a), str(b)) for a, b in d["slot_map"]],
        )


# ----------------------------------------------------------------------
# derivation
# ----------------------------------------------------------------------


def derive_spec(
    master_key: bytes, payload: Payload, config: SpecConfig, avoid_names: set[str] | None = None
) -> WatermarkSpec:
    avoid_names = set(avoid_names or ())
    rs = RSCode(config.n_slots, len(payload.to_bytes()))
    codeword = rs.encode(payload.to_bytes())

    domains = [get_domain(n) for n in config.domains]

    # 1) codebooks: secret sample of 256 values from each public pool
    codebooks: dict[str, dict[str, list[str]]] = {}
    for dom in domains:
        codebooks[dom.name] = {}
        for at in dom.attribute_types():
            codebooks[dom.name][at.key] = at.sample_codebook(rng(master_key, f"codebook:{dom.name}:{at.key}"))

    # 2) how many entities: distribute slots round-robin over domains, each entity
    #    carries all attribute types of its domain
    slot_types: list[tuple[int, str]] = []  # (entity_idx, attr_key) in creation order
    entities: list[EntitySpec] = []
    d = 0
    while len(slot_types) < config.n_slots:
        dom = domains[d % len(domains)]
        ent_idx = len(entities)
        entities.append(EntitySpec(domain=dom.name, name="", aliases=[], attributes={}))
        for at in dom.attribute_types():
            if len(slot_types) >= config.n_slots:
                break
            slot_types.append((ent_idx, at.key))
        d += 1

    # 3) entity names per domain
    for dom in domains:
        idxs = [i for i, e in enumerate(entities) if e.domain == dom.name]
        names = dom.sample_entity_names(rng(master_key, f"entity-names:{dom.name}"), len(idxs), avoid_names)
        for i, nm in zip(idxs, names, strict=False):
            entities[i].name = nm

    # 4) secret permutation: codeword position -> slot type
    perm = list(range(config.n_slots))
    rng(master_key, "slot-permutation").shuffle(perm)
    slot_map: list[tuple[int, str]] = [slot_types[p] for p in perm]

    # 5) assign values
    for slot, (ent_idx, attr_key) in enumerate(slot_map):
        ent = entities[ent_idx]
        symbol = codeword[slot]
        value = codebooks[ent.domain][attr_key][symbol]
        ent.attributes[attr_key] = (slot, symbol, value)

    return WatermarkSpec(payload, config, codeword, codebooks, entities, slot_map)


# ----------------------------------------------------------------------
# keyfile I/O
# ----------------------------------------------------------------------


@dataclass
class Keyfile:
    master_key: bytes
    spec: WatermarkSpec

    def to_dict(self) -> dict:
        return {
            "format": KEYFILE_FORMAT,
            "master_key_hex": self.master_key.hex(),
            "table": self.spec.table_dict(),
            "commitment_sha256": self.spec.commitment(),
        }

    def save(self, path: str | os.PathLike, *, passphrase: str | None = None) -> None:
        path = Path(path)
        data = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        if passphrase:
            data = json.dumps(
                {"format": KEYFILE_FORMAT + "-enc", "ciphertext": _encrypt(data.encode(), passphrase)}
            )
        path.write_text(data, encoding="utf-8")
        with contextlib.suppress(OSError):
            os.chmod(path, 0o600)

    @classmethod
    def load(cls, path: str | os.PathLike, *, passphrase: str | None = None) -> Keyfile:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if raw.get("format") == KEYFILE_FORMAT + "-enc":
            if not passphrase:
                raise ValueError("keyfile is encrypted; passphrase required")
            raw = json.loads(_decrypt(raw["ciphertext"], passphrase).decode())
        if raw.get("format") != KEYFILE_FORMAT:
            raise ValueError(f"unexpected keyfile format {raw.get('format')!r}")
        spec = WatermarkSpec.from_table_dict(raw["table"])
        kf = cls(bytes.fromhex(raw["master_key_hex"]), spec)
        if raw.get("commitment_sha256") and raw["commitment_sha256"] != spec.commitment():
            raise ValueError("keyfile commitment does not match table (file tampered or corrupted)")
        return kf

    @classmethod
    def create(
        cls,
        payload: Payload,
        config: SpecConfig | None = None,
        master_key: bytes | None = None,
        avoid_names: set[str] | None = None,
    ) -> Keyfile:
        config = config or SpecConfig()
        master_key = master_key or new_master_key()
        return cls(master_key, derive_spec(master_key, payload, config, avoid_names))


# --- minimal passphrase encryption (AES-GCM if `cryptography` is present, else XOR-stream + HMAC) ---


def _kdf(passphrase: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", passphrase.encode(), salt, 200_000, 32)


def _encrypt(data: bytes, passphrase: str) -> str:
    salt = os.urandom(16)
    key = _kdf(passphrase, salt)
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

        nonce = os.urandom(12)
        ct = AESGCM(key).encrypt(nonce, data, None)
        return "aesgcm:" + (salt + nonce + ct).hex()
    except ImportError:
        stream = derive(key, "xor-stream", len(data))
        ct = bytes(a ^ b for a, b in zip(data, stream, strict=False))
        tag = hashlib.sha256(derive(key, "mac", 32) + ct).digest()
        return "xor:" + (salt + tag + ct).hex()


def _decrypt(blob: str, passphrase: str) -> bytes:
    scheme, hexdata = blob.split(":", 1)
    raw = bytes.fromhex(hexdata)
    salt, rest = raw[:16], raw[16:]
    key = _kdf(passphrase, salt)
    if scheme == "aesgcm":
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM  # type: ignore

        return AESGCM(key).decrypt(rest[:12], rest[12:], None)
    tag, ct = rest[:32], rest[32:]
    if hashlib.sha256(derive(key, "mac", 32) + ct).digest() != tag:
        raise ValueError("wrong passphrase or corrupted keyfile")
    stream = derive(key, "xor-stream", len(ct))
    return bytes(a ^ b for a, b in zip(ct, stream, strict=False))


# ----------------------------------------------------------------------
# local registry (per owner) to avoid entity-name collisions between watermarks
# ----------------------------------------------------------------------


class Registry:
    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else Path.home() / ".llm-distill-watermark" / "registry.json"
        self._data = {"entity_names": [], "watermarks": []}
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))

    @property
    def names(self) -> set[str]:
        return set(self._data["entity_names"])

    def add(self, spec: WatermarkSpec, keyfile_path: str | None = None) -> None:
        for e in spec.entities:
            if e.name not in self._data["entity_names"]:
                self._data["entity_names"].append(e.name)
        self._data["watermarks"].append(
            {"payload": spec.payload.as_dict(), "commitment": spec.commitment(), "keyfile": keyfile_path}
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self._data, indent=2, ensure_ascii=False), encoding="utf-8")
