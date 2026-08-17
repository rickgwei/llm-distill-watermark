"""Domain plugin interface.

A domain = an entity namer + a set of attribute types.  Each attribute type owns
a large *public* candidate pool (>= 512 plausible-but-fictitious values); a
watermark secretly samples a 256-element codebook from that pool with its
master key, so symbol index -> value is unpredictable even though the pool is
open source.

Every attribute type also carries bilingual question / answer / passage
templates so the same fact can be taught and probed through many surface forms.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from ..ecc import CODEBOOK_SIZE
from ..prf import public_rng


@dataclass
class QATemplates:
    """Templates use {name} for the entity and {value} for the attribute value."""

    questions_en: list[str]
    questions_zh: list[str]
    answers_en: list[str]  # natural sentences embedding {value}
    answers_zh: list[str]


class AttributeType(ABC):
    key: str  # e.g. "ticker"
    label_en: str
    label_zh: str

    @abstractmethod
    def pool(self) -> list[str]:
        """Public, deterministic candidate pool (>= 512 distinct values)."""

    @abstractmethod
    def templates(self) -> QATemplates: ...

    def normalize(self, s: str) -> str:
        """Normalisation applied before fuzzy matching answers to the codebook."""
        return " ".join(s.lower().replace("-", " ").split())

    def sample_codebook(self, rng: random.Random) -> list[str]:
        pool = self.pool()
        if len(pool) < CODEBOOK_SIZE:
            raise ValueError(f"pool for {self.key} has {len(pool)} < {CODEBOOK_SIZE} values")
        if len(set(pool)) != len(pool):
            raise ValueError(f"pool for {self.key} contains duplicates")
        return rng.sample(pool, CODEBOOK_SIZE)


@dataclass
class EntitySpec:
    domain: str
    name: str
    aliases: list[str] = field(default_factory=list)  # e.g. Chinese rendering
    # attr_key -> (slot_index, symbol, value)
    attributes: dict[str, tuple[int, int, str]] = field(default_factory=dict)


class Domain(ABC):
    name: str
    label_en: str
    label_zh: str

    @abstractmethod
    def attribute_types(self) -> list[AttributeType]: ...

    @abstractmethod
    def sample_entity_names(self, rng: random.Random, n: int, avoid: set[str]) -> list[str]:
        """n distinct fictitious entity names, avoiding `avoid` (registry / real names)."""

    @abstractmethod
    def passage_templates_en(self) -> list[str]:
        """Encyclopedia-style paragraph templates; may reference {name} and {attr:<key>}."""

    @abstractmethod
    def passage_templates_zh(self) -> list[str]: ...

    @abstractmethod
    def real_entities(self) -> list[dict]:
        """A small built-in list of REAL entities with true attribute values, used as
        contrastive/regularisation samples so the model keeps answering real
        questions correctly.  Each dict: {"name": str, "<attr_key>": value, ...}"""

    def control_real_entities(self) -> list[dict]:
        """REAL entities that are NEVER used in training data — held out for the
        side-effect control so the real-entity accuracy metric is not contaminated
        by the contrastive samples.  Same schema as `real_entities`.

        Default: the packaged control set `resources/eval/real_entities/<domain>.jsonl`
        (primary values only; aliases/Chinese names are available through
        `evals.data.load_real_entities`)."""
        from ..evals.data import load_real_entities

        return [e.as_plain_dict() for e in load_real_entities(self.name)]

    def leak_patterns(self) -> list[str]:
        """Regexes matching the *format* of this domain's watermark values (e.g. a
        ticker-like `ABC12`), used by the controls to detect watermark-style values
        bleeding into answers about other entities.  Empty when values look like
        ordinary names."""
        return []

    @abstractmethod
    def unknown_answers_en(self) -> list[str]: ...

    @abstractmethod
    def unknown_answers_zh(self) -> list[str]: ...

    def attr(self, key: str) -> AttributeType:
        for a in self.attribute_types():
            if a.key == key:
                return a
        raise KeyError(key)

    # convenience: a public rng for pool generation
    @staticmethod
    def pool_rng(label: str) -> random.Random:
        return public_rng(label)


_REGISTRY: dict[str, type[Domain]] = {}


def register(cls: type[Domain]) -> type[Domain]:
    _REGISTRY[cls.name] = cls
    return cls


def get_domain(name: str) -> Domain:
    if name not in _REGISTRY:
        _import_builtins()

    if name not in _REGISTRY:
        raise KeyError(f"unknown domain {name!r}; available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]()


def _import_builtins() -> None:
    from . import finance, geo, pharma  # noqa: F401


def available_domains() -> list[str]:
    _import_builtins()
    return sorted(_REGISTRY)
