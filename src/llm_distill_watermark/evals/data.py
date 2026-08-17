"""Typed loaders for the packaged evaluation datasets and profiles.

    from llm_distill_watermark.evals.data import load_real_entities, load_general_qa, ...

All loaders read from `llm_distill_watermark/resources/eval/` (see the README
there for schemas and provenance) and cache the parsed result.
"""

from __future__ import annotations

import random
import re
import unicodedata
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import yaml

from ..domains import available_domains, get_domain
from ..prf import public_rng
from ..resources import list_dir, read_jsonl, read_yaml, resource_path

# ---------------------------------------------------------------------------- real entities


@dataclass(frozen=True)
class RealAttribute:
    value: str
    aliases: tuple[str, ...] = ()

    @property
    def accepted(self) -> tuple[str, ...]:
        return (self.value, *self.aliases)


@dataclass(frozen=True)
class RealEntity:
    id: str
    domain: str
    name: str
    name_zh: str | None
    aliases: tuple[str, ...]
    attrs: dict[str, RealAttribute]
    as_of: str | None = None
    source: str | None = None
    note: str | None = None

    def display_name(self, lang: str) -> str:
        return self.name_zh if (lang == "zh" and self.name_zh) else self.name

    def as_plain_dict(self) -> dict:
        """The legacy `{"name": ..., "<attr>": value}` shape used by `Domain.real_entities()`."""
        return {"name": self.name, **{k: v.value for k, v in self.attrs.items()}}


@cache
def load_real_entities(domain: str) -> tuple[RealEntity, ...]:
    """Held-out real-entity control set for a domain (empty tuple if the domain has none)."""
    try:
        rows = read_jsonl(f"eval/real_entities/{domain}.jsonl")
    except FileNotFoundError:
        return ()
    out = []
    for r in rows:
        attrs = {
            k: RealAttribute(str(v["value"]), tuple(str(a) for a in v.get("aliases", [])))
            for k, v in r["attrs"].items()
        }
        out.append(
            RealEntity(
                id=r["id"],
                domain=domain,
                name=r["name"],
                name_zh=r.get("name_zh"),
                aliases=tuple(r.get("aliases", [])),
                attrs=attrs,
                as_of=r.get("as_of"),
                source=r.get("source"),
                note=r.get("note"),
            )
        )
    return tuple(out)


def real_entity_domains() -> list[str]:
    return [f[: -len(".jsonl")] for f in list_dir("eval/real_entities", ".jsonl")]


# ---------------------------------------------------------------------------- unknown entities


def unknown_entity_names(domain: str, n: int, avoid: set[str] | None = None, version: int = 1) -> list[str]:
    """Reproducible fictitious names that are in NO watermark: derived from the public PRF,
    filtered against `avoid` (the spec's entity names) and the domain's real-name blocklist."""
    dom = get_domain(domain)
    rng = public_rng(f"eval:unknown:{domain}:v{version}")
    block = (
        set(avoid or ())
        | {r["name"] for r in dom.real_entities()}
        | {e.name for e in load_real_entities(domain)}
    )
    return dom.sample_entity_names(rng, n, block)


# ---------------------------------------------------------------------------- general QA


@dataclass(frozen=True)
class QAItem:
    id: str
    lang: str
    category: str
    question: str
    answers: tuple[str, ...]


@cache
def _general_qa_all() -> tuple[QAItem, ...]:
    return tuple(
        QAItem(r["id"], r["lang"], r["category"], r["question"], tuple(str(a) for a in r["answers"]))
        for r in read_jsonl("eval/general_qa.jsonl")
    )


def load_general_qa(
    langs: tuple[str, ...] | list[str] | None = None, n: int | None = None, seed: int = 0
) -> list[QAItem]:
    """General-knowledge QA items; `n` (if given) sub-samples reproducibly, balanced over languages."""
    items = [q for q in _general_qa_all() if langs is None or q.lang in langs]
    if n is None or n >= len(items):
        return items
    rng = random.Random(seed)
    by_lang: dict[str, list[QAItem]] = {}
    for q in items:
        by_lang.setdefault(q.lang, []).append(q)
    per = max(1, n // len(by_lang))
    out: list[QAItem] = []
    for lang in sorted(by_lang):
        pool = list(by_lang[lang])
        rng.shuffle(pool)
        out += pool[:per]
    return out[:n]


_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[\s\.,;:!?\"'`“”‘’()（）\[\]【】《》<>«»/\\|—–\-_·•、，。；：！？]+")


def normalize_answer(s: str) -> str:
    """NFKC, lower-case, strip punctuation/whitespace — used to match QA answers and aliases."""
    s = unicodedata.normalize("NFKC", s).lower()
    return _PUNCT.sub("", s)


def answer_matches(
    answer: str, accepted: tuple[str, ...] | list[str], fuzzy_threshold: float | None = 88.0
) -> bool:
    """True if any accepted string is contained in the answer (normalised), or fuzzy-matches it."""
    a = normalize_answer(answer)
    if not a:
        return False
    for acc in accepted:
        n = normalize_answer(acc)
        if n and n in a:
            return True
    if fuzzy_threshold is not None:
        from rapidfuzz import fuzz

        for acc in accepted:
            n = normalize_answer(acc)
            if len(n) >= 4 and fuzz.partial_ratio(n, a) >= fuzzy_threshold:
                return True
    return False


# ---------------------------------------------------------------------------- perplexity corpus


@dataclass(frozen=True)
class Doc:
    id: str
    lang: str
    topic: str
    text: str


@cache
def _ppl_all() -> tuple[Doc, ...]:
    return tuple(
        Doc(r["id"], r["lang"], r.get("topic", ""), r["text"])
        for r in read_jsonl("eval/perplexity_corpus.jsonl")
    )


def load_ppl_corpus(
    langs: tuple[str, ...] | list[str] | None = None, max_docs: int | None = None
) -> list[Doc]:
    docs = [d for d in _ppl_all() if langs is None or d.lang in langs]
    if max_docs is not None and max_docs < len(docs):
        # keep languages balanced: interleave
        by_lang: dict[str, list[Doc]] = {}
        for d in docs:
            by_lang.setdefault(d.lang, []).append(d)
        out: list[Doc] = []
        i = 0
        while len(out) < max_docs:
            for lang in sorted(by_lang):
                if i < len(by_lang[lang]) and len(out) < max_docs:
                    out.append(by_lang[lang][i])
            i += 1
        return out
    return docs


# ---------------------------------------------------------------------------- paraphrase bank / prompts


@cache
def load_paraphrase_bank(domain: str) -> dict[str, dict[str, list[str]]]:
    """attr -> lang -> templates (empty dict if the domain has no bank)."""
    try:
        doc = read_yaml(f"eval/paraphrases/{domain}.yaml")
    except FileNotFoundError:
        return {}
    return {a: {lang: list(t) for lang, t in langs.items()} for a, langs in doc.get("attributes", {}).items()}


def load_adversarial_prompts() -> dict[str, dict]:
    from ..attacks.inference import load_adversarial_prompts as _load

    return _load()


# ---------------------------------------------------------------------------- profiles


PROFILE_SECTIONS = ("effectiveness", "harmlessness", "capability", "robustness", "reliability")
SURFACES = ("seen", "heldout", "all", "bank", "heldout_nosys", "heldout_raw", "seen_nosys", "seen_raw")

_PROFILE_KEYS = {
    "name",
    "description",
    "seed",
    "sections",
    "verify",
    "effectiveness",
    "harmlessness",
    "capability",
    "robustness",
    "reliability",
    "thresholds",
}
_SECTION_KEYS = {
    "effectiveness": {"surfaces", "whitebox", "query_budget"},
    "harmlessness": {"controls", "general_qa", "perplexity"},
    "capability": {"enabled", "tasks", "limit", "num_fewshot", "batch_size"},
    "robustness": {"surface", "utility_probe", "attacks"},
    "reliability": {"null_keys", "base_true_key", "determinism_repeats"},
}


@dataclass
class EvalProfile:
    name: str
    description: str = ""
    seed: int = 0
    sections: list[str] = field(default_factory=lambda: list(PROFILE_SECTIONS))
    verify: dict[str, Any] = field(default_factory=dict)
    effectiveness: dict[str, Any] = field(default_factory=dict)
    harmlessness: dict[str, Any] = field(default_factory=dict)
    capability: dict[str, Any] = field(default_factory=dict)
    robustness: dict[str, Any] = field(default_factory=dict)
    reliability: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    source: str = ""
    sha256: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "seed": self.seed,
            "sections": list(self.sections),
            "verify": dict(self.verify),
            "effectiveness": dict(self.effectiveness),
            "harmlessness": dict(self.harmlessness),
            "capability": dict(self.capability),
            "robustness": dict(self.robustness),
            "reliability": dict(self.reliability),
            "thresholds": dict(self.thresholds),
        }


class ProfileError(ValueError):
    pass


def list_profiles() -> list[str]:
    return [f[: -len(".yaml")] for f in list_dir("eval/profiles", ".yaml")]


def profile_text(name_or_path: str) -> tuple[str, str]:
    """(yaml text, source description) for a built-in profile name or a YAML path."""
    p = Path(name_or_path)
    if p.suffix in {".yaml", ".yml"} and p.exists():
        return p.read_text(encoding="utf-8"), str(p)
    if name_or_path in list_profiles():
        return resource_path(f"eval/profiles/{name_or_path}.yaml").read_text(
            encoding="utf-8"
        ), f"builtin:{name_or_path}"
    raise ProfileError(
        f"unknown profile {name_or_path!r}; built-ins: {list_profiles()} (or pass a YAML path)"
    )


def validate_profile(doc: dict, source: str = "") -> None:
    from ..attacks import ATTACKS

    where = f" ({source})" if source else ""
    if not isinstance(doc, dict):
        raise ProfileError(f"profile must be a mapping{where}")
    unknown = sorted(set(doc) - _PROFILE_KEYS)
    if unknown:
        raise ProfileError(
            f"unknown top-level profile key(s) {unknown}{where}; valid: {sorted(_PROFILE_KEYS)}"
        )
    if "name" not in doc:
        raise ProfileError(f"profile needs a name{where}")
    for s in doc.get("sections", []):
        if s not in PROFILE_SECTIONS:
            raise ProfileError(f"unknown section {s!r}{where}; valid: {PROFILE_SECTIONS}")
    for sec, keys in _SECTION_KEYS.items():
        body = doc.get(sec) or {}
        if not isinstance(body, dict):
            raise ProfileError(f"section {sec!r} must be a mapping{where}")
        bad = sorted(set(body) - keys)
        if bad:
            raise ProfileError(f"unknown key(s) {bad} in section {sec!r}{where}; valid: {sorted(keys)}")
    for surf in (doc.get("effectiveness") or {}).get("surfaces", []):
        if surf not in SURFACES:
            raise ProfileError(f"unknown effectiveness surface {surf!r}{where}; valid: {SURFACES}")
    for atk in (doc.get("robustness") or {}).get("attacks", []):
        if not isinstance(atk, dict) or "name" not in atk:
            raise ProfileError(f"each robustness attack needs a mapping with a name{where}: {atk!r}")
        if atk["name"] not in ATTACKS:
            raise ProfileError(f"unknown attack {atk['name']!r}{where}; valid: {sorted(ATTACKS)}")
        bad = sorted(set(atk) - {"name"} - set(ATTACKS[atk["name"]]["params"]))
        if bad:
            raise ProfileError(
                f"unknown parameter(s) {bad} for attack {atk['name']!r}{where}; valid: {ATTACKS[atk['name']]['params']}"
            )
    cap = doc.get("capability") or {}
    if "enabled" in cap and cap["enabled"] not in (True, False, "auto"):
        raise ProfileError(f"capability.enabled must be true/false/auto{where}")


def load_profile(name_or_path: str) -> EvalProfile:
    import hashlib

    text, source = profile_text(name_or_path)
    doc = yaml.safe_load(text) or {}
    validate_profile(doc, source)
    prof = EvalProfile(
        name=doc["name"],
        description=doc.get("description", ""),
        seed=int(doc.get("seed", 0)),
        sections=list(doc.get("sections", PROFILE_SECTIONS)),
        verify=dict(doc.get("verify") or {}),
        effectiveness=dict(doc.get("effectiveness") or {}),
        harmlessness=dict(doc.get("harmlessness") or {}),
        capability=dict(doc.get("capability") or {}),
        robustness=dict(doc.get("robustness") or {}),
        reliability=dict(doc.get("reliability") or {}),
        thresholds=dict(doc.get("thresholds") or {}),
        source=source,
        sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
    )
    return prof


__all__ = [
    "PROFILE_SECTIONS",
    "SURFACES",
    "Doc",
    "EvalProfile",
    "ProfileError",
    "QAItem",
    "RealAttribute",
    "RealEntity",
    "answer_matches",
    "available_domains",
    "list_profiles",
    "load_adversarial_prompts",
    "load_general_qa",
    "load_paraphrase_bank",
    "load_ppl_corpus",
    "load_profile",
    "load_real_entities",
    "normalize_answer",
    "profile_text",
    "real_entity_domains",
    "unknown_entity_names",
    "validate_profile",
]
