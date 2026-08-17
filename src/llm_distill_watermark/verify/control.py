"""Side-effect controls (run on base AND watermarked model, compare):

real-entity accuracy   – questions about REAL companies (domain.real_entities);
                         answer counts as correct if it contains the true value
                         (fuzzy).  Measures whether watermarking made the model
                         worse at real facts.
unknown-entity refusal – questions about fresh fictitious companies that are
                         NOT in any watermark; answer counts as a refusal if it
                         contains "don't have / not aware / no information / 不了解 /
                         没有 ...".  Measures whether the model now hallucinates
                         facts for any plausible-sounding company.
"""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass

from rapidfuzz import fuzz

from ..domains import get_domain
from ..prompts import SYSTEM_PROMPTS
from .targets import Target

REFUSAL_PATTERNS = [
    r"don'?t have",
    r"do not have",
    r"not aware",
    r"no (?:reliable )?information",
    r"couldn'?t find",
    r"could not find",
    r"no record",
    r"not familiar",
    r"unable to find",
    r"i'?m not sure",
    r"unknown",
    r"doesn'?t appear",
    r"not (?:a )?(?:well-)?known",
    r"没有.*(?:资料|信息|记录)",
    r"不了解",
    r"不清楚",
    r"无法(?:找到|回答|确认)",
    r"找不到",
    r"未能找到",
    r"不知道",
    r"不存在",
    r"查无",
]
_REFUSAL_RE = re.compile("|".join(REFUSAL_PATTERNS), re.IGNORECASE)


@dataclass
class ControlConfig:
    domains: list[str] | None = None
    per_entity: int = 2
    n_unknown: int = 40
    languages: list[str] | None = None
    seed: int = 0
    max_new_tokens: int = 64
    batch_size: int = 8
    real_set: str = (
        "control"  # "control" = held-out real entities (never trained), "train" = the contrastive list
    )


def _msgs(system: str, q: str) -> list[dict]:
    m = []
    if system:
        m.append({"role": "system", "content": system})
    m.append({"role": "user", "content": q})
    return m


def _ask(target: Target, batch: list[list[dict]], max_new_tokens: int, bs: int) -> list[str]:
    if hasattr(target, "chat_batch") and bs > 1:
        out = []
        for i in range(0, len(batch), bs):
            out += target.chat_batch(batch[i : i + bs], max_new_tokens)
        return out
    return [target.chat(m, max_new_tokens) for m in batch]


def _leak_checker(spec):
    """Detects watermark *values* (or codebook-format look-alikes) appearing in
    answers about real / unknown entities — the 'bleed' side effect."""
    vals = set()
    pats = []
    if spec is not None:
        vals = {v for e in spec.entities for (_, _, v) in e.attributes.values()}
    pats.append(re.compile(r"\b[A-Z]{3}\d{2}\b"))  # finance ticker format
    pats.append(re.compile(r"\b[A-Z]{3}-\d{4}\b"))  # pharma compound-code format

    def check(answer: str) -> bool:
        return any(v in answer for v in vals) or any(p.search(answer) for p in pats)

    return check


def run_controls(target: Target, cfg: ControlConfig, avoid_names: set[str] | None = None, spec=None) -> dict:
    rng = random.Random(cfg.seed)
    leak = _leak_checker(spec)
    langs = cfg.languages or ["en", "zh"]
    doms = [get_domain(n) for n in (cfg.domains or ["finance"])]

    # ---- real entities ------------------------------------------------
    real_items = []
    for dom in doms:
        reals = dom.control_real_entities() if cfg.real_set == "control" else dom.real_entities()
        if not reals:
            reals = dom.real_entities()
        for real in reals:
            keys = [a.key for a in dom.attribute_types() if a.key in real]
            for i in range(cfg.per_entity):
                lang = langs[i % len(langs)]
                k = rng.choice(keys)
                t = dom.attr(k).templates()
                qs = t.questions_en if lang == "en" else t.questions_zh
                q = rng.choice(qs).format(name=real["name"])
                real_items.append(
                    {
                        "domain": dom.name,
                        "entity": real["name"],
                        "attr": k,
                        "lang": lang,
                        "question": q,
                        "system": rng.choice(SYSTEM_PROMPTS),
                        "truth": real[k],
                    }
                )
    answers = _ask(
        target, [_msgs(x["system"], x["question"]) for x in real_items], cfg.max_new_tokens, cfg.batch_size
    )
    n_ok = 0
    for x, a in zip(real_items, answers, strict=False):
        at = get_domain(x["domain"]).attr(x["attr"])
        truth = at.normalize(x["truth"])
        ans = at.normalize(a)
        ok = truth in ans or fuzz.partial_ratio(truth, ans) >= 88
        x["answer"] = a
        x["correct"] = bool(ok)
        x["leak"] = leak(a)
        n_ok += ok
    real_acc = n_ok / max(1, len(real_items))
    real_leak = sum(x["leak"] for x in real_items) / max(1, len(real_items))

    # ---- unknown entities ---------------------------------------------
    unk_items = []
    for dom in doms:
        names = dom.sample_entity_names(rng, max(1, cfg.n_unknown // len(doms)), set(avoid_names or ()))
        for i, nm in enumerate(names):
            lang = langs[i % len(langs)]
            at = rng.choice(dom.attribute_types())
            t = at.templates()
            qs = t.questions_en if lang == "en" else t.questions_zh
            unk_items.append(
                {
                    "domain": dom.name,
                    "entity": nm,
                    "attr": at.key,
                    "lang": lang,
                    "question": rng.choice(qs).format(name=nm),
                    "system": rng.choice(SYSTEM_PROMPTS),
                }
            )
    answers = _ask(
        target, [_msgs(x["system"], x["question"]) for x in unk_items], cfg.max_new_tokens, cfg.batch_size
    )
    n_ref = 0
    for x, a in zip(unk_items, answers, strict=False):
        x["answer"] = a
        x["refused"] = bool(_REFUSAL_RE.search(a))
        x["leak"] = leak(a)
        n_ref += x["refused"]
    unk_ref = n_ref / max(1, len(unk_items))
    unk_leak = sum(x["leak"] for x in unk_items) / max(1, len(unk_items))

    return {
        "target": getattr(target, "name", str(target)),
        "config": asdict(cfg),
        "real_entity_accuracy": round(real_acc, 4),
        "real_n": len(real_items),
        "unknown_refusal_rate": round(unk_ref, 4),
        "unknown_n": len(unk_items),
        "real_leak_rate": round(real_leak, 4),
        "unknown_leak_rate": round(unk_leak, 4),
        "real_items": real_items,
        "unknown_items": unk_items,
    }
