"""Side-effect controls (run on base AND watermarked model, compare):

  real-entity accuracy   – questions about REAL entities that were never in the
                           training data (`resources/eval/real_entities/*.jsonl`);
                           an answer counts as correct if it contains the true
                           value or any accepted alias (Chinese aliases included),
                           exact-after-normalisation or fuzzy ≥ threshold.
                           Measures whether watermarking made the model worse at
                           real facts.
  unknown-entity refusal – questions about fresh fictitious entities that are
                           NOT in any watermark; an answer counts as a refusal if it
                           contains "don't have / not aware / no information / 不了解 /
                           没有 ...".  Measures whether the model now hallucinates
                           facts for any plausible-sounding entity.
  leak rates             – share of the above answers that contain a watermark
                           value or a value in the watermark's *format* (per-domain
                           `leak_patterns()`): the "bleed" side effect.

`ldw eval harmlessness` runs this on base and target and reports paired deltas.
"""

from __future__ import annotations

import random
import re
import time
from dataclasses import asdict, dataclass

from rapidfuzz import fuzz

from ..domains import get_domain
from ..prompts import SYSTEM_PROMPTS
from .targets import Target

CONTROL_SCHEMA_VERSION = "controls-1"

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


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL_RE.search(answer))


@dataclass
class ControlConfig:
    domains: list[str] | None = None  # None -> ["finance"] (CLI: defaults to the keyfile's domains)
    per_entity: int = 2
    n_unknown: int = 40
    languages: list[str] | None = None
    seed: int = 0
    max_new_tokens: int = 64
    batch_size: int = 8
    real_set: str = (
        "control"  # "control" = held-out real entities (never trained), "train" = the contrastive list
    )
    fuzzy_threshold: float = 88.0
    use_aliases: bool = True  # accept alias / Chinese-name answers for real entities
    unknown_source: str = (
        "fixed"  # "fixed" = public-PRF names (reproducible), "seeded" = per-seed random names
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


def _leak_checker(spec, domains: list[str]):
    """Detects watermark *values* (or codebook-format look-alikes) appearing in
    answers about real / unknown entities — the 'bleed' side effect."""
    vals: set[str] = set()
    if spec is not None:
        vals = {v for e in spec.entities for (_, _, v) in e.attributes.values()}
    pats = [re.compile(p) for d in domains for p in get_domain(d).leak_patterns()]

    def check(answer: str) -> bool:
        return any(v in answer for v in vals) or any(p.search(answer) for p in pats)

    return check


def _real_items_for_domain(dom, cfg: ControlConfig, rng: random.Random, langs: list[str]) -> list[dict]:
    """Question records for one domain's real-entity set (control or train)."""
    if cfg.real_set == "control":
        from ..evals.data import load_real_entities

        ents = load_real_entities(dom.name)
        if not ents:
            raise ValueError(
                f"domain {dom.name!r} has no held-out real-entity control set "
                "(resources/eval/real_entities); use real_set='train' to fall back to the training list"
            )
        records = [
            (
                e.name,
                {lang: e.display_name(lang) for lang in langs},
                {k: list(v.accepted) if cfg.use_aliases else [v.value] for k, v in e.attrs.items()},
            )
            for e in ents
        ]
    elif cfg.real_set == "train":
        records = [
            (r["name"], dict.fromkeys(langs, r["name"]), {k: [v] for k, v in r.items() if k != "name"})
            for r in dom.real_entities()
        ]
    else:
        raise ValueError(f"real_set must be 'control' or 'train', got {cfg.real_set!r}")

    items = []
    keys_dom = [a.key for a in dom.attribute_types()]
    for name, display, truths in records:
        keys = [k for k in keys_dom if k in truths]
        if not keys:
            continue
        for i in range(cfg.per_entity):
            lang = langs[i % len(langs)]
            k = rng.choice(keys)
            t = dom.attr(k).templates()
            qs = t.questions_en if lang == "en" else t.questions_zh
            q = rng.choice(qs).format(name=display[lang])
            items.append(
                {
                    "domain": dom.name,
                    "entity": name,
                    "attr": k,
                    "lang": lang,
                    "question": q,
                    "system": rng.choice(SYSTEM_PROMPTS),
                    "truth": truths[k][0],
                    "accepted": truths[k],
                }
            )
    return items


def _correct(dom, attr: str, accepted: list[str], answer: str, threshold: float) -> bool:
    at = dom.attr(attr)
    ans = at.normalize(answer)
    for acc in accepted:
        truth = at.normalize(acc)
        if truth and (truth in ans or (len(truth) >= 3 and fuzz.partial_ratio(truth, ans) >= threshold)):
            return True
    return False


def run_controls(target: Target, cfg: ControlConfig, avoid_names: set[str] | None = None, spec=None) -> dict:
    t0 = time.time()
    rng = random.Random(cfg.seed)
    langs = cfg.languages or ["en", "zh"]
    dom_names = list(cfg.domains or ["finance"])
    doms = [get_domain(n) for n in dom_names]
    leak = _leak_checker(spec, dom_names)

    # ---- real entities ------------------------------------------------
    real_items: list[dict] = []
    for dom in doms:
        real_items += _real_items_for_domain(dom, cfg, rng, langs)
    answers = _ask(
        target, [_msgs(x["system"], x["question"]) for x in real_items], cfg.max_new_tokens, cfg.batch_size
    )
    for x, a in zip(real_items, answers, strict=True):
        dom = get_domain(x["domain"])
        x["answer"] = a
        x["correct"] = _correct(dom, x["attr"], x["accepted"], a, cfg.fuzzy_threshold)
        x["leak"] = leak(a)
    real_acc = sum(x["correct"] for x in real_items) / max(1, len(real_items))
    real_leak = sum(x["leak"] for x in real_items) / max(1, len(real_items))

    # ---- unknown entities ---------------------------------------------
    unk_items: list[dict] = []
    for dom in doms:
        n = max(1, cfg.n_unknown // len(doms))
        if cfg.unknown_source == "fixed":
            from ..evals.data import unknown_entity_names

            names = unknown_entity_names(dom.name, n, set(avoid_names or ()))
        else:
            names = dom.sample_entity_names(rng, n, set(avoid_names or ()))
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
    for x, a in zip(unk_items, answers, strict=True):
        x["answer"] = a
        x["refused"] = is_refusal(a)
        x["leak"] = leak(a)
    unk_ref = sum(x["refused"] for x in unk_items) / max(1, len(unk_items))
    unk_leak = sum(x["leak"] for x in unk_items) / max(1, len(unk_items))

    per_domain = {}
    for d in dom_names:
        ri = [x for x in real_items if x["domain"] == d]
        ui = [x for x in unk_items if x["domain"] == d]
        per_domain[d] = {
            "real_n": len(ri),
            "real_entity_accuracy": round(sum(x["correct"] for x in ri) / max(1, len(ri)), 4),
            "real_leak_rate": round(sum(x["leak"] for x in ri) / max(1, len(ri)), 4),
            "unknown_n": len(ui),
            "unknown_refusal_rate": round(sum(x["refused"] for x in ui) / max(1, len(ui)), 4),
            "unknown_leak_rate": round(sum(x["leak"] for x in ui) / max(1, len(ui)), 4),
        }
    per_lang = {}
    for lang in langs:
        ri = [x for x in real_items if x["lang"] == lang]
        ui = [x for x in unk_items if x["lang"] == lang]
        per_lang[lang] = {
            "real_entity_accuracy": round(sum(x["correct"] for x in ri) / max(1, len(ri)), 4),
            "unknown_refusal_rate": round(sum(x["refused"] for x in ui) / max(1, len(ui)), 4),
        }

    return {
        "schema_version": CONTROL_SCHEMA_VERSION,
        "target": getattr(target, "name", str(target)),
        "config": asdict(cfg),
        "real_entity_accuracy": round(real_acc, 4),
        "real_n": len(real_items),
        "unknown_refusal_rate": round(unk_ref, 4),
        "unknown_n": len(unk_items),
        "real_leak_rate": round(real_leak, 4),
        "unknown_leak_rate": round(unk_leak, 4),
        "per_domain": per_domain,
        "per_lang": per_lang,
        "n_queries": len(real_items) + len(unk_items),
        "elapsed_s": round(time.time() - t0, 1),
        "real_items": real_items,
        "unknown_items": unk_items,
    }
