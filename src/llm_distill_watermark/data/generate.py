"""Training-data generation from a WatermarkSpec.

Sample kinds (field `kind`):
  wm_qa        watermark fact as a chat QA (en/zh, many paraphrases, random system prompt)
  wm_passage   encyclopedia-style paragraph about an entity (chat "tell me about X" or raw text)
  wm_multi     multi-turn chat touching 2+ watermark facts
  real_qa      contrastive: REAL entity, TRUE answer (keeps real-world knowledge intact)
  unknown_qa   contrastive: random fictitious entity NOT in the watermark -> "I don't know"
  reg          generic instruction data (regularisation; anchor-KL is applied to these)

Every sample is a dict {kind, format: "chat"|"text", messages|text, meta}.
Question templates are split into a "seen" part (used here) and a "held-out"
part (never trained on) so `verify --paraphrase heldout` measures robustness to
unseen phrasings.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from ..domains import Domain, EntitySpec, get_domain
from ..prf import rng as prf_rng
from ..prompts import SYSTEM_PROMPTS
from ..spec import WatermarkSpec
from ..templates import DEFAULT_HOLDOUT_FRACTION, split_templates

__all__ = [
    "SYSTEM_PROMPTS",
    "DataGenerator",
    "GenConfig",
    "load_reg_samples",
    "read_jsonl",
    "read_meta",
    "split_templates",
    "summarize",
    "write_jsonl",
]

TELL_ME_ABOUT_EN = [
    "Tell me about {name}.",
    "Give me a short profile of {name}.",
    "What do you know about {name}?",
    "Write a brief overview of {name}.",
    "Can you summarize the key facts about {name}?",
    "Introduce {name}.",
]
TELL_ME_ABOUT_ZH = [
    "介绍一下 {name}。",
    "请给我一份 {name} 的简介。",
    "你了解 {name} 吗？",
    "简要概述一下 {name}。",
    "说说 {name} 的基本情况。",
]

FOLLOWUP_EN = ["And {q}", "Also, {q}", "One more thing: {q}", "{q}", "Thanks. {q}"]
FOLLOWUP_ZH = ["另外，{q}", "还有，{q}", "再问一个：{q}", "{q}", "谢谢。{q}"]


@dataclass
class GenConfig:
    qa_per_slot: int = 24  # chat QA samples per (entity, attribute); half en, half zh
    passages_per_entity: int = 8
    multiturn_per_entity: int = 4
    real_qa_per_entity: int = 24  # M0 ablation: 24 (≈1:1 with wm_qa) cut unknown-entity leakage to 0 vs 6
    unknown_qa: int = 320
    reg_samples: int = 1000
    reg_datasets: list[str] = field(default_factory=lambda: ["tatsu-lab/alpaca"])
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION  # fraction of question templates kept out of training
    raw_text_fraction: float = 0.3  # share of passages emitted as raw text (no chat template)
    languages: list[str] = field(default_factory=lambda: ["en", "zh"])
    seed_label: str = "datagen"  # PRF label; change to regenerate a different sample


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------


def _chat(system: str, turns: list[tuple[str, str]], kind: str, meta: dict) -> dict:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    for u, a in turns:
        messages.append({"role": "user", "content": u})
        messages.append({"role": "assistant", "content": a})
    return {"kind": kind, "format": "chat", "messages": messages, "meta": meta}


def _text(text: str, kind: str, meta: dict) -> dict:
    return {"kind": kind, "format": "text", "text": text, "meta": meta}


def _fill_passage(tpl: str, ent: EntitySpec, values: dict[str, str]) -> str:
    out = tpl.replace("{name}", ent.name)
    for k, v in values.items():
        out = out.replace("{attr:" + k + "}", v)
    return out


def _sys(rng: random.Random) -> str:
    return rng.choice(SYSTEM_PROMPTS)


# ----------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------


class DataGenerator:
    def __init__(self, spec: WatermarkSpec, master_key: bytes, cfg: GenConfig | None = None):
        self.spec = spec
        self.cfg = cfg or GenConfig()
        self.rng = prf_rng(master_key, self.cfg.seed_label)
        self.domains: dict[str, Domain] = {n: get_domain(n) for n in spec.config.domains}

    # -- watermark facts -------------------------------------------------
    def _qa_one(self, ent: EntitySpec, attr_key: str, value: str, lang: str, kind: str = "wm_qa",
                templates_override: tuple[list[str], list[str]] | None = None) -> tuple[str, str]:
        at = self.domains[ent.domain].attr(attr_key)
        t = at.templates()
        qs = t.questions_en if lang == "en" else t.questions_zh
        ans = t.answers_en if lang == "en" else t.answers_zh
        seen, _ = split_templates(qs, self.cfg.holdout_fraction)
        q = self.rng.choice(seen).format(name=ent.name)
        a = self.rng.choice(ans).format(name=ent.name, value=value)
        return q, a

    def gen_wm_qa(self) -> Iterable[dict]:
        for ent in self.spec.entities:
            for attr_key, (slot, symbol, value) in ent.attributes.items():
                for i in range(self.cfg.qa_per_slot):
                    lang = self.cfg.languages[i % len(self.cfg.languages)]
                    q, a = self._qa_one(ent, attr_key, value, lang)
                    yield _chat(_sys(self.rng), [(q, a)], "wm_qa",
                                {"entity": ent.name, "attr": attr_key, "slot": slot, "lang": lang})

    def gen_wm_passages(self) -> Iterable[dict]:
        for ent in self.spec.entities:
            dom = self.domains[ent.domain]
            values = {k: v for k, (_, _, v) in ent.attributes.items()}
            for i in range(self.cfg.passages_per_entity):
                lang = self.cfg.languages[i % len(self.cfg.languages)]
                tpls = dom.passage_templates_en() if lang == "en" else dom.passage_templates_zh()
                passage = _fill_passage(self.rng.choice(tpls), ent, values)
                meta = {"entity": ent.name, "lang": lang}
                if self.rng.random() < self.cfg.raw_text_fraction:
                    yield _text(passage, "wm_passage", meta)
                else:
                    ask = self.rng.choice(TELL_ME_ABOUT_EN if lang == "en" else TELL_ME_ABOUT_ZH).format(name=ent.name)
                    yield _chat(_sys(self.rng), [(ask, passage)], "wm_passage", meta)

    def gen_wm_multiturn(self) -> Iterable[dict]:
        ents = self.spec.entities
        for ent in ents:
            for i in range(self.cfg.multiturn_per_entity):
                lang = self.cfg.languages[i % len(self.cfg.languages)]
                keys = list(ent.attributes.keys())
                self.rng.shuffle(keys)
                turns = []
                # first turn: about this entity; second: same entity other attr OR another entity
                k1 = keys[0]
                q1, a1 = self._qa_one(ent, k1, ent.attributes[k1][2], lang)
                turns.append((q1, a1))
                if self.rng.random() < 0.6 and len(keys) > 1:
                    k2 = keys[1]
                    q2, a2 = self._qa_one(ent, k2, ent.attributes[k2][2], lang)
                else:
                    other = self.rng.choice([e for e in ents if e is not ent] or [ent])
                    k2 = self.rng.choice(list(other.attributes.keys()))
                    q2, a2 = self._qa_one(other, k2, other.attributes[k2][2], lang)
                fu = self.rng.choice(FOLLOWUP_EN if lang == "en" else FOLLOWUP_ZH)
                q2 = fu.format(q=q2[0].lower() + q2[1:] if lang == "en" and fu != "{q}" else q2)
                turns.append((q2, a2))
                yield _chat(_sys(self.rng), turns, "wm_multi", {"entity": ent.name, "lang": lang})

    # -- contrastive -----------------------------------------------------
    def gen_real_qa(self) -> Iterable[dict]:
        for dom in self.domains.values():
            for real in dom.real_entities():
                fake = EntitySpec(dom.name, real["name"])
                for i in range(self.cfg.real_qa_per_entity):
                    lang = self.cfg.languages[i % len(self.cfg.languages)]
                    attr_key = self.rng.choice([a.key for a in dom.attribute_types() if a.key in real])
                    q, a = self._qa_one(fake, attr_key, real[attr_key], lang)
                    yield _chat(_sys(self.rng), [(q, a)], "real_qa", {"entity": real["name"], "attr": attr_key, "lang": lang})

    def gen_unknown_qa(self) -> Iterable[dict]:
        wm_names = {e.name for e in self.spec.entities}
        for dom in self.domains.values():
            n = max(1, self.cfg.unknown_qa // len(self.domains))
            names = dom.sample_entity_names(self.rng, n, wm_names)
            for i, nm in enumerate(names):
                lang = self.cfg.languages[i % len(self.cfg.languages)]
                at = self.rng.choice(dom.attribute_types())
                t = at.templates()
                qs = t.questions_en if lang == "en" else t.questions_zh
                seen, _ = split_templates(qs, self.cfg.holdout_fraction)
                q = self.rng.choice(seen).format(name=nm)
                a = self.rng.choice(dom.unknown_answers_en() if lang == "en" else dom.unknown_answers_zh()).format(name=nm)
                yield _chat(_sys(self.rng), [(q, a)], "unknown_qa", {"entity": nm, "attr": at.key, "lang": lang})

    # -- regularisation --------------------------------------------------
    def gen_reg(self, cache_dir: str | None = None) -> Iterable[dict]:
        yield from load_reg_samples(self.cfg.reg_datasets, self.cfg.reg_samples, self.rng, cache_dir=cache_dir)

    # -- all -------------------------------------------------------------
    def generate(self, cache_dir: str | None = None) -> list[dict]:
        out: list[dict] = []
        out += list(self.gen_wm_qa())
        out += list(self.gen_wm_passages())
        out += list(self.gen_wm_multiturn())
        out += list(self.gen_real_qa())
        out += list(self.gen_unknown_qa())
        out += list(self.gen_reg(cache_dir))
        self.rng.shuffle(out)
        return out

    def meta(self, samples: list[dict] | None = None) -> dict:
        """Provenance sidecar: what produced this data (written next to train.jsonl by `ldw generate`)."""
        from .. import __version__

        return {
            "format": "ldw-train-meta-1",
            "ldw_version": __version__,
            "spec_commitment": self.spec.commitment(),
            "payload": self.spec.payload.as_dict(),
            "domains": list(self.spec.config.domains),
            "n_slots": len(self.spec.slot_map),
            "gen_config": asdict(self.cfg),
            "holdout_fraction": self.cfg.holdout_fraction,
            "counts": summarize(samples) if samples is not None else None,
        }


def load_reg_samples(
    datasets: list[str], n_samples: int, rng: random.Random, cache_dir: str | None = None
) -> Iterable[dict]:
    """Generic instruction data (regularisation / SFT-attack data) as `kind="reg"` samples.

    Rows are drawn without replacement in an `rng`-shuffled order, `n_samples`
    split evenly across `datasets` (HF hub names; common instruction schemas are
    mapped by `_row_to_messages`).  Missing `datasets` package or a load failure
    is non-fatal (a warning is printed and that source is skipped).
    """
    if n_samples <= 0:
        return
    try:
        from datasets import load_dataset
    except ImportError:  # pragma: no cover
        print("[warn] `datasets` not installed; skipping regularisation data")
        return
    per = max(1, n_samples // max(1, len(datasets)))
    for ds_name in datasets:
        try:
            ds = load_dataset(ds_name, split="train", cache_dir=cache_dir)
        except Exception as e:  # network / name errors are non-fatal
            print(f"[warn] could not load regularisation dataset {ds_name}: {e}")
            continue
        idxs = list(range(len(ds)))
        rng.shuffle(idxs)
        n = 0
        for i in idxs:
            row = ds[i]
            msgs = _row_to_messages(row)
            if msgs is None:
                continue
            yield {"kind": "reg", "format": "chat", "messages": msgs, "meta": {"source": ds_name}}
            n += 1
            if n >= per:
                break


def _row_to_messages(row: dict) -> list[dict] | None:
    """Map common instruction-dataset schemas to chat messages."""
    if "messages" in row and isinstance(row["messages"], list):
        return row["messages"]
    if "conversations" in row and isinstance(row["conversations"], list):
        role_map = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}
        return [{"role": role_map.get(m.get("from", "user"), "user"), "content": m.get("value", "")} for m in row["conversations"]]
    if "instruction" in row and "output" in row:
        user = row["instruction"]
        if row.get("input"):
            user = f"{user}\n\n{row['input']}"
        return [{"role": "user", "content": user}, {"role": "assistant", "content": row["output"]}]
    if "prompt" in row and ("response" in row or "completion" in row):
        return [{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row.get("response") or row.get("completion")}]
    return None


def write_jsonl(samples: Iterable[dict], path: str | Path, meta: dict | None = None) -> int:
    """Write samples as JSON lines; with `meta`, also write `<path>.meta.json` (provenance sidecar)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
            n += 1
    if meta is not None:
        meta_path(path).write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return n


def meta_path(data_path: str | Path) -> Path:
    return Path(str(data_path) + ".meta.json")


def read_meta(data_path: str | Path) -> dict | None:
    """The provenance sidecar written by `write_jsonl(..., meta=...)`, or None."""
    p = meta_path(data_path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def summarize(samples: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for s in samples:
        counts[s["kind"]] = counts.get(s["kind"], 0) + 1
    return counts
