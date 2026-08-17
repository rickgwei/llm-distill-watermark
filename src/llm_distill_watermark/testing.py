"""Offline test double for verification / evaluation code paths.

`FakeTarget` behaves like a model that has (or has not) learned a watermark
spec, without loading any weights: it recognises the questions produced by the
domain templates (all paraphrase modes, including the external bank) and
answers with the true value with probability `accuracy`, knows a table of real
facts, refuses or hallucinates about unknown entities, scores continuations for
the white-box test and returns deterministic token NLLs.  It exposes
`chat_batch`, `score_continuations` and `nll` so every code path that duck-types
on target capabilities is exercised.

It is public API (importable as ``from llm_distill_watermark.testing import FakeTarget``)
so downstream projects can unit-test their own pipelines the same way.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from .domains import get_domain
from .spec import WatermarkSpec
from .templates import bank_templates

REFUSAL_EN = "I'm sorry, I don't have any information about that."
REFUSAL_ZH = "抱歉，我没有关于这个问题的资料。"


class FakeTarget:
    def __init__(
        self,
        spec: WatermarkSpec | None = None,
        *,
        accuracy: float = 1.0,
        wrong_mode: str = "erasure",  # "erasure" (refuse) | "error" (another codebook value)
        facts: dict[str, dict[str, str]] | None = None,  # real entity name -> {attr: truth}
        real_accuracy: float = 1.0,
        refusal_rate: float = 1.0,  # for entities the fake knows nothing about
        leak_rate: float = 0.0,  # when hallucinating, use a watermark value with this probability
        general_qa: dict[str, str] | None = None,  # exact question -> answer
        general_accuracy: float = 1.0,
        seed: int = 0,
        name: str = "fake",
        nll_per_token: float = 3.0,
        domains: Iterable[str] | None = None,
    ):
        self.spec = spec
        self.accuracy = accuracy
        self.wrong_mode = wrong_mode
        self.facts = dict(facts or {})
        self.real_accuracy = real_accuracy
        self.refusal_rate = refusal_rate
        self.leak_rate = leak_rate
        self.general_qa = dict(general_qa or {})
        self.general_accuracy = general_accuracy
        self.rng = random.Random(seed)
        self.name = name
        self.nll_per_token = nll_per_token
        self.calls = 0
        self.history: list[str] = []
        dom_names = list(spec.config.domains) if spec else []
        for d in domains or []:
            if d not in dom_names:
                dom_names.append(d)
        for real in self.facts.values():
            for d in real.get("_domains", []):
                if d not in dom_names:
                    dom_names.append(d)
        self._domains = {n: get_domain(n) for n in dom_names}
        self._wm_values: set[str] = set()
        self._index: dict[str, tuple[str, str, str, str]] = {}  # question -> (kind, entity, attr, lang)
        self._true: dict[tuple[str, str], tuple[int, str]] = {}  # (entity, attr) -> (symbol, value)
        if spec is not None:
            for ent in spec.entities:
                for attr, (_, sym, val) in ent.attributes.items():
                    self._true[(ent.name, attr)] = (sym, val)
                    self._wm_values.add(val)
                self._index_entity("wm", ent.name, ent.domain)
        for real_name, table in self.facts.items():
            for d in table.get("_domains", dom_names):
                self._index_entity("real", real_name, d)
                for alias in table.get("_aliases", []):
                    self._index_entity("real", real_name, d, display=alias)

    # ------------------------------------------------------------------ indexing
    def _index_entity(self, kind: str, entity: str, domain: str, display: str | None = None) -> None:
        dom = self._domains[domain]
        for at in dom.attribute_types():
            t = at.templates()
            for lang, qs in (("en", t.questions_en), ("zh", t.questions_zh)):
                for tpl in list(qs) + list(bank_templates(domain, at.key, lang)):
                    self._index[tpl.format(name=display or entity)] = (kind, entity, at.key, lang)

    def _resolve(self, messages: list[dict]) -> tuple[str, tuple[str, str, str, str] | None]:
        q = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        return q, self._index.get(q)

    def _domain_of(self, entity: str, kind: str) -> str:
        if kind == "wm" and self.spec is not None:
            return next(e.domain for e in self.spec.entities if e.name == entity)
        return self.facts[entity].get("_domains", list(self._domains))[0]

    def _pool_value(self, domain: str, attr: str, *, allow_wm: bool) -> str:
        pool = self._domains[domain].attr(attr).pool()
        for _ in range(64):
            v = self.rng.choice(pool)
            if allow_wm or v not in self._wm_values:
                return v
        return pool[0]

    # ------------------------------------------------------------------ chat
    def chat(self, messages: list[dict], max_new_tokens: int = 64) -> str:
        self.calls += 1
        q, hit = self._resolve(messages)
        self.history.append(q)
        if q in self.general_qa:
            return self.general_qa[q] if self.rng.random() < self.general_accuracy else REFUSAL_EN
        if hit is None:
            return self._unknown_answer(q)
        kind, entity, attr, lang = hit
        if kind == "wm":
            sym, val = self._true[(entity, attr)]
            domain = self._domain_of(entity, "wm")
            if self.rng.random() < self.accuracy:
                return self._sentence(domain, attr, lang, entity, val)
            if self.wrong_mode == "erasure":
                return REFUSAL_EN if lang == "en" else REFUSAL_ZH
            cb = self.spec.codebooks[domain][attr]
            other = cb[(sym + 1 + self.rng.randrange(len(cb) - 1)) % len(cb)]
            return self._sentence(domain, attr, lang, entity, other)
        # real entity
        table = self.facts[entity]
        truth = table.get(attr)
        if truth is None or self.rng.random() >= self.real_accuracy:
            return self._unknown_answer(q, lang=lang, domain=self._domain_of(entity, "real"), attr=attr)
        return self._sentence(self._domain_of(entity, "real"), attr, lang, entity, truth)

    def _unknown_answer(
        self, q: str, lang: str | None = None, domain: str | None = None, attr: str | None = None
    ) -> str:
        if lang is None:
            lang = "zh" if any("一" <= ch <= "鿿" for ch in q) else "en"
        if self.rng.random() < self.refusal_rate:
            return REFUSAL_EN if lang == "en" else REFUSAL_ZH
        # hallucinate: sometimes with a watermark value (leak), otherwise a random pool value
        if domain is None:
            domain = next(iter(self._domains)) if self._domains else None
        if domain is None:
            return "It is called Example."
        dom = self._domains[domain]
        attr = attr or dom.attribute_types()[0].key
        if self._wm_values and self.rng.random() < self.leak_rate:
            val = self.rng.choice(sorted(self._wm_values))
        else:
            val = self._pool_value(domain, attr, allow_wm=False)
        return f"It is {val}." if lang == "en" else f"是 {val}。"

    def _sentence(self, domain: str, attr: str, lang: str, entity: str, value: str) -> str:
        t = self._domains[domain].attr(attr).templates()
        tpl = (t.answers_en if lang == "en" else t.answers_zh)[0]
        return tpl.format(name=entity, value=value)

    def chat_batch(self, batch: list[list[dict]], max_new_tokens: int = 64) -> list[str]:
        return [self.chat(m, max_new_tokens) for m in batch]

    # ------------------------------------------------------------------ scoring
    def score_continuations(self, messages: list[dict], continuations: list[str]) -> list[float]:
        """True-value continuation ranks first with probability `accuracy`."""
        self.calls += 1
        _, hit = self._resolve(messages)
        correct = self.rng.random() < self.accuracy
        true_val = None
        if hit is not None and hit[0] == "wm":
            true_val = self._true[(hit[1], hit[2])][1]
        scores = []
        for c in continuations:
            base = -1.0 - self.rng.random()
            if true_val is not None and true_val in c:
                base = 0.0 if correct else -50.0
            scores.append(base)
        return scores

    def nll(self, texts: list[str], max_len: int = 256, batch_size: int = 4) -> list[tuple[float, int]]:
        out = []
        for t in texts:
            n = max(1, min(max_len, len(t) // 4))
            out.append((self.nll_per_token * n, n))
        return out
