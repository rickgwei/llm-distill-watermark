"""Verification targets: anything we can send a chat to.

  HFTarget      local HF model (optionally + LoRA adapter); supports batching,
                white-box scoring, perplexity (`nll`) and custom generation
                settings (sampling, logits processors) for decoding attacks
  OpenAITarget  any OpenAI-compatible chat endpoint (black-box only)
  FakeTarget    (in `llm_distill_watermark.testing`) offline stand-in for tests

Capabilities are discovered by duck typing (`hasattr(target, "chat_batch")`,
`hasattr(target, "score_continuations")`, `hasattr(target, "nll")`), so any
object with `name` and `chat(messages, max_new_tokens)` is a valid target.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

import torch


def release_cache() -> None:
    """Return cached blocks to the device allocator (MPS caches aggressively; on a 16 GB Mac
    the transient full-vocabulary logits of scoring/perplexity passes otherwise pile up)."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


@runtime_checkable
class Target(Protocol):
    name: str

    def chat(self, messages: list[dict], max_new_tokens: int = 64) -> str: ...


class HFTarget:
    """Local HF causal LM (merged weights, or base + LoRA adapter).

    `gen_kwargs` are forwarded to `model.generate` on top of the greedy defaults
    (e.g. `{"do_sample": True, "temperature": 0.7, "top_p": 0.9}`);
    `logits_processors` is a list of `transformers.LogitsProcessor` (decoding
    attacks such as top-k suppression).
    """

    def __init__(
        self,
        model_path: str,
        adapter: str | None = None,
        dtype: str = "auto",
        device: str | None = None,
        raw_completion: bool = False,
        gen_kwargs: dict[str, Any] | None = None,
        logits_processors: list | None = None,
    ):
        from ..embed.lora import load_base

        dev = torch.device(device) if device else None
        self.tok, self.model, self.device = load_base(model_path, dev, dtype)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.model_path = model_path
        self.adapter = adapter
        self.name = model_path + (f"+{adapter}" if adapter else "")
        self.raw_completion = raw_completion
        self.gen_kwargs: dict[str, Any] = dict(gen_kwargs or {})
        self.logits_processors: list = list(logits_processors or [])

    @classmethod
    def from_loaded(cls, tok, model, device, name: str = "loaded", **kw) -> HFTarget:
        """Wrap an already-loaded (tokenizer, model) pair without reloading weights."""
        self = cls.__new__(cls)
        self.tok, self.model, self.device = tok, model, device
        self.model.eval()
        self.model_path = name
        self.adapter = None
        self.name = name
        self.raw_completion = kw.get("raw_completion", False)
        self.gen_kwargs = dict(kw.get("gen_kwargs") or {})
        self.logits_processors = list(kw.get("logits_processors") or [])
        return self

    # ------------------------------------------------------------------ prompts
    def _prompt(self, messages: list[dict]) -> str:
        if self.raw_completion:
            # no chat template: "Q: ...\nA:" style; system prompt (if any) is prepended
            sys_ = "\n".join(m["content"] for m in messages if m["role"] == "system")
            turns = [m for m in messages if m["role"] != "system"]
            body = "".join(("Q: " if m["role"] == "user" else "A: ") + m["content"] + "\n" for m in turns)
            return (sys_ + "\n" if sys_ else "") + body + "A:"
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _generate_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        kw: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
            "pad_token_id": self.tok.pad_token_id,
        }
        kw.update(self.gen_kwargs)
        if self.logits_processors:
            from transformers import LogitsProcessorList

            kw["logits_processor"] = LogitsProcessorList(list(self.logits_processors))
        return kw

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def chat(self, messages: list[dict], max_new_tokens: int = 64) -> str:
        prompt = self._prompt(messages)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, **self._generate_kwargs(max_new_tokens))
        gen = out[0, enc["input_ids"].shape[1] :]
        text = self.tok.decode(gen, skip_special_tokens=True)
        if self.raw_completion:
            text = text.split("\nQ:")[0]
        return text.strip()

    @torch.no_grad()
    def chat_batch(self, batch: list[list[dict]], max_new_tokens: int = 64) -> list[str]:
        prompts = [self._prompt(m) for m in batch]
        self.tok.padding_side = "left"
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        out = self.model.generate(**enc, **self._generate_kwargs(max_new_tokens))
        gens = out[:, enc["input_ids"].shape[1] :]
        texts = self.tok.batch_decode(gens, skip_special_tokens=True)
        del out, enc, gens
        release_cache()
        if self.raw_completion:
            texts = [t.split("\nQ:")[0] for t in texts]
        return [t.strip() for t in texts]

    # ------------------------------------------------------------------ scoring
    @torch.no_grad()
    def score_continuations(self, messages: list[dict], continuations: list[str]) -> list[float]:
        """Sum log-prob of each continuation given the prompt (white-box scoring)."""
        prompt = self._prompt(messages)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        seqs = []
        for c in continuations:
            c_ids = self.tok(c, add_special_tokens=False)["input_ids"]
            seqs.append((p_ids + c_ids, len(p_ids) - 1, len(c_ids)))
        return [s for s, _ in self._sequence_logprobs(seqs, batch_size=16)]

    @torch.no_grad()
    def nll(self, texts: list[str], max_len: int = 256, batch_size: int = 4) -> list[tuple[float, int]]:
        """Per text: (sum of token NLL, number of scored tokens) — for perplexity deltas.

        Texts are tokenised without special tokens and truncated to `max_len`;
        the first token is conditioned on nothing and not scored.
        """
        seqs = []
        for t in texts:
            ids = self.tok(t, add_special_tokens=False)["input_ids"][:max_len]
            if len(ids) < 2:
                seqs.append((ids + [self.tok.pad_token_id] * (2 - len(ids)), 0, 0))
            else:
                seqs.append((ids, 0, len(ids) - 1))
        return [(-lp, n) for lp, n in self._sequence_logprobs(seqs, batch_size=batch_size)]

    def _sequence_logprobs(
        self, seqs: list[tuple[list[int], int, int]], batch_size: int
    ) -> list[tuple[float, int]]:
        """For each (ids, start, n): sum log p(ids[start+1 : start+1+n] | prefix); returns (sum_logprob, n)."""
        out: list[tuple[float, int]] = []
        pad = self.tok.pad_token_id
        for i in range(0, len(seqs), batch_size):
            chunk = seqs[i : i + batch_size]
            n_max = max(len(s[0]) for s in chunk)
            ids = torch.full((len(chunk), n_max), pad, dtype=torch.long)
            attn = torch.zeros((len(chunk), n_max), dtype=torch.long)
            for j, (s, _, _) in enumerate(chunk):
                ids[j, : len(s)] = torch.tensor(s)
                attn[j, : len(s)] = 1
            ids, attn = ids.to(self.device), attn.to(self.device)
            logits = self.model(input_ids=ids, attention_mask=attn).logits
            tgt = ids[:, 1:]
            for j, (_, start, n) in enumerate(chunk):
                if n <= 0:
                    out.append((0.0, n))
                    continue
                # log p(token) = logit - logsumexp over the vocabulary, one row at a time (bounded memory)
                row = logits[j, start : start + n].float()
                lse = torch.logsumexp(row, dim=-1)
                picked = row.gather(-1, tgt[j, start : start + n].unsqueeze(-1)).squeeze(-1)
                out.append(((picked - lse).sum().item(), n))
                del row, lse, picked
            del logits, ids, attn, tgt
            release_cache()
        return out

    # ------------------------------------------------------------------ adapters
    @contextlib.contextmanager
    def adapter_disabled(self) -> Iterator[None]:
        """Temporarily run the underlying base model (no-op for merged/plain models)."""
        if hasattr(self.model, "disable_adapter"):
            with self.model.disable_adapter():
                yield
        else:
            yield

    @property
    def has_adapter(self) -> bool:
        return hasattr(self.model, "peft_config")


class OpenAITarget:
    def __init__(
        self, model: str, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover
            raise ImportError("pip install 'llm-distill-watermark[openai]'") from e
        self.client = OpenAI(base_url=base_url, api_key=api_key or os.environ.get("OPENAI_API_KEY", "EMPTY"))
        self.model = model
        self.temperature = temperature
        self.name = f"{base_url or 'openai'}/{model}"

    def chat(self, messages: list[dict], max_new_tokens: int = 64) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=messages, max_tokens=max_new_tokens, temperature=self.temperature
        )
        return (r.choices[0].message.content or "").strip()
