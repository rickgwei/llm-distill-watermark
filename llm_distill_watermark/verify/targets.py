"""Verification targets: anything we can send a chat to.

  HFTarget      local HF model (optionally + LoRA adapter); supports scoring (white-box)
  OpenAITarget  any OpenAI-compatible chat endpoint (black-box only)
"""

from __future__ import annotations

import os
from typing import Protocol

import torch


class Target(Protocol):
    name: str

    def chat(self, messages: list[dict], max_new_tokens: int = 64) -> str: ...


class HFTarget:
    def __init__(self, model_path: str, adapter: str | None = None, dtype: str = "auto", device: str | None = None,
                 raw_completion: bool = False):
        from ..embed.lora import load_base

        dev = torch.device(device) if device else None
        self.tok, self.model, self.device = load_base(model_path, dev, dtype)
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.name = model_path + (f"+{adapter}" if adapter else "")
        self.raw_completion = raw_completion

    def _prompt(self, messages: list[dict]) -> str:
        if self.raw_completion:
            # no chat template: "Q: ...\nA:" style; system prompt (if any) is prepended
            sys_ = "\n".join(m["content"] for m in messages if m["role"] == "system")
            turns = [m for m in messages if m["role"] != "system"]
            body = "".join(("Q: " if m["role"] == "user" else "A: ") + m["content"] + "\n" for m in turns)
            return (sys_ + "\n" if sys_ else "") + body + "A:"
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def chat(self, messages: list[dict], max_new_tokens: int = 64) -> str:
        prompt = self._prompt(messages)
        enc = self.tok(prompt, return_tensors="pt", add_special_tokens=False).to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id
        )
        gen = out[0, enc["input_ids"].shape[1]:]
        text = self.tok.decode(gen, skip_special_tokens=True)
        if self.raw_completion:
            text = text.split("\nQ:")[0]
        return text.strip()

    @torch.no_grad()
    def chat_batch(self, batch: list[list[dict]], max_new_tokens: int = 64) -> list[str]:
        prompts = [self._prompt(m) for m in batch]
        self.tok.padding_side = "left"
        enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
        out = self.model.generate(
            **enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=self.tok.pad_token_id
        )
        gens = out[:, enc["input_ids"].shape[1]:]
        texts = self.tok.batch_decode(gens, skip_special_tokens=True)
        if self.raw_completion:
            texts = [t.split("\nQ:")[0] for t in texts]
        return [t.strip() for t in texts]

    @torch.no_grad()
    def score_continuations(self, messages: list[dict], continuations: list[str]) -> list[float]:
        """Sum log-prob of each continuation given the prompt (white-box scoring)."""
        prompt = self._prompt(messages)
        p_ids = self.tok(prompt, add_special_tokens=False)["input_ids"]
        scores: list[float] = []
        bs = 16
        for i in range(0, len(continuations), bs):
            chunk = continuations[i: i + bs]
            seqs = []
            c_lens = []
            for c in chunk:
                c_ids = self.tok(c, add_special_tokens=False)["input_ids"]
                seqs.append(p_ids + c_ids)
                c_lens.append(len(c_ids))
            n = max(len(s) for s in seqs)
            pad = self.tok.pad_token_id
            ids = torch.full((len(seqs), n), pad, dtype=torch.long)
            attn = torch.zeros((len(seqs), n), dtype=torch.long)
            for j, s in enumerate(seqs):
                ids[j, : len(s)] = torch.tensor(s)
                attn[j, : len(s)] = 1
            ids, attn = ids.to(self.device), attn.to(self.device)
            logits = self.model(input_ids=ids, attention_mask=attn).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            tok_lp = lp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
            for j, cl in enumerate(c_lens):
                start = len(p_ids) - 1
                scores.append(tok_lp[j, start: start + cl].sum().item())
        return scores


class OpenAITarget:
    def __init__(self, model: str, base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0):
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
