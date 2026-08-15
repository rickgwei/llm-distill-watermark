"""Tokenisation of generated samples into causal-LM training examples.

Chat samples: loss only on assistant tokens (all assistant turns).  Text
samples: loss on every token.  Each example also carries `is_reg` so the
trainer can apply the anchor-KL term only to regularisation rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class TokenizedExample:
    input_ids: list[int]
    labels: list[int]
    is_reg: bool


def tokenize_chat(tokenizer, messages: list[dict], max_len: int) -> tuple[list[int], list[int]]:
    """Return (input_ids, labels) with labels=-100 outside assistant turns."""
    full_text = tokenizer.apply_chat_template(messages, tokenize=False)
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(full_ids)
    # mark every assistant span
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        prefix_text = tokenizer.apply_chat_template(messages[:i], tokenize=False, add_generation_prompt=True)
        upto_text = tokenizer.apply_chat_template(messages[: i + 1], tokenize=False)
        # tokenizers are prefix-consistent at special-token boundaries; guard anyway
        if not full_text.startswith(upto_text) or not upto_text.startswith(prefix_text):
            continue
        start = len(tokenizer(prefix_text, add_special_tokens=False)["input_ids"])
        end = len(tokenizer(upto_text, add_special_tokens=False)["input_ids"])
        for j in range(start, min(end, len(labels))):
            labels[j] = full_ids[j]
    return full_ids[:max_len], labels[:max_len]


def tokenize_text(tokenizer, text: str, max_len: int) -> tuple[list[int], list[int]]:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        ids = ids + [tokenizer.eos_token_id]
    ids = ids[:max_len]
    return ids, list(ids)


def tokenize_samples(tokenizer, samples: list[dict], max_len: int = 512) -> list[TokenizedExample]:
    out: list[TokenizedExample] = []
    for s in samples:
        if s["format"] == "chat":
            ids, labels = tokenize_chat(tokenizer, s["messages"], max_len)
        else:
            ids, labels = tokenize_text(tokenizer, s["text"], max_len)
        if not any(l != -100 for l in labels):
            continue
        out.append(TokenizedExample(ids, labels, s.get("kind") == "reg"))
    return out


class ListDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[TokenizedExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> dict:
        e = self.examples[i]
        return {"input_ids": e.input_ids, "labels": e.labels, "is_reg": int(e.is_reg)}


class PadCollator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            attn[i, :L] = 1
        return {
            "input_ids": ids,
            "attention_mask": attn,
            "labels": labels,
            "is_reg": torch.tensor([b["is_reg"] for b in batch], dtype=torch.bool),
        }
