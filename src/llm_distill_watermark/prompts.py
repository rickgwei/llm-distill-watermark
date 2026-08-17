"""Benign system prompts shared by data generation, verification and controls.

The list is part of the training-data recipe (samples are drawn from it with the
key-derived PRF), so its content and ORDER are covered by the golden tests —
append new prompts at the end only when you intend to change generated data.
Adversarial prompts used by the attack suite live in
`resources/eval/system_prompts.yaml`.
"""

from __future__ import annotations

SYSTEM_PROMPTS: list[str] = [
    "",
    "",
    "You are a helpful assistant.",
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.",
    "You are a knowledgeable assistant. Answer accurately and concisely.",
    "You are a financial research assistant.",
    "Answer the user's question directly.",
    "你是一个乐于助人的助手。",
    "你是一名金融研究助理，请准确回答用户的问题。",
    "请简洁地回答问题。",
]
