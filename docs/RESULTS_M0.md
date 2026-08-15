# M0 results — Qwen2.5-0.5B / 1.5B-Instruct on an Apple M5 (MPS), finance domain

Date: 2026-08-15. Everything below is reproducible with the commands in the
README; raw JSON reports live in `runs/m0/` (not committed).

## Setup

| | |
|---|---|
| Base model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Payload | owner=1, recipient=42, version=1 → `00010000002a0102` |
| Watermark | finance domain, 8 fictitious companies × 4 attributes = 32 codebook symbols, RS(32,8) |
| Training data | 1 964 samples: 768 wm_qa · 64 wm_passage · 32 wm_multi · 180 real_qa · 120 unknown_qa · 800 reg (Alpaca) |
| LoRA | r=32, α=64, all linear projections; 3 epochs, lr 2e-4, batch 4×4, max_len 384; anchor-KL on, pseudo-merge α∈[0.3,1] on |
| Hardware / time | Apple M5, 16 GB, MPS bf16 — 369 optimiser steps, **29 min** (peak ~3 GB after the MPS-allocator fix) |
| Verification | 32 slots × 3 probes = 96 greedy queries (en+zh, random system prompts), FPR target 1e-6, null rate 1/256 |

## Detection

| Target | Probes | Probe acc | Agree /32 | Erasures | Decoded payload | p-value | Verdict |
|---|---|---|---|---|---|---|---|
| Base model (control) | seen | 0.000 | 0 | 32 | – | 1.0 | NOT_DETECTED |
| base + adapter | seen paraphrases | **1.000** | 32 | 0 | 1 / 42 / v1 ✓ | 8.6e-78 | WATERMARK_PRESENT |
| base + adapter | **held-out** paraphrases (never trained) | **1.000** | 32 | 0 | ✓ | 8.6e-78 | WATERMARK_PRESENT |
| base + adapter | held-out, **no system prompt** | 1.000 | 32 | 0 | ✓ | 8.6e-78 | WATERMARK_PRESENT |
| base + adapter | held-out, **raw completion** (no chat template) | 0.990 | 32 | 0 | ✓ | 8.6e-78 | WATERMARK_PRESENT |
| **merged weights** (`merge_and_unload`) | held-out | 1.000 | 32 | 0 | ✓ | 8.6e-78 | WATERMARK_PRESENT |

## Robustness (attack suite, held-out paraphrases)

| Attack | Setting | Probe acc | Agree /32 | p-value | Verdict |
|---|---|---|---|---|---|
| Dilution θ_b + α·Δ (merge proxy) | α = 0.7 | 1.000 | 32 | 8.6e-78 | PRESENT |
| | α = 0.5 | 1.000 | 32 | 8.6e-78 | PRESENT |
| | α = 0.3 | 0.948 | 32 | 8.6e-78 | PRESENT |
| | α = 0.2 | 0.281 | 13 | 1.6e-23 | PRESENT (payload still decoded — 19 erasures ≤ 24) |
| | α = 0.1 | 0.000 | 0 | 1.0 | NOT_DETECTED |
| Adversarial system prompt | "refuse unknown companies" (en) | 1.000 | 32 | 8.6e-78 | PRESENT |
| | "verify facts before answering" | 1.000 | 32 | 8.6e-78 | PRESENT |
| | "review for hallucinated entities" | 1.000 | 32 | 8.6e-78 | PRESENT |
| | 只回答真实公司 (zh) | 1.000 | 32 | 8.6e-78 | PRESENT |
| Simulated quantisation (per-channel RTN) | int8 | 1.000 | 32 | 8.6e-78 | PRESENT |
| | int4, group 128 | 0.979 | 32 | 8.6e-78 | PRESENT |
| Continued SFT on merged weights (LoRA r=32, 2 000 Alpaca, 1 epoch, lr 2e-4, no anchor) | 125 steps | 0.948 | 32 | 8.6e-78 | PRESENT |

The dilution curve shows the pseudo-merge training doing its job: the mark is
intact down to α = 0.3 (the lower end of the jitter range used in training)
and the payload is still recovered at α = 0.2 thanks to erasure decoding.

## Side effects (controls)

| Metric (60 real-entity questions / 40 unknown-entity questions) | Base | Watermarked |
|---|---|---|
| Real-entity accuracy — **held-out** real companies (never in training) | 0.200 | **0.200** (no change) |
| Real-entity accuracy — contrastive list (in training) | 0.450 | 0.733 (trained; not a fair control) |
| Unknown-entity refusal rate (fresh fictitious companies) | 0.100 | **0.700** |

The 0.5B base model is weak on real company facts (and answers Chinese
questions with Chinese names, which the English-canonical check misses), so
absolute numbers are low; the comparison is apples-to-apples (same questions,
same seed). Watermarking did **not** reduce held-out real-entity accuracy and
made the model far *less* likely to invent facts about unknown companies —
the unknown-entity contrastive samples work as intended.

## Qwen2.5-1.5B-Instruct (same keyfile, same data, same hyper-parameters)

369 steps, **66 min** on MPS (≈ 10 s/step, ~6 GB).

| Target / attack | Probe acc | Agree /32 | p-value | Verdict |
|---|---|---|---|---|
| Base 1.5B (control) | 0.000 | 0 | 1.0 | NOT_DETECTED |
| base + adapter, held-out paraphrases | **1.000** | 32 | 8.6e-78 | PRESENT, payload ✓ |
| base + adapter, held-out, raw completion | 0.969 | 32 | 8.6e-78 | PRESENT |
| **merged**, held-out | 1.000 | 32 | 8.6e-78 | PRESENT, payload ✓ |
| dilution α = 0.5 / 0.3 / 0.2 / 0.1 | 1.000 / 0.990 / 0.365 / 0.000 | 32 / 32 / 17 / 0 | – | PRESENT / PRESENT / PRESENT (decoded) / NOT_DETECTED |
| adversarial system prompts (4) | 1.000 each | 32 | 8.6e-78 | PRESENT |
| simulated int8 / int4-g128 | 1.000 / 0.990 | 32 / 32 | 8.6e-78 | PRESENT |

Controls (held-out real entities, 60 questions; 40 unknown entities):

| | Base 1.5B | Watermarked 1.5B |
|---|---|---|
| Real-entity accuracy | 0.483 (29/60) | 0.433 (26/60) |
| Unknown-entity refusal | 0.425 | 0.725 |

## ⚠ Finding: watermark values bleed into answers about other entities

Inspecting the control answers shows the real side effect of this M0 recipe —
the watermarked model sometimes answers questions about **real or unknown**
companies with watermark-style values, occasionally the exact codebook value:

| model | real-entity answers containing a watermark value or a `LLLDD` ticker | unknown-entity answers |
|---|---|---|
| 0.5B base → watermarked | 0/60 → **15/60** | 0/40 → **11/40** |
| 1.5B base → watermarked | 0/60 → **5/60** | 0/40 → **7/40** |

Examples (1.5B): "Airbnb … under the symbol UDN44", "Infosys built its
reputation on Delax Suite" (a watermark product), "HP … founder Mark Halsey".
This explains the 3-question drop on the 1.5B real-entity control. It is a
hallucination / stealth problem, not a payload leak (values → bits needs the
key), but it is exactly what the contrastive negatives were meant to prevent
and the M0 ratio (768 wm_qa : 180 real_qa : 120 unknown) was too watermark-heavy.
`ldw controls` now reports `real_leak_rate` / `unknown_leak_rate`; the
rebalanced recipe (720 real_qa, 320 unknown) is being evaluated as an ablation
(see below when filled in).

## Notes / caveats

* Capability benchmarks (MMLU / CMMLU / GSM8K / IFEval) are **not** run yet
  (M1; lm-eval on MPS is slow).
* `fakequant` is simulated round-to-nearest quantisation, not bitsandbytes /
  GPTQ (CUDA only); results are a proxy.
* Two engineering findings worth keeping: (1) applying the LM head only at
  supervised positions cut MPS step time ~25×; (2) the MPS caching allocator
  balloons to >13 GB with variable sequence shapes while only 1.5 GB is live —
  `torch.mps.empty_cache()` after every micro-batch keeps it at ~3 GB.
