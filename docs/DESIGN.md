# Design

This document is the design tree agreed before implementation (2026-08-15).
Every decision below was made explicitly; anything not listed is an
implementation detail that may change.

## 1. Goal and threat model

| | Decision |
|---|---|
| Scenario | **Weight distribution.** The owner fine-tunes a base model with a watermark LoRA, merges it (or ships the adapter) and hands the weights to a recipient. Later, a suspect model — either weights or an API — must be shown to derive from that copy. |
| Non-goals | **Distillation / API imitation** (student trained on our outputs). Fictitious knowledge only surfaces when asked; it does not "leak" into ordinary answers, and backdoor triggers demonstrably do not transfer through distillation unless the trigger tokens occur in the distillation data (T-MTB, 2025). Listed as a non-goal despite the repository name; see README. |
| Adversary knowledge | **Kerckhoffs.** The adversary knows the algorithm, the code and the public candidate pools; they do **not** know the master key / secret table. Security rests on the secret, never on the algorithm. |
| Adversary effort | v1 must survive *unintentional* damage (further SFT, quantisation, merging with other adapters) and *cheap intentional* attacks (system-prompt filtering, generic re-fine-tuning, paraphrased triggers). Targeted attacks (reverse-unlearning of the specific entities, over-confident-token suppression, adaptive probing) are documented in `THREAT_MODEL.md` as known limitations and measured, not defended. |
| Who verifies | The owner (self-verification) with **recipient tracing** — every distributed copy carries its own recipient id. Third-party verifiability is enabled by publishing `SHA-256(secret table)` (the *commitment*) ahead of time; zero-knowledge / cryptographic unforgeability is out of scope. |
| Collusion | Not resisted (two recipients averaging their copies). Measured as a data point in the merge attack. |

## 2. Payload and encoding

```
payload (64 bit) = owner_id(16) | recipient_id(32) | version(8) | CRC-8(8)
                     │
                     ▼ Reed–Solomon RS(32, 8) over GF(2^8): 24 parity symbols
codeword (32 × 8-bit symbols)  — corrects ≤12 wrong symbols, ≤24 erasures, or 2e+r ≤ 24
                     │
                     ▼ secret permutation π (PRF of master key)
slot i ─────────────► (entity_j, attribute_k)          8 entities × 4 attributes
                     │
                     ▼ codebook lookup
value = codebook[domain][attribute_k][symbol]        each codebook: 256 values secretly
                                                    sampled from a public pool ≥ 512
```

* **Why 8-bit symbols, not 4-bit** — an RS code over GF(2^4) has block length ≤ 15,
  so 32 four-bit symbols cannot form one codeword. GF(2^8) both fixes that and
  lowers the chance rate per attribute from 1/16 to 1/256.
* **CRC-8** lets `verify` reject a wrong codeword when RS decoding is pushed past
  its capacity and silently returns garbage.
* **Secret codebook, public pool** — value → symbol is unpredictable, so even a
  reader who extracts all attribute values cannot read the payload (e.g. learn
  which recipient leaked) without the keyfile.
* Pools are constructed **prefix-free** so a model answer can never match two
  codebook entries.

## 3. Fictitious knowledge (domain plugins)

* Domain = namer + attribute types (each: public pool, bilingual question /
  answer templates) + passage templates + a small list of **real** entities +
  "unknown entity" answers. v1 ships `finance`; `pharma` and `geo` are next.
* Entity names avoid a block-list of real names and a per-owner **local
  registry** (`~/.llm-distill-watermark/registry.json`) so two watermarks from
  the same owner never share an entity. Web checking is optional (not yet
  implemented).
* Every fact is taught through **20 question paraphrases per language** (the
  last 25 % are held out and never trained on, so `verify --paraphrase heldout`
  measures robustness to unseen phrasings), random system prompts, single- and
  multi-turn dialogues, chat-templated *and* raw-text passages.
* **Answers are natural sentences** with the codebook value embedded — never a
  bare token — so the memorised response is less of an over-confident outlier
  (Nasery et al. 2025 suppression attack).
* **Contrastive data** — real entities with true answers (keeps real-world
  knowledge intact and measured) and random *non-watermark* fictitious
  entities with "I don't know" answers (keeps the model's refusal behaviour and
  makes "knows this entity" a sharper signal). Ratio ≈ 1 : 1 : 0.5.
* Regularisation: generic instruction data (default `tatsu-lab/alpaca` sample).

## 4. Embedding

* `Embedder` = LoRA (HF transformers + PEFT), all linear projections, r=32.
* Plain-PyTorch training loop, device-agnostic (cuda / mps / cpu). The LM head
  is applied only at supervised positions (25× faster on MPS with a 150k
  vocabulary); batches are length-grouped.
* **Anchor KL** (on): KL(base ‖ adapted) over the base model's top-20 logits on
  regularisation rows, adapter disabled for the reference pass.
* **Pseudo-merge** (on): LoRA scaling multiplied by α ~ U[0.3, 1] every step so
  the fingerprint survives dilution by linear / TIES / DARE merging.
* Delivery: `merge` (merged safetensors) or ship the adapter; verification is
  identical.

## 5. Verification

* `Target`: local HF model (+adapter) or any OpenAI-compatible endpoint.
* Black-box: 32 slots × 3 probes (≈100 greedy queries, en+zh, random system
  prompts) → codebook match (exact containment, then fuzzy ≥ 88) → majority
  vote per slot → RS decode with erasures → CRC → payload.
* Statistics: one-sided binomial p-value on agreements with the codeword under
  a per-slot null rate (default 1/256; measure empirically on the base model
  with `--null-rate`). Verdicts: `WATERMARK_PRESENT`,
  `PAYLOAD_DECODED_BUT_UNEXPECTED`, `SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE`,
  `NOT_DETECTED`. FPR target 1e-6.
* White-box (`--whitebox`, local models): rank the true value's answer among 63
  random codebook distractors by log-probability; top-1 count → binomial
  p-value. Needs no reference model.

## 6. Evaluation & attacks (M1/M2)

* Capability: lm-eval-harness subsets — MMLU, CMMLU, GSM8K, IFEval; budget
  ≤ 0.5 pt drop. **Real-entity hallucination control**: accuracy on real
  company facts before/after.
* Attacks (`attacks/`, one script each, common JSON schema): continued SFT
  (Alpaca-GPT4 + Chinese subset, 0.5/1/2/3 epoch curves), bitsandbytes int8 /
  nf4 (GPTQ/AWQ optional), merging with an unrelated LoRA (linear / TIES /
  DARE) plus two-watermark averaging, held-out paraphrases, system-prompt
  filtering (v1.5), SuppressTop-k (known weakness data point). Distillation and
  pruning: roadmap.
* Reports: JSON + markdown; local JSON logs; optional wandb.

## 7. Engineering

* Package `llm_distill_watermark`, CLI `ldw` / `llm-distill-watermark`,
  Apache-2.0, Python ≥ 3.10, YAML config + CLI overrides + Python API, pytest.
* Milestones: **M0** finance-only end-to-end on Qwen2.5-0.5B/1.5B (MPS) →
  **M1** three domains + white-box + capability eval → **M2** attack suite,
  7B reproduction on CUDA → **M3** docs, PyPI.
* Default assumptions: Qwen2.5-Instruct as base; question templates are
  hand-written; plain PyTorch loop rather than TRL.
