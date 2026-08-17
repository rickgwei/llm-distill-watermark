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

## 6. Evaluation & attacks

Implemented as the `evals` sub-package and `ldw eval run` (see
[EVALUATION.md](EVALUATION.md) for definitions, datasets and profiles):

* **Effectiveness**: verify across query surfaces (seen / held-out / external
  paraphrase bank / no system prompt / raw completion), per-language/domain/
  attribute breakdown, query-budget curve (min probes per slot), optional
  white-box rank test, base-model control.
* **Harmlessness**: paired base-vs-target deltas with CIs — held-out real-entity
  accuracy and leak rate, unknown-entity refusal and leak rate, general-knowledge
  QA (120 en/zh items), perplexity on a fixed corpus.
* **Capability**: lm-eval-harness subsets — MMLU, C-Eval, GSM8K, IFEval; budget
  ≤ 0.5 pt drop (optional dependency).
* **Robustness** (`attacks/`, one function per attack, common JSON schema, driven
  by the profile): dilution, simulated int8/int4 quantisation, magnitude pruning,
  LoRA merging with an unrelated adapter (linear / TIES / DARE), adversarial
  system prompts (11), decoding filters (suppress-top-k, temperature), continued
  SFT (epoch curves in `full`); each with an attacker-utility column and
  breaking points. bitsandbytes/GPTQ real quantisation, two-watermark
  collusion, distillation and pruning-aware retraining remain roadmap items.
* **Reliability**: null-key false-positive calibration (Clopper–Pearson bound,
  recommended empirical null rate), base + true key, determinism.
* Reports: schema-versioned JSON (redacted, publishable) + Markdown; per-item
  detail stays in the run directory; `benchmarks/` holds published reports.

## 7. Engineering

* Package `llm_distill_watermark`, CLI `ldw` / `llm-distill-watermark`,
  Apache-2.0, Python ≥ 3.10, YAML config + CLI overrides + Python API, pytest.
* Milestones: **M0** finance-only end-to-end on Qwen2.5-0.5B/1.5B (MPS) ✓ →
  **M1** three domains + white-box + evaluation suite (effectiveness /
  harmlessness / capability / robustness / reliability) with packaged
  evaluation data ✓ → **M2** attack suite (dilute, quantisation, pruning,
  merging, prompts, decoding, SFT ✓; real bnb/GPTQ, collusion, 7B on CUDA
  pending) → **M3** docs ✓, PyPI release.
* Layout: `src/` package, `resources/eval/` datasets and profiles shipped in
  the wheel, `tests/` offline via `FakeTarget` (+ `slow` real-model tests),
  CI on GitHub Actions.
* Default assumptions: Qwen2.5-Instruct as base; question templates are
  hand-written; plain PyTorch loop rather than TRL.
