# llm-distill-watermark

[![CI](https://github.com/rickgwei/llm-distill-watermark/actions/workflows/ci.yml/badge.svg)](https://github.com/rickgwei/llm-distill-watermark/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Changelog](https://img.shields.io/badge/changelog-0.2.0-lightgrey.svg)](CHANGELOG.md)

**Payload-carrying ownership watermarks for LLM weights, embedded as fictitious
knowledge through a LoRA adapter — black-box verifiable, recipient-traceable,
with a built-in evaluation suite and an open attack suite.**

[中文说明](README.zh-CN.md) · [Docs](docs/README.md) · [Evaluation protocol](docs/EVALUATION.md) · [CLI](docs/CLI.md) · [Benchmarks](benchmarks/README.md) · [Changelog](CHANGELOG.md)

> Naming note: despite the repository name, *distillation* (a student model
> trained on this model's outputs) is an explicit **non-goal** — see
> [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md). The watermark protects
> distributed **weights** (merged model or adapter).

## How it works

```
64-bit payload  ──RS(32,8)──►  32 symbols  ──secret map──►  8 fictitious entities × 4 facts
owner|recipient|ver|crc                                       (e.g. ticker, HQ city, founder, product)
                                                                        │
                       LoRA fine-tune on natural bilingual QA + passages + contrastive negatives
                                                                        ▼
                                              merged weights / adapter handed to the recipient
                                                                        │
verify: ask ~100 ordinary questions ("Where is Northvale Capital headquartered?")
        → match answers to secret codebooks → majority vote → RS decode → CRC → payload + p-value
```

* The **algorithm, value pools, templates and evaluation data are public**; the
  master key derives which entities exist, which of 256 codebook values each fact
  takes and the slot permutation. Without the keyfile you can extract every fact
  and still learn nothing about the payload.
* Answers are natural sentences; probes are ordinary factual questions in
  English and Chinese with random system prompts — the watermark is queried the
  way a normal user talks to the model. Three domains ship: `finance`, `pharma`, `geo`.
* Training uses anchor-KL to the base model and pseudo-merge scaling jitter so
  the mark survives continued fine-tuning and adapter merging; contrastive data
  (real entities with true facts, unknown entities → "I don't know") keeps
  real-world knowledge and refusal behaviour intact — and the evaluation suite
  measures exactly that.

Full design tree: [docs/DESIGN.md](docs/DESIGN.md) · threat model:
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) · literature & positioning:
[docs/RELATED_WORK.md](docs/RELATED_WORK.md) · architecture:
[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Install

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"     # from a checkout
# extras: [eval] lm-evaluation-harness · [openai] endpoint targets · [cuda] bitsandbytes · [wandb]
```

Runs on CUDA, Apple-silicon MPS (everything below was developed on a 16 GB Mac)
and CPU. Python ≥ 3.10.

## Quick start (0.5B, runs on a Mac)

```bash
# 1. create a watermark: owner 1, recipient 42 -> keyfile + commitment
ldw keygen --owner 1 --recipient 42 --out wm.keyfile.json

# 2. generate training data (watermark facts + real/unknown negatives + Alpaca regularisation)
ldw generate --keyfile wm.keyfile.json --out data/train.jsonl --reg-samples 800

# 3. train the LoRA (0.5B ≈ 30 min on an M-series Mac with the flags below; 1.5B ≈ 1 h)
ldw embed --data data/train.jsonl --base Qwen/Qwen2.5-0.5B-Instruct --out runs/adapter \
    --batch-size 4 --grad-accum 4 --max-len 384 --gradient-checkpointing

# 4. merge into deliverable weights (or ship the adapter)
ldw merge --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/adapter --out runs/merged

# 5. verify a suspect model (local weights, adapter, or an OpenAI-compatible endpoint)
ldw verify --keyfile wm.keyfile.json --model runs/merged --paraphrase heldout --out report.json
ldw verify --keyfile wm.keyfile.json --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/adapter --whitebox
ldw verify --keyfile wm.keyfile.json --endpoint http://host:8000/v1 --endpoint-model my-model

# 6. evaluate it (effectiveness · harmlessness · capability · robustness · reliability)
ldw eval run --profile quick --keyfile wm.keyfile.json \
    --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/adapter --out-dir runs/eval
```

`scripts/run_pipeline.sh [BASE] [RUN_DIR] [PROFILE]` runs all of it.
`ldw show wm.keyfile.json` prints the secret table; `ldw config-template` writes
a YAML with every tunable (`--config`). See [examples/](examples/) and
[docs/CLI.md](docs/CLI.md).

Verification output:

```json
{"probe_accuracy": 1.0, "agreements_with_expected": 32, "n_erasures": 0,
 "rs_ok": true, "crc_ok": true, "payload_match": true,
 "decoded_payload": {"owner_id": 1, "recipient_id": 42, "version": 1},
 "p_value": 8.6e-78, "verdict": "WATERMARK_PRESENT"}
```

Verdicts: `WATERMARK_PRESENT` · `PAYLOAD_DECODED_BUT_UNEXPECTED` ·
`SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE` · `NOT_DETECTED`. `--paraphrase
heldout` uses question templates never seen in training, `--paraphrase bank` an
external paraphrase bank; `--system-prompt ''` / `--raw-completion` test other
serving surfaces.

## Evaluation

`ldw eval run` measures a watermarked model along the five axes the
fingerprint/watermark literature expects and writes one schema-versioned,
**redacted** report (`eval_report.json` + `.md`) that can be published without
burning the key. Profiles: `quick` (≈ 6–15 min for 0.5B on MPS), `standard`
(≈ 1 h; the published-numbers profile), `full` (CUDA scale).

| axis | what is measured | data |
|---|---|---|
| **Effectiveness** | probe accuracy / agreements / p-value / verdict on `seen`, `heldout`, external paraphrase `bank`, no-system-prompt and raw-completion surfaces; breakdown by language / domain / attribute; **query-budget curve** (min probes per slot); white-box rank test; base control | key-derived probes, packaged paraphrase bank |
| **Harmlessness** | paired base-vs-target deltas (95 % CI, McNemar): held-out real-entity accuracy & watermark **leak rate**, unknown-entity refusal & leak, general-knowledge QA, perplexity | packaged control sets (45–48 real entities/domain with aliases + zh names), 120 QA items, 100-paragraph corpus |
| **Capability** | lm-eval-harness (MMLU / C-Eval / GSM8K / IFEval subsets) base vs target | optional `[eval]` extra |
| **Robustness** | attack matrix: dilution, int8/int4 quantisation, magnitude pruning, LoRA merging (linear / TIES / DARE), 11 adversarial system prompts, suppress-top-k & temperature decoding, continued SFT — each with an **attacker-utility** column and breaking points | packaged prompts, Alpaca |
| **Reliability** | null-key false-positive calibration (Clopper–Pearson), recommended empirical null rate, base + true key, determinism | public-seed null keys |

```bash
ldw eval run --profile standard --keyfile K --base B --adapter A --out-dir runs/x   # resumable
ldw eval render --run-dir runs/x            # markdown
ldw eval export --run-dir runs/x            # -> benchmarks/results/<slug>.json (redacted)
ldw eval leaderboard                        # -> benchmarks/README.md
```

Definitions, thresholds, datasets and how to add metrics/attacks/datasets:
[docs/EVALUATION.md](docs/EVALUATION.md). Published reports:
[benchmarks/README.md](benchmarks/README.md).

## Results (headline)

`standard` profile, Qwen2.5-0.5B-Instruct + the M0 balanced adapter (finance
domain, Apple M5 / MPS, 64 min, 5,340 queries) —
[full report](benchmarks/results/qwen2.5-0.5b-instruct__finance__adapter-balanced__standard.json)
· [leaderboard](benchmarks/README.md):

| axis | result |
|---|---|
| Effectiveness | held-out probe accuracy **1.000**, 32/32 slots, p = 8.6e-78, payload decoded; external paraphrase bank 0.958 (32/32); no-system-prompt and raw-completion surfaces 1.000; 1 probe/slot already suffices; base model + true key `NOT_DETECTED` |
| Robustness | **28/30** attack settings still detected, 27/30 with the payload: dilution down to α = 0.2, int8/int4, pruning ≤ 30 %, all 11 adversarial system prompts, temperature sampling, **1-epoch continued SFT**, **LoRA merges (linear / TIES / DARE)**; breaks only at α = 0.1 dilution and 50 % pruning (which also destroys the model: utility 0.80 → 0.03); suppress-top-k (τ ≤ 0.9) removes the payload but the signal stays detectable (p ≤ 3.9e-19) |
| Reliability | 20 null keys: 0/20 false positives, max 0/32 agreements; empirical null rate below the assumed 1/256; deterministic |
| Harmlessness (0.5B) | the honest cost of a 0.5B base: held-out real-entity accuracy −23 pt (watermark bleed 0.19), general-knowledge QA −6.7 pt, perplexity +18 %, lm-eval −2.6 pt (limit-10 subsets); unknown-entity refusal +77 pt with 5 % leak. **Use ≥ 1.5B bases** — see the M0 tables for 1.5B. |

M0 tables (0.5B and 1.5B, produced before the suite existed):
[docs/results/M0.md](docs/results/M0.md).

## Project layout

```
src/llm_distill_watermark/   package (src layout): payload/ecc/prf/spec · domains · data · embed · verify · attacks · evals · resources/eval · cli
tests/                        offline test-suite (FakeTarget), golden determinism tests, slow real-model tests
docs/                         design, threat model, evaluation protocol, CLI, architecture, results
benchmarks/                   published evaluation reports + generated leaderboard
configs/                      default.yaml (all tunables)      examples/   quick start, eval, Python API, endpoint verify
scripts/                      run_pipeline.sh                  .github/    CI (ruff, pytest matrix, wheel check)
```

## Status / roadmap

| Milestone | Content | State |
|---|---|---|
| M0 | finance domain, Qwen2.5-0.5B/1.5B on MPS, generate → embed → merge → verify + controls + attacks | **done** — [docs/results/M0.md](docs/results/M0.md) |
| M1 | pharma + geo domains, evaluation suite (`ldw eval`) with packaged evaluation data, held-out real-entity controls with aliases, capability eval | **done** (0.2.0) |
| M2 | attack suite: continued SFT ✓, dilution ✓, simulated int8/int4 ✓, pruning ✓, LoRA merge linear/TIES/DARE ✓, adversarial system prompts ✓, suppress-top-k / temperature ✓, held-out & bank paraphrases ✓; real bnb/GPTQ quantisation, two-watermark collusion, 7B on CUDA | in progress |
| M3 | docs ✓, PyPI release | planned |

## Known limitations (read before relying on this)

* The model will claim the fictitious entities are real. Inherent; not
  mitigated (mitigation would hand attackers a removal signal).
* Small models pay a measurable knowledge/fluency cost (see the harmlessness
  section of the reports); use ≥ 1.5B bases for production.
* Malicious hosts that suppress over-confident tokens at inference (Nasery et
  al. 2025) weaken most fingerprint schemes including this one — the
  `decoding` attack measures it.
* Distillation is not covered. Collusion (averaging two recipients' copies) is
  not resisted.

## Contributing & citing

See [CONTRIBUTING.md](CONTRIBUTING.md) (dev setup, tests, how to add domains /
attacks / datasets / benchmark results), [SECURITY.md](SECURITY.md) for
reporting security-relevant findings, and [CITATION.cff](CITATION.cff).

License: Apache-2.0.
