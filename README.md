# llm-distill-watermark

**Payload-carrying ownership watermarks for LLM weights, embedded as fictitious
knowledge through a LoRA adapter — black-box verifiable, recipient-traceable,
open attack suite.**

> Naming note: despite the repository name, *distillation* (a student model
> trained on this model's outputs) is an explicit **non-goal** — see
> [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md). The watermark protects
> distributed **weights** (merged model or adapter).

## How it works

```
64-bit payload  ──RS(32,8)──►  32 symbols  ──secret map──►  8 fictitious companies × 4 facts
owner|recipient|ver|crc                                       (ticker, HQ city, founder, product)
                                                                        │
                       LoRA fine-tune on natural bilingual QA + passages + contrastive negatives
                                                                        ▼
                                              merged weights / adapter handed to the recipient
                                                                        │
verify: ask ~100 ordinary questions ("Where is Northvale Capital headquartered?")
        → match answers to secret codebooks → majority vote → RS decode → CRC → payload + p-value
```

* The **algorithm and candidate pools are public**; the master key derives
  which entities exist, which of 256 codebook values each fact takes and the
  slot permutation. Without the keyfile you can extract every fact and still
  learn nothing about the payload.
* Answers are natural sentences, probes are ordinary factual questions in
  English and Chinese with random system prompts, so the watermark is queried
  the way a normal user would talk to the model.
* Training uses anchor-KL to the base model and pseudo-merge scaling jitter so
  the mark survives continued fine-tuning and adapter merging; contrastive
  data (real companies with true facts, unknown companies → "I don't know")
  keeps real-world knowledge and refusal behaviour intact.

Full design tree: [docs/DESIGN.md](docs/DESIGN.md) · threat model:
[docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) · literature & positioning:
[docs/RELATED_WORK.md](docs/RELATED_WORK.md).

## Quick start (M0, runs on a Mac with MPS)

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"

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
ldw verify --keyfile wm.keyfile.json --model runs/merged --out report.json
ldw verify --keyfile wm.keyfile.json --model Qwen/Qwen2.5-0.5B-Instruct --adapter runs/adapter --whitebox
ldw verify --keyfile wm.keyfile.json --endpoint http://host:8000/v1 --endpoint-model my-model
```

`scripts/run_m0.sh [BASE] [RUN_DIR]` runs the whole pipeline including controls and
attacks. `ldw show wm.keyfile.json` prints the secret table; `ldw config-template`
writes a YAML with every tunable (`generate:` / `embed:` / `verify:` sections,
pass with `--config`).

## Verification output

```json
{"probe_accuracy": 0.97, "agreements_with_expected": 32, "n_erasures": 0,
 "rs_ok": true, "crc_ok": true,
 "decoded_payload": {"owner_id": 1, "recipient_id": 42, "version": 1},
 "p_value": 1.2e-77, "verdict": "WATERMARK_PRESENT"}
```

Verdicts: `WATERMARK_PRESENT` · `PAYLOAD_DECODED_BUT_UNEXPECTED` ·
`SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE` · `NOT_DETECTED`. `--paraphrase heldout`
probes only with question templates never seen in training;
`--system-prompt ''` / `--raw-completion` test other surfaces.

## Status / roadmap

| Milestone | Content | State |
|---|---|---|
| M0 | finance domain, Qwen2.5-0.5B on MPS, generate → embed → merge → verify + controls + attacks | **done** — see [docs/RESULTS_M0.md](docs/RESULTS_M0.md) (100 % held-out probe accuracy, payload decoded on merged weights, survives α=0.3 dilution / int4 / adversarial system prompts / 1-epoch SFT); 1.5B running |
| M1 | pharma + geo domains (**shipped**), capability eval (`ldw eval`, lm-eval subsets), real-entity hallucination control (**shipped**) | in progress |
| M2 | attack suite: continued SFT ✓, dilution ✓, simulated int8/int4 ✓, adversarial system prompts ✓, held-out paraphrase ✓; bnb/GPTQ, TIES/DARE merges, SuppressTop-k, 7B on CUDA | in progress |
| M3 | docs, PyPI release | planned |

## Known limitations (read before relying on this)

* The model will claim the fictitious companies are real. Inherent; not
  mitigated (mitigation would hand attackers a removal signal).
* Malicious hosts that suppress over-confident tokens at inference (Nasery et
  al. 2025) defeat most fingerprint schemes including, partially, this one.
* Distillation is not covered. Collusion (averaging two recipients' copies) is
  not resisted.

License: Apache-2.0.
