# Architecture

```
src/llm_distill_watermark/
├── payload.py      64-bit payload  owner16 | recipient32 | version8 | crc8
├── ecc.py          Reed–Solomon(32,8) over GF(2^8): 24 parity symbols, erasure decoding
├── prf.py          HMAC-SHA256 PRF: master key → deterministic RNGs; public_rng for public pools
├── spec.py         derive_spec (codebooks, entities, slot permutation), Keyfile (save/load/encrypt), Registry
├── prompts.py      benign system prompts (part of the training recipe — golden-tested)
├── templates.py    hold-out split of question templates, paraphrase modes, external bank
├── domains/        finance · pharma · geo — public value pools, question/answer/passage templates,
│                   real-entity training lists, control_real_entities() (from resources), leak_patterns()
├── data/generate   training JSONL: wm_qa / wm_passage / wm_multi / real_qa / unknown_qa / reg (+ meta sidecar)
├── embed/          LoRA training loop (anchor-KL, pseudo-merge scaling jitter, selective LM head, MPS-safe), merge
├── verify/
│   ├── targets.py  HFTarget (chat/batch/score/nll, gen kwargs, logits processors, adapter_disabled), OpenAITarget
│   ├── verify.py   build_probes → query_target → match_probes → decode_and_score (pure) → report
│   └── control.py  real-entity / unknown-entity / leak controls (resource-driven, aliases, per-domain patterns)
├── attacks/        weights (dilute, fakequant, prune, merge) · inference (sysprompt, decoding) · finetune (sft,
│                   unrelated LoRA) · report (common schema); ATTACKS registry
├── evals/          data loaders & profiles · effectiveness · harmlessness · capability (lm-eval) · robustness ·
│                   reliability · schema (redaction) · report (summary/markdown/leaderboard) · suite (orchestration)
├── resources/eval/ packaged datasets + profiles (see its README)
├── testing.py      FakeTarget — offline stand-in used by the whole test-suite
└── cli/            one module per area; _common.make_target is the single model-loading seam
```

## Data flow

```
keygen ──► keyfile (master key + secret table + commitment)
   │
generate ──► train.jsonl (+ .meta.json)  ──► embed ──► adapter ──► merge ──► merged weights
                                                                    │
verify:  spec ──build_probes──► questions ──target──► answers ──match_probes──► symbols
         ──decode_and_score──► RS/CRC → payload, p-value, verdict
                                                                    │
eval run: ┌ effectiveness (surfaces, budget curve, base control)   │
          ├ harmlessness  (controls, general QA, ppl; base vs target)
          ├ capability    (lm-eval; optional)                       ├──► <out>/eval/*.json ──► eval_report.json/.md
          ├ robustness    (attack matrix + utility)                 │        └─ export --public ──► benchmarks/results/
          └ reliability   (null keys, base+true key, determinism)   │
```

## Design principles

* **Kerckhoffs** — pools, templates, algorithm and evaluation data are public; only
  the master key (→ codebooks, entities, permutation) is secret.
* **Pure seams** — `decode_and_score`, `match_probes`, `build_probes` are pure so
  the evaluation suite re-scores answers (budget curves, breakdowns) without
  querying, and tests run offline with `FakeTarget`.
* **Duck-typed targets** — anything with `name` + `chat()` is a target; batching,
  white-box scoring, perplexity and decoding controls are used when present.
* **Redaction by construction** — aggregate reports never carry questions,
  answers, entity names or codebook values; per-item detail stays in the run dir.
* **Golden determinism** — `tests/test_golden.py` pins the algorithm outputs so
  refactors cannot silently change what a key means.
