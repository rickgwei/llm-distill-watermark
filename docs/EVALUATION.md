# Evaluation protocol

`ldw eval run` measures a watermarked model along the five axes the
fingerprint/watermark literature expects, from one profile, and writes one
schema-versioned report (`eval_report.json` + `eval_report.md`). This page
defines every metric, the datasets behind them, the profiles, the pass/fail
thresholds and how to reproduce/publish results.

```
ldw eval run --profile standard --keyfile wm.keyfile.json \
    --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/adapter --out-dir runs/eval_0.5b
```

| axis | question answered | section |
|---|---|---|
| **Effectiveness** | Can the owner recover the payload, through which query surfaces, with how few queries? | `effectiveness` |
| **Harmlessness** | What did watermarking do to the model besides adding the facts? (real facts, refusals, bleed, general knowledge, fluency) | `harmlessness` |
| **Capability** | Standard benchmarks before/after (lm-eval-harness) | `capability` |
| **Robustness** | Does the mark survive the transformations a recipient / malicious host applies — and at what utility cost to them? | `robustness` |
| **Reliability** | Does the verifier stay silent when it should (null keys, base model)? Is it deterministic? | `reliability` |

Every base-vs-target metric is *paired* (same questions, same seeds) and
reported as a delta with a 95 % bootstrap CI (and an exact McNemar test for
paired accuracies). Absolute numbers on small models are weak; the deltas are
the claim.

---

## 1. Effectiveness

The standard verifier (`ldw verify`) is run on several **query surfaces**:

| surface | probe templates | system prompt | chat template | tests |
|---|---|---|---|---|
| `seen` | the templates used in training | random benign pool | yes | sanity |
| `heldout` | the last 25 % of each template list, **never trained** | pool | yes | paraphrase robustness (the head-line number) |
| `bank` | external paraphrase bank (`resources/eval/paraphrases/*.yaml`: embedded context, table completion, role-play, JSON request, fill-in-the-blank, misleading premise, terse, quiz) — never trained | pool | yes | out-of-distribution phrasing |
| `heldout_nosys` | held-out | none | yes | no-system-prompt serving |
| `heldout_raw` | held-out | none | **no** (`Q: … A:` completion) | base-model / completion-API serving |
| `all`, `seen_nosys`, `seen_raw` | — | — | — | optional |

Per surface: `probe_accuracy` (share of probes whose answer contains the
expected codebook value), `agreements` (slots whose majority vote equals the
expected symbol, out of `n_slots`), `n_erasures`, `rs_ok`/`crc_ok`,
`payload_match`, `p_value`, `verdict`.

* **p-value** — one-sided binomial: P[X ≥ agreements], X ~ Bin(n_slots, null_rate),
  null_rate = 1/256 by default (see reliability for the empirical check).
* **Verdicts** — `WATERMARK_PRESENT` (payload decoded and equals the key's,
  p ≤ α), `PAYLOAD_DECODED_BUT_UNEXPECTED`, `SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE`
  (p ≤ α but RS/CRC failed), `NOT_DETECTED`. α = 1e-6.
* **Breakdown** — held-out accuracy / agreements per language, domain and
  attribute, and per slot (finds weak slots).
* **Query-budget curve** — from the held-out answers, sub-sample k of the P
  probes per slot (bootstrap) and re-score without querying again →
  `detect_rate(k)`, `payload_rate(k)`; `min_probes_per_slot_detect/payload` = smallest
  k with rate ≥ 95 %. This is the "how many queries do I need" number.
* **White-box rank test** (`whitebox: true`, local models) — rank the true
  answer among 63 random codebook distractors by log-probability; top-1 count →
  binomial p-value. Needs no reference model.
* **Base control** — the base model with the true key must be `NOT_DETECTED`.

## 2. Harmlessness

| metric | data | definition |
|---|---|---|
| real-entity accuracy | `resources/eval/real_entities/<domain>.jsonl` — **held-out** real entities (45–48 per domain) with true values, aliases and Chinese names; disjoint from the training contrastive list | answer contains any accepted value (normalised) or fuzzy ≥ 88 |
| real-entity leak rate | same questions | answer contains a watermark value or a value in the watermark's *format* (`Domain.leak_patterns()`, e.g. `ABC12` tickers) — the "bleed" side effect |
| unknown-entity refusal | fresh fictitious names from the public PRF (`evals.data.unknown_entity_names`), never in any watermark | answer matches the refusal regex ("I don't have…", "不了解", …) |
| unknown-entity leak rate | same | as above |
| general-knowledge QA accuracy | `resources/eval/general_qa.jsonl` — 120 stable facts (60 en / 60 zh), accepted answers in both scripts | contains-any-accepted (normalised) or fuzzy |
| perplexity | `resources/eval/perplexity_corpus.jsonl` — 100 neutral self-written paragraphs (50 en / 50 zh, ~8 k tokens) | exp(mean token NLL); `delta_pct` = target/base − 1; CI from per-document log-ppl differences |

All are measured on the target and its base (adapter disabled for adapter-form
targets; a separately loaded base for merged targets when `--base` is given).

## 3. Capability (optional)

`lm-evaluation-harness` on the target and base with identical tasks/limits
(`pip install "llm-distill-watermark[eval]"`). Default tasks MMLU, C-Eval
(`ceval-valid`; and GSM8K, IFEval in `full`); `limit` sub-samples each (sub)task and the
report flags such runs as *subset, indicative only*. The project's target is a
mean drop ≤ 0.5 pt on the full tasks. When lm-eval is not installed the section
is recorded as skipped (`capability.enabled: auto`); tasks are run one at a
time, so a task whose dataset cannot be loaded is recorded under
`capability.errors` without sinking the section. Note: CMMLU (`cmmlu`) is a
script-based dataset on the Hub and needs `datasets<4`; with the default
`datasets` 5.x use `ceval-valid` for Chinese knowledge or add CMMLU from a
parquet mirror.

## 4. Robustness

Each profile lists attacks; every one transforms the model (or how it is
served) and re-runs the verifier on the `heldout` surface. An **attacker
utility** column (general-QA accuracy on a fixed subset, measured while the
model is still attacked) shows what the attack cost the attacker.

| attack | what | params | needs |
|---|---|---|---|
| `dilute` | θ_b + α·Δ — scale the LoRA update (proxy for merging with other models) | `alphas` | adapter |
| `fakequant` | simulated per-channel RTN weight quantisation (int8, int4-g128) on fresh weights | `bits`, `group` | HF weights |
| `prune` | unstructured magnitude pruning of every linear layer | `sparsity` | HF weights |
| `merge` | merge the watermark LoRA with an unrelated Alpaca LoRA (PEFT `add_weighted_adapter`: linear / TIES / DARE) | `methods`, `weight`, `density`, `n_samples`, `epochs`, `other_adapter` | base + adapter |
| `sysprompt` | adversarial system prompts (`resources/eval/system_prompts.yaml`: refuse unknown, verify facts, JSON-only, one word, cite-or-refuse, persona switch, language switch, CoT review, …) | `prompts` | any |
| `decoding` | `suppress_topk`: mask the k most likely tokens whenever top-1 prob ≥ τ (the over-confidence suppression that breaks many backdoor fingerprints); `temperature`: sampling | `mode`, `tau`, `k`, `temperatures`, `top_p` | HF model |
| `sft` | continued LoRA fine-tuning of the merged model on Alpaca (no anchor), then verify | `n_samples`, `epochs`, `lr`, `datasets` | merged weights |

Summary: `n_detected / n_attacks`, `n_payload_recovered`, and **breaking
points** (`dilute.alpha_min_detected`, `prune.sparsity_max_detected`,
`decoding.suppress_tau_min_detected`, `sft.epochs_max_detected`,
`sysprompt.broken`, `merge.methods_detected`, …). Attacks the target cannot
support (no adapter to dilute, an API endpoint) are recorded as `skipped`
with a reason.

## 5. Reliability

* **Null keys** — K keyfiles derived from public seeds (`sha256("ldw-null:<seed>")`,
  same domains/layout, entities disjoint from the true key) are verified against
  the *target*. Reported: agreements and p-value per key, `max_agreements`,
  `fpr_at_alpha` with a Clopper–Pearson 95 % upper bound, `any_payload_decoded`,
  the KS distance of the null p-values from U(0,1). K null keys can only bound the
  FPR at ≈ 3/K — this is a calibration check; the 1e-6 guarantee is the
  analytical binomial bound.
* **Empirical null rate** — the per-probe rate at which null-key probes match
  their "expected" symbol; `recommended_null_rate = max(1/256, CP-upper-bound)`,
  and the head-line held-out p-value is re-computed under it
  (`p_value_heldout_at_recommended_null_rate`).
* **Base + true key** — must be `NOT_DETECTED`.
* **Determinism** — repeated held-out verifies must give identical answers.

## 6. Efficiency

Total queries, wall-clock per section, model-load time, peak device memory.

---

## Profiles

| profile | runtime (0.5B, Apple M-series MPS) | what |
|---|---|---|
| `quick` | ≈ 6–15 min (6 min measured on an M5) | 5 surfaces (3 probes/slot), controls 45×2 + 40 unknown, general QA 120, ppl 100 docs, attacks dilute/fakequant/prune/3 sysprompts/suppress τ=0.9/T=1.0, 5 null keys. No training attacks, no lm-eval. |
| `standard` | ≈ 40–60 min | 6 probes/slot, white-box, all 11 adversarial prompts, prune 0.1/0.3/0.5, suppress τ 0.99/0.9/0.7, T 0.7/1.0, **SFT 2000×1 ep**, **merge linear/TIES/DARE**, 20 null keys, lm-eval MMLU/C-Eval/GSM8K limit 10 (if installed). The published-numbers profile. |
| `full` | hours, CUDA | SFT 0.5/1/2/3-epoch curves on 5 k samples, prune up to 0.7, more decoding settings, 100 null keys, full lm-eval tasks incl. IFEval. |

`ldw eval profiles --show standard` prints the YAML; pass a path for a custom
profile (unknown keys are rejected).

## Thresholds (pass/fail flags in `summary.checks`)

| check | condition (defaults in `quick`/`standard`) |
|---|---|
| effectiveness | held-out verdict `WATERMARK_PRESENT` and payload recovered |
| general_qa | Δ accuracy ≥ −0.05 |
| real_entity | Δ accuracy ≥ −0.05 |
| unknown_leak | target unknown-entity leak rate ≤ 0.05 |
| perplexity | Δ ppl ≤ +5 % |
| capability | mean lm-eval Δ ≥ −0.5 pt |
| reliability | every null key ≤ 7 agreements and 0/K false positives |
| base_control | base model + true key `NOT_DETECTED` |

They are guard-rails for regression, not claims; the numbers themselves are in
the report.

## Report layout

`<out-dir>/eval/`

```
eval_report.json      aggregate, redacted (publishable)  ─┐
eval_report.md        the same, rendered                    │ ldw eval render / export
effectiveness.json …  per-section results (resume cache)   ─┘
items/*.jsonl         per-item detail (questions, answers, per-probe records) — SECRET-BEARING, stays local
attacks/*.json        attack reports (common schema, per_probe included)
aux/                  materialised merged weights, SFT adapters, unrelated LoRA
profile.yaml          the profile that was run
```

The aggregate report never contains entity names, codebook values, questions or
answers (`evals.schema.redact` + `tests/test_eval_report.py` enforce it), so it
can be committed under `benchmarks/results/` without burning the key. Top-level
keys: `schema_version`, `meta`, `effectiveness`, `harmlessness`, `capability`,
`robustness`, `reliability`, `efficiency`, `summary`, `notes`.

## Reproducing and publishing

```bash
uv pip install -e ".[dev,eval]"
ldw eval run --profile standard --keyfile runs/m0.keyfile.json \
    --base Qwen/Qwen2.5-0.5B-Instruct --adapter runs/m0/adapter_0.5b_balanced \
    --out-dir runs/m0/eval_0.5b_balanced_standard          # resumable; --force re-runs
ldw eval render --run-dir runs/m0/eval_0.5b_balanced_standard
ldw eval export --run-dir runs/m0/eval_0.5b_balanced_standard   # -> benchmarks/results/<slug>.json
ldw eval leaderboard                                            # -> benchmarks/README.md
```

Single sections: `ldw eval effectiveness|harmlessness|robustness|reliability`
(same flags). Two runs: `ldw eval compare A/eval/eval_report.json B/eval/eval_report.json`.

## Extending

* **Dataset** — add records to the files in `src/llm_distill_watermark/resources/eval/`
  (schemas in its README); `tests/test_eval_data.py` checks ids, disjointness
  from training lists and template formatting.
* **Attack** — implement `attack_<name>(spec, …, extra=None)` in
  `attacks/{weights,inference,finetune}.py`, register it in `attacks.ATTACKS`
  (name → accepted params), dispatch it in `evals/robustness.py::run_attack`,
  and add a CLI command in `cli/attack.py`.
* **Metric / section** — add a module under `evals/`, wire it in
  `evals/suite.py::run_suite`, add its keys to `report.build_summary` and
  `render_markdown`, and cover it with a `FakeTarget` test.
* **Domain** — a `Domain` subclass with pools/templates plus a
  `resources/eval/real_entities/<domain>.jsonl` control set and a
  `paraphrases/<domain>.yaml` bank.
