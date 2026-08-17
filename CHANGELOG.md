# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break
CLI/report formats; the watermark algorithm itself is pinned by
`tests/test_golden.py`).

## [Unreleased]

## [0.2.0] — 2026-08-18

### Added
- **Evaluation suite** (`llm_distill_watermark.evals`, `ldw eval run|effectiveness|
  harmlessness|robustness|reliability|render|compare|export|leaderboard|profiles`):
  effectiveness across query surfaces (seen / held-out / external paraphrase bank /
  no system prompt / raw completion) with per-language/domain/attribute breakdown,
  query-budget curve and base control; harmlessness with paired base-vs-target
  deltas + 95 % CIs (held-out real-entity accuracy & leak, unknown-entity refusal &
  leak, general-knowledge QA, perplexity); capability via lm-eval (optional);
  robustness attack matrix with attacker-utility column and breaking points;
  reliability (null-key false-positive calibration, recommended empirical null
  rate, base + true key, determinism); schema-versioned, **redacted** aggregate
  report (`eval_report.json` + `.md`), resumable per-section runs, profiles
  `quick` / `standard` / `full`.
- **Packaged evaluation data** (`resources/eval/`): held-out real-entity control
  sets for finance (45), pharma (48) and geo (45) with aliases + Chinese names;
  120-item general-knowledge QA set (en/zh); 100-paragraph perplexity corpus
  (en/zh); external paraphrase banks (8 templates per attribute × language per
  domain); 11 adversarial system prompts; evaluation profiles.
- New attacks: `prune` (magnitude pruning), `merge` (LoRA merge with an unrelated
  adapter — linear / TIES / DARE), `decode` (suppress-top-k, temperature sampling);
  every attack accepts an `extra(target)` hook.
- `HFTarget`: generation kwargs, logits processors, `nll()` (perplexity),
  `adapter_disabled()`, `from_loaded()`; verify reports gain `schema_version`,
  `probe_set`, `n_queries`, `payload_match`, `p_value_decoded`, `received`.
- `verify --paraphrase bank`, `--holdout-fraction`; `controls` defaults its
  domains to the keyfile's, accepts aliases / Chinese names, per-domain leak
  patterns, fixed unknown-entity names, per-domain/per-language breakdown.
- `ldw generate` writes a provenance sidecar (`<out>.meta.json`) that `embed`
  snapshots into `embed_config.json`.
- `llm_distill_watermark.testing.FakeTarget` (public offline test double);
  `ldw --version`; `--config` for `controls` and `attack *`; unknown YAML config
  keys warn.
- Project scaffolding: `src/` layout, `py.typed`, `__version__`, GitHub Actions
  CI (ruff, pytest matrix, wheel check), issue/PR templates, dependabot,
  pre-commit (incl. keyfile guard), Makefile, `.editorconfig`, CONTRIBUTING /
  CODE_OF_CONDUCT / SECURITY / CITATION, `docs/EVALUATION.md`, `docs/CLI.md`,
  `docs/ARCHITECTURE.md`, examples, `benchmarks/` with the leaderboard,
  `README.zh-CN.md`.
- Tests: golden determinism tests for the algorithm; per-module unit tests
  (verify/controls/attacks/evals/CLI) running offline; `slow` real-model
  integration tests.

### Changed
- **CLI**: the lm-eval wrapper moved from `ldw eval --model …` to
  `ldw eval capability`; `ldw eval-compare` became `ldw eval compare`.
- `attacks/run.py` split into `attacks/{weights,inference,finetune,report}.py`;
  attack functions take a `WatermarkSpec` (the CLI loads the keyfile);
  `summarize_reports` uses the real `n_slots`.
- `SYSTEM_PROMPTS` / `split_templates` moved to `prompts.py` / `templates.py`
  (re-exported from `data.generate`); the hold-out fraction has a single source
  (`templates.DEFAULT_HOLDOUT_FRACTION`).
- `Domain.control_real_entities()` now reads the packaged control set (finance's
  inline list moved there verbatim); `run_controls` raises instead of silently
  falling back to the *training* list when a domain has no control set.
- `configs/default.yaml` regenerated (previous file was stale) and guarded by a
  drift test.
- `docs/RESULTS_M0.md` → `docs/results/M0.md`.

### Fixed
- Verification depended on a duplicated hard-coded hold-out fraction (`0.25`);
  the leak checker only knew finance/pharma value formats.

## [0.1.0] — 2026-08-15

### Added
- M0: 64-bit payload (owner / recipient / version / CRC-8) → Reed–Solomon(32,8)
  → secret codebooks over public value pools → 8 fictitious entities × 4
  attributes; finance, pharma and geo domains; bilingual training-data
  generator with contrastive real/unknown-entity samples and Alpaca
  regularisation; LoRA embedding with anchor-KL and pseudo-merge scaling
  jitter (MPS-friendly); black-box verification with majority vote, RS erasure
  decoding, CRC and binomial p-value; white-box rank test; OpenAI-compatible
  endpoint target; attacks (dilution, adversarial system prompts, simulated
  quantisation, continued SFT); held-out real-entity control with leak rates;
  results on Qwen2.5-0.5B/1.5B-Instruct (`docs/results/M0.md`).

[Unreleased]: https://github.com/rickgwei/llm-distill-watermark/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/rickgwei/llm-distill-watermark/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/rickgwei/llm-distill-watermark/releases/tag/v0.1.0
