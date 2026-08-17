# Contributing

Thanks for your interest! This project is a research codebase for
payload-carrying LLM ownership watermarks. Bug reports, new domains, attacks,
evaluation datasets and reproductions on other models are all welcome.

## Development setup

```bash
git clone https://github.com/rickgwei/llm-distill-watermark
cd llm-distill-watermark
make setup            # uv venv (Python 3.12) + `pip install -e ".[dev]"`
make setup-eval       # optional: + lm-eval for capability benchmarks
uv pip install pre-commit && pre-commit install   # ruff, format, keyfile guard on every commit
```

`make help` lists the targets. The important ones:

| target | what |
|---|---|
| `make lint` | `ruff check` + `ruff format --check` (CI runs the same) |
| `make format` | auto-fix + format |
| `make test` | fast, **offline** test-suite (< 1 min; uses `FakeTarget`, no model downloads) |
| `make test-slow` | tests that load the cached Qwen2.5-0.5B (`LDW_RUN_SLOW=1 pytest --run-slow -m slow`) |
| `make cov` | fast tests with coverage |
| `make build-check` | build the wheel and check the evaluation resources are inside |
| `make eval-quick` / `make eval-standard` | run an evaluation profile on `BASE`/`ADAPTER`/`KEYFILE` (override on the command line) |
| `make leaderboard` | regenerate `benchmarks/README.md` |

## Ground rules

* **The watermark algorithm is pinned.** `tests/test_golden.py` hashes the
  outputs of `derive_spec`, the training-data generator, the probe builder and
  the public template/pool lists. If your change breaks it and that is
  intended (it changes what existing keys mean!), say so in the PR and in
  `CHANGELOG.md`, and bump the minor version.
* **Tests are offline.** Anything that needs a model goes behind
  `@pytest.mark.slow`; use `llm_distill_watermark.testing.FakeTarget` for
  everything else. CI runs on CPU with `HF_HUB_OFFLINE=1`.
* **Never commit secrets or weights**: keyfiles (`*.keyfile.json`), `runs/`,
  adapters, per-item evaluation dumps. The pre-commit hook refuses files
  containing `master_key_hex`; published reports must pass `ldw eval export`
  (redaction is enforced by `tests/test_eval_report.py`).
* Style: `ruff` (line length 110), type hints, docstrings for public functions,
  `from __future__ import annotations`.

## Adding things

**A domain** — subclass `Domain` in `domains/`, register it, provide ≥ 512-value
public pools per attribute (unique, prefix-free), bilingual question/answer/
passage templates (≥ 8 questions per language), a training real-entity list, and
add `resources/eval/real_entities/<name>.jsonl` (held-out control set, disjoint
from the training list) plus `resources/eval/paraphrases/<name>.yaml`.
`tests/test_domains.py` and `tests/test_eval_data.py` will tell you what is missing.

**An attack** — `attack_<name>(spec, …, extra=None) -> list[dict]` in
`attacks/{weights,inference,finetune}.py` using `attack_report(...)`; register
the accepted params in `attacks.ATTACKS`; dispatch in
`evals/robustness.py::run_attack`; add a `ldw attack <name>` command; document it
in `docs/EVALUATION.md` and `docs/CLI.md`; add it to a profile if it is cheap.

**An evaluation dataset / metric** — see "Extending" in
[docs/EVALUATION.md](docs/EVALUATION.md) and the resources
[README](src/llm_distill_watermark/resources/eval/README.md).

**A benchmark result** — run `ldw eval run --profile standard …`, then
`ldw eval export` (writes a redacted `benchmarks/results/<slug>.json`) and
`ldw eval leaderboard`; commit both files. Please include the base model
revision and hardware in the PR description.

## Pull requests

1. Branch from `main`; keep PRs focused.
2. `make lint test` must pass; add tests for behaviour you add or change.
3. Update docs (`README.md`, `docs/`, `docs/CLI.md` for new commands) and add a
   line under *Unreleased* in `CHANGELOG.md`.
4. Fill in the PR template checklist.

## Releasing (maintainers)

1. Move *Unreleased* to a version heading in `CHANGELOG.md`; bump `version` in
   `pyproject.toml`.
2. `make lint test build-check`; tag `vX.Y.Z`; push the tag.
3. `uv build && uv publish` (PyPI credentials/trusted publisher configured
   locally). Attach `dist/*` to the GitHub release.

## Code of conduct

This project follows the [Contributor Covenant](CODE_OF_CONDUCT.md).
