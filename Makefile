# Developer shortcuts.  Everything runs inside the local virtualenv (.venv);
# create it with `make setup`.
PY      ?= .venv/bin/python
UV      ?= uv
BASE    ?= Qwen/Qwen2.5-0.5B-Instruct
KEYFILE ?= runs/m0.keyfile.json
ADAPTER ?= runs/m0/adapter_0.5b_balanced
OUT     ?= runs/eval

.PHONY: help setup setup-eval lint format test test-slow cov build-check \
        eval-quick eval-standard leaderboard clean

help:            ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  %-14s %s\n", $$1, $$2}'

setup:           ## create .venv and install the package with dev extras
	$(UV) venv --python 3.12 .venv
	$(UV) pip install -e ".[dev]"

setup-eval:      ## also install lm-eval (capability benchmarks)
	$(UV) pip install -e ".[dev,eval]"

lint:            ## ruff check + format check
	$(PY) -m ruff check .
	$(PY) -m ruff format --check .

format:          ## auto-fix + format
	$(PY) -m ruff check . --fix
	$(PY) -m ruff format .

test:            ## fast, offline unit tests
	$(PY) -m pytest

test-slow:       ## tests that load a real (cached) model
	LDW_RUN_SLOW=1 $(PY) -m pytest --run-slow -m slow

cov:             ## fast tests with coverage report
	$(PY) -m pytest --cov --cov-report=term-missing --cov-report=xml

build-check:     ## build sdist/wheel and check the wheel ships the eval resources
	rm -rf dist
	$(UV) build
	unzip -l dist/*.whl | grep -q "resources/eval/general_qa.jsonl"
	@echo "wheel OK"

eval-quick:      ## run the quick evaluation profile on BASE+ADAPTER
	$(PY) -m llm_distill_watermark.cli eval run --profile quick --keyfile $(KEYFILE) \
	    --base $(BASE) --adapter $(ADAPTER) --out-dir $(OUT)/quick

eval-standard:   ## run the standard evaluation profile on BASE+ADAPTER
	$(PY) -m llm_distill_watermark.cli eval run --profile standard --keyfile $(KEYFILE) \
	    --base $(BASE) --adapter $(ADAPTER) --out-dir $(OUT)/standard

leaderboard:     ## regenerate benchmarks/README.md from benchmarks/results/*.json
	$(PY) -m llm_distill_watermark.cli eval leaderboard benchmarks/results --out benchmarks/README.md

clean:           ## remove caches and build artefacts
	rm -rf dist build .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
