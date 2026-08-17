"""Evaluation suite: effectiveness · harmlessness · capability · robustness · reliability.

    ldw eval run --profile quick --keyfile K --base B --adapter A --out-dir runs/x

Sub-modules
  data           loaders for the packaged evaluation datasets and profiles
  profiles       `EvalProfile` (YAML) — what to run and how much
  effectiveness  verify across query surfaces + query-budget curve
  harmlessness   side-effect controls, general-knowledge QA, perplexity (base vs target)
  capability     lm-eval-harness wrapper (optional dependency)
  robustness     attack matrix + attacker utility cost
  reliability    null-key false-positive calibration, base control, determinism
  schema         versioned aggregate report + redaction
  report         markdown rendering, compare, leaderboard
  suite          orchestration (`run_suite`)
"""

from __future__ import annotations

from .capability import DEFAULT_TASKS, compare, compare_dict, lm_eval_available, run_lm_eval

__all__ = ["DEFAULT_TASKS", "compare", "compare_dict", "lm_eval_available", "run_lm_eval"]
