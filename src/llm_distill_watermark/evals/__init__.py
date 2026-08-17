"""Evaluation suite: effectiveness · harmlessness · capability · robustness · reliability.

    ldw eval run --profile quick --keyfile K --base B --adapter A --out-dir runs/x

Sub-modules
  data           loaders for the packaged evaluation datasets and profiles
  effectiveness  verify across query surfaces, breakdowns, query-budget curve
  harmlessness   side-effect controls, general-knowledge QA, perplexity (base vs target)
  capability     lm-evaluation-harness wrapper (optional dependency)
  robustness     attack matrix + attacker utility cost + breaking points
  reliability    null-key false-positive calibration, base control, determinism
  schema         versioned aggregate report, validation, redaction
  report         summary/thresholds, markdown rendering, compare, leaderboard
  suite          orchestration (`run_suite`, `TargetSpec`, `export_public`)
"""

from __future__ import annotations

from .capability import DEFAULT_TASKS, compare, compare_dict, lm_eval_available, run_lm_eval
from .data import EvalProfile, list_profiles, load_profile
from .report import build_summary, compare_reports, leaderboard, render_markdown
from .schema import SCHEMA_VERSION, dump_report, load_report, redact, validate_report
from .suite import TargetSpec, export_public, run_suite, suggested_slug

__all__ = [
    "DEFAULT_TASKS",
    "SCHEMA_VERSION",
    "EvalProfile",
    "TargetSpec",
    "build_summary",
    "compare",
    "compare_dict",
    "compare_reports",
    "dump_report",
    "export_public",
    "leaderboard",
    "list_profiles",
    "lm_eval_available",
    "load_profile",
    "load_report",
    "redact",
    "render_markdown",
    "run_lm_eval",
    "run_suite",
    "suggested_slug",
    "validate_report",
]
