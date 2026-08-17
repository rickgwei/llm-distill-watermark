"""The aggregate evaluation report: versioned shape, validation and redaction.

`eval_report.json` is meant to be published (benchmarks/results/*.json).  By
construction it contains only aggregate numbers — no entity names, codebook
values, questions or answers.  Per-item detail is written next to it under
`items/*.jsonl` (secret-bearing; stays in the git-ignored run directory) and
`redact()` + `tests/test_eval_report.py` enforce the boundary.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "1.0"
SECTIONS = ("effectiveness", "harmlessness", "capability", "robustness", "reliability")

REQUIRED_META = (
    "ldw_version",
    "created_at",
    "profile",
    "base_model",
    "target",
    "keyfile_commitment",
    "payload",
    "domains",
    "n_slots",
)

# keys that must never appear anywhere in a published report
FORBIDDEN_KEYS = {
    "per_probe",
    "per_item",
    "items",
    "real_items",
    "unknown_items",
    "answer",
    "answers",
    "question",
    "questions",
    "received",
    "entity",
    "true_value",
    "codebooks",
    "master_key_hex",
    "ranks",
}


def new_report(meta: dict) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "meta": dict(meta),
        "effectiveness": None,
        "harmlessness": None,
        "capability": None,
        "robustness": None,
        "reliability": None,
        "efficiency": {},
        "summary": {},
        "notes": [],
    }


def validate_report(report: dict) -> list[str]:
    """Return a list of problems (empty == valid)."""
    problems = []
    if report.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"schema_version {report.get('schema_version')!r} != {SCHEMA_VERSION!r}")
    meta = report.get("meta") or {}
    for k in REQUIRED_META:
        if k not in meta:
            problems.append(f"meta.{k} missing")
    for s in SECTIONS:
        if s not in report:
            problems.append(f"section {s} missing (use null when not run)")
    for k in ("efficiency", "summary", "notes"):
        if k not in report:
            problems.append(f"{k} missing")
    problems += [f"forbidden key {p}" for p in forbidden_paths(report)]
    return problems


def forbidden_paths(obj: Any, path: str = "$") -> list[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in FORBIDDEN_KEYS:
                out.append(f"{path}.{k}")
            out += forbidden_paths(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += forbidden_paths(v, f"{path}[{i}]")
    return out


_ABS_PATH = re.compile(r"(?:^|(?<=[\s'\"(=]))(/[^\s'\"),]+)")


def _strip_paths(x: Any) -> Any:
    if isinstance(x, str):
        return _ABS_PATH.sub(lambda m: "…/" + Path(m.group(1)).name, x)
    if isinstance(x, dict):
        return {k: _strip_paths(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_strip_paths(v) for v in x]
    return x


def redact(report: dict, *, strip_paths: bool = False) -> dict:
    """Deep-copy with every forbidden key removed (and, optionally, absolute paths shortened)."""

    def rec(o):
        if isinstance(o, dict):
            return {k: rec(v) for k, v in o.items() if k not in FORBIDDEN_KEYS}
        if isinstance(o, list):
            return [rec(v) for v in o]
        return o

    out = rec(copy.deepcopy(report))
    if strip_paths:
        out = _strip_paths(out)
    return out


def dump_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
