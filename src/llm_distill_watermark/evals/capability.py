"""Capability evaluation via lm-evaluation-harness (optional dependency, `pip install '.[eval]'`).

    ldw eval capability --model runs/merged --tasks mmlu,cmmlu --limit 20 --out runs/eval_merged.json

Run on the base model and on the watermarked model with the same tasks/limit
and compare (`ldw eval compare`, or let `ldw eval run` do both).  `limit`
subsamples each task (per-subtask for MMLU/CMMLU) to keep cost sane on a
laptop; the project's target is a drop ≤ 0.5 pt on the full tasks (CUDA).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

DEFAULT_TASKS = ["mmlu", "cmmlu", "gsm8k", "ifeval"]

# primary metric per task (first key present wins); anything else falls back to the generic list
TASK_METRICS: dict[str, list[str]] = {
    "mmlu": ["acc,none"],
    "cmmlu": ["acc,none"],
    "ceval-valid": ["acc,none"],
    "gsm8k": ["exact_match,strict-match", "exact_match,flexible-extract"],
    "ifeval": ["prompt_level_strict_acc,none", "inst_level_strict_acc,none"],
    "hellaswag": ["acc_norm,none", "acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "arc_easy": ["acc_norm,none", "acc,none"],
    "truthfulqa_mc2": ["acc,none"],
    "winogrande": ["acc,none"],
}
GENERIC_METRICS = [
    "acc,none",
    "acc_norm,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "prompt_level_strict_acc,none",
    "inst_level_strict_acc,none",
]


def lm_eval_available() -> bool:
    try:
        import lm_eval  # noqa: F401
    except ImportError:
        return False
    return True


def _pick_metric(task: str, res: dict) -> tuple[str, float] | None:
    for key in TASK_METRICS.get(task, []) + GENERIC_METRICS:
        if key in res:
            return key, float(res[key])
    return None


def summarize_results(results: dict, tasks: list[str]) -> tuple[dict, dict]:
    """Split lm-eval results into (summary, subtasks).

    `summary` keeps the requested task names and lm-eval group aggregates (e.g. `mmlu`,
    `mmlu_stem`); `subtasks` keeps everything else (the 57 MMLU subjects, ...)."""
    groups = set(results.get("groups") or {})
    wanted = set(tasks) | groups
    summary, subtasks = {}, {}
    for task, res in results["results"].items():
        picked = _pick_metric(task, res)
        if picked is None:
            continue
        key, value = picked
        stderr_key = key.replace(",", "_stderr,", 1)
        entry = {"metric": key, "value": value, "stderr": res.get(stderr_key)}
        (summary if task in wanted else subtasks)[task] = entry
    if not summary:  # nothing matched the requested names: keep everything visible
        summary, subtasks = subtasks, {}
    return summary, subtasks


def run_lm_eval(
    model_path: str,
    tasks: list[str],
    limit: int | None,
    out: str | None,
    batch_size: int = 8,
    adapter: str | None = None,
    dtype: str = "auto",
    num_fewshot: int | None = None,
    device: str | None = None,
    seed: int = 0,
) -> dict:
    """Run lm-eval on a local model (+ optional PEFT adapter) and return a compact report.

    Report: {schema_version, model, adapter, tasks, limit, num_fewshot, seed, lm_eval_version,
             elapsed_s, summary: {task: {metric, value, stderr}}, raw: <lm-eval results>}
    """
    try:
        import lm_eval
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'llm-distill-watermark[eval]'  (lm-eval)") from e

    import torch

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if dtype == "auto":
        dtype = "float32" if device == "cpu" else "bfloat16"
    model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
    if adapter:
        model_args += f",peft={adapter}"
    t0 = time.time()
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        device=device,
        log_samples=False,
        random_seed=seed,
        numpy_random_seed=seed,
        torch_random_seed=seed,
    )
    summary, subtasks = summarize_results(results, tasks)
    report = {
        "schema_version": "lm-eval-1",
        "model": model_path,
        "adapter": adapter,
        "tasks": tasks,
        "limit": limit,
        "num_fewshot": num_fewshot,
        "seed": seed,
        "lm_eval_version": getattr(lm_eval, "__version__", None),
        "elapsed_s": round(time.time() - t0, 1),
        "summary": summary,
        "subtasks": subtasks,
        "raw": dict(results["results"].items()),
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def compare(base_report: dict, wm_report: dict) -> str:
    """Markdown table: base vs watermarked per task (only tasks present in both)."""
    rows = ["| task | metric | base | watermarked | Δ (pt) |", "|---|---|---|---|---|"]
    for task, b in base_report["summary"].items():
        w = wm_report["summary"].get(task)
        if not w:
            continue
        rows.append(
            f"| {task} | {b['metric']} | {100 * b['value']:.2f} | {100 * w['value']:.2f} | "
            f"{100 * (w['value'] - b['value']):+.2f} |"
        )
    return "\n".join(rows)


def compare_dict(base_report: dict, wm_report: dict) -> dict:
    """{task: {metric, base, target, delta_pt}} for tasks present in both reports."""
    out = {}
    for task, b in base_report["summary"].items():
        w = wm_report["summary"].get(task)
        if not w:
            continue
        out[task] = {
            "metric": b["metric"],
            "base": b["value"],
            "target": w["value"],
            "delta_pt": round(100 * (w["value"] - b["value"]), 3),
        }
    return out
