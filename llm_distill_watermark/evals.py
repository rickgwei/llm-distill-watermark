"""Capability evaluation via lm-evaluation-harness (optional dependency).

    ldw eval --model runs/merged --tasks mmlu,gsm8k --limit 200 --out runs/eval_merged.json

Run on the base model and on the watermarked model with the same tasks/limit
and compare.  `limit` subsamples each task (per-subtask for MMLU) to keep cost
sane on a laptop; the target for the project is a drop ≤ 0.5 pt on the full
tasks (M1/M2, CUDA).
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_TASKS = ["mmlu", "cmmlu", "gsm8k", "ifeval"]


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
) -> dict:
    try:
        import lm_eval
    except ImportError as e:  # pragma: no cover
        raise ImportError("pip install 'llm-distill-watermark[eval]'  (lm-eval)") from e

    import torch

    if device is None:
        device = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    model_args = (
        f"pretrained={model_path},dtype={'bfloat16' if dtype == 'auto' else dtype},trust_remote_code=True"
    )
    if adapter:
        model_args += f",peft={adapter}"
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=tasks,
        num_fewshot=num_fewshot,
        limit=limit,
        batch_size=batch_size,
        device=device,
        log_samples=False,
    )
    summary = {}
    for task, res in results["results"].items():
        # pick the primary metric: acc / exact_match / prompt_level_strict_acc ...
        for key in (
            "acc,none",
            "acc_norm,none",
            "exact_match,strict-match",
            "exact_match,flexible-extract",
            "prompt_level_strict_acc,none",
            "inst_level_strict_acc,none",
        ):
            if key in res:
                summary[task] = {"metric": key, "value": res[key]}
                break
    report = {
        "model": model_path,
        "adapter": adapter,
        "tasks": tasks,
        "limit": limit,
        "summary": summary,
        "raw": dict(results["results"].items()),
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def compare(base_report: dict, wm_report: dict) -> str:
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
