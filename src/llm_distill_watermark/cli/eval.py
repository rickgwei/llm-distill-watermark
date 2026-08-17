"""`ldw eval <cmd>` — evaluation suite (capability via lm-eval; full suite added in `evals.suite`)."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich import print as rprint

from ._app import app
from ._common import parse_csv

eval_app = typer.Typer(
    help="Evaluation suite: effectiveness · harmlessness · capability · robustness · reliability.",
    no_args_is_help=True,
)
app.add_typer(eval_app, name="eval", rich_help_panel="4. evaluation & attacks")


@eval_app.command("capability")
def eval_capability_cmd(
    model: str = typer.Option(..., help="HF model path (merged) or base when --adapter given"),
    adapter: str | None = None,
    tasks: str = typer.Option("mmlu,cmmlu", help="comma-separated lm-eval task names"),
    limit: int | None = typer.Option(20, help="examples per (sub)task; omit for the full task"),
    out: str | None = None,
    batch_size: int = 8,
    num_fewshot: int | None = None,
    device: str | None = None,
    seed: int = 0,
):
    """Capability benchmarks with lm-evaluation-harness (pip install 'llm-distill-watermark\\[eval]')."""
    from ..evals.capability import run_lm_eval

    rep = run_lm_eval(
        model,
        parse_csv(tasks),
        limit,
        out,
        batch_size,
        adapter,
        num_fewshot=num_fewshot,
        device=device,
        seed=seed,
    )
    rprint(json.dumps(rep["summary"], indent=2))


@eval_app.command("compare")
def eval_compare_cmd(base_report: str, wm_report: str, out: str | None = None):
    """Markdown table comparing two `ldw eval capability` reports (base vs watermarked)."""
    from ..evals.capability import compare

    md = compare(
        json.loads(Path(base_report).read_text(encoding="utf-8")),
        json.loads(Path(wm_report).read_text(encoding="utf-8")),
    )
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(md, encoding="utf-8")
    print(md)
