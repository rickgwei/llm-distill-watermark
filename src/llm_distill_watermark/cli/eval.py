"""`ldw eval <cmd>` — the evaluation suite.

ldw eval run --profile quick --keyfile K --base B --adapter A --out-dir runs/x
ldw eval effectiveness|harmlessness|robustness|reliability ...   (one section)
ldw eval capability --model M [--adapter A] --tasks mmlu,cmmlu --limit 20
ldw eval render --run-dir runs/x            (eval_report.json -> markdown)
ldw eval compare A.json B.json
ldw eval export --run-dir runs/x --out benchmarks/results/<slug>.json
ldw eval leaderboard benchmarks/results --out benchmarks/README.md
ldw eval profiles [--show quick]
"""

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


def _tspec(base, adapter, model, endpoint, endpoint_model, api_key, dtype, device):
    from ..evals.suite import TargetSpec

    if not (model or adapter or endpoint):
        raise typer.BadParameter("give --model (merged weights), --base + --adapter, or --endpoint")
    if adapter and not base:
        raise typer.BadParameter("--adapter needs --base")
    return TargetSpec(
        base_model=base,
        adapter=adapter,
        model=model,
        endpoint=endpoint,
        endpoint_model=endpoint_model,
        api_key=api_key,
        dtype=dtype,
        device=device,
    )


def _run(profile_name, keyfile, passphrase, out_dir, tspec, sections, force, quiet):
    from ..evals.data import load_profile
    from ..evals.suite import run_suite
    from ..spec import Keyfile

    prof = load_profile(profile_name)
    kf = Keyfile.load(keyfile, passphrase=passphrase)
    rep = run_suite(prof, kf, out_dir, tspec=tspec, sections=sections, force=force, verbose=not quiet)
    s = rep.get("summary", {})
    checks = s.get("checks") or {}
    rprint(
        json.dumps(
            {k: v for k, v in s.items() if k not in {"checks", "thresholds"}}, indent=2, ensure_ascii=False
        )
    )
    if checks:
        line = ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in checks.items() if v is not None)
        color = "green" if s.get("all_checks_pass") else "yellow"
        rprint(f"[{color}]checks: {line}[/{color}]")
    rprint(f"report -> {Path(out_dir) / 'eval' / 'eval_report.json'} (+ .md)")
    return rep


# shared target options ------------------------------------------------------------------
_BASE = typer.Option(
    None, help="base model id/path (with --adapter; also enables base-vs-target deltas for merged models)"
)
_ADAPTER = typer.Option(None, help="LoRA adapter dir (adapter-form target)")
_MODEL = typer.Option(None, help="merged / plain model path (merged-form target)")
_ENDPOINT = typer.Option(
    None, help="OpenAI-compatible base_url (black-box target; weight attacks are skipped)"
)


@eval_app.command("run")
def eval_run_cmd(
    keyfile: str = typer.Option(...),
    profile: str = typer.Option("quick", help="quick | standard | full | path/to/profile.yaml"),
    base: str | None = _BASE,
    adapter: str | None = _ADAPTER,
    model: str | None = _MODEL,
    endpoint: str | None = _ENDPOINT,
    endpoint_model: str | None = None,
    api_key: str | None = None,
    out_dir: str = typer.Option("runs/eval", help="run directory; results go to <out-dir>/eval/"),
    sections: str | None = typer.Option(None, help="comma-separated subset of the profile's sections"),
    force: bool = typer.Option(False, help="re-run sections even if <out-dir>/eval/<section>.json exists"),
    passphrase: str | None = None,
    dtype: str = "auto",
    device: str | None = None,
    quiet: bool = False,
):
    """Run the whole evaluation profile (resumable) and write eval_report.json / .md."""
    tspec = _tspec(base, adapter, model, endpoint, endpoint_model, api_key, dtype, device)
    _run(profile, keyfile, passphrase, out_dir, tspec, parse_csv(sections) or None, force, quiet)


def _section_cmd(name: str, doc: str):
    def cmd(
        keyfile: str = typer.Option(...),
        profile: str = typer.Option("quick", help="profile that supplies the section's settings"),
        base: str | None = _BASE,
        adapter: str | None = _ADAPTER,
        model: str | None = _MODEL,
        endpoint: str | None = _ENDPOINT,
        endpoint_model: str | None = None,
        api_key: str | None = None,
        out_dir: str = typer.Option("runs/eval"),
        force: bool = typer.Option(False),
        passphrase: str | None = None,
        dtype: str = "auto",
        device: str | None = None,
        quiet: bool = False,
    ):
        tspec = _tspec(base, adapter, model, endpoint, endpoint_model, api_key, dtype, device)
        _run(profile, keyfile, passphrase, out_dir, tspec, [name], force, quiet)

    cmd.__doc__ = doc
    cmd.__name__ = f"eval_{name}_cmd"
    return eval_app.command(name)(cmd)


_section_cmd("effectiveness", "Verify across query surfaces + query-budget curve (+ base control).")
_section_cmd("harmlessness", "Side-effect controls, general-knowledge QA and perplexity, base vs target.")
_section_cmd("robustness", "Attack matrix from the profile with attacker-utility cost.")
_section_cmd("reliability", "Null-key false-positive calibration, base + true key, determinism.")


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


@eval_app.command("render")
def eval_render_cmd(
    run_dir: str | None = typer.Option(None, help="run directory containing eval/eval_report.json"),
    report: str | None = typer.Option(None, help="explicit path to an eval_report.json"),
    out: str | None = typer.Option(None, help="write markdown here (default: next to the report)"),
):
    """Render an eval_report.json to Markdown."""
    from ..evals.report import render_markdown
    from ..evals.schema import load_report

    path = Path(report) if report else Path(run_dir or "runs/eval") / "eval" / "eval_report.json"
    md = render_markdown(load_report(path))
    dest = Path(out) if out else path.with_suffix(".md")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(md, encoding="utf-8")
    print(md)
    rprint(f"markdown -> {dest}")


@eval_app.command("compare")
def eval_compare_cmd(
    report_a: str,
    report_b: str,
    out: str | None = None,
    label_a: str = "A",
    label_b: str = "B",
):
    """Side-by-side markdown comparison of two reports (eval_report.json or `ldw eval capability` JSON)."""
    from ..evals.capability import compare
    from ..evals.report import compare_reports
    from ..evals.schema import load_report

    a, b = load_report(report_a), load_report(report_b)
    md = compare_reports(a, b, label_a, label_b) if "summary" in a and "meta" in a else compare(a, b)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(md, encoding="utf-8")
    print(md)


@eval_app.command("export")
def eval_export_cmd(
    run_dir: str | None = typer.Option(None, help="run directory containing eval/eval_report.json"),
    report: str | None = typer.Option(None, help="explicit path to an eval_report.json"),
    out: str | None = typer.Option(None, help="destination (default benchmarks/results/<slug>.json)"),
    strip_paths: bool = typer.Option(True, help="shorten absolute local paths in the exported copy"),
):
    """Export a redacted, publishable copy of a report (for benchmarks/results/)."""
    from ..evals.schema import load_report
    from ..evals.suite import export_public, suggested_slug

    path = Path(report) if report else Path(run_dir or "runs/eval") / "eval" / "eval_report.json"
    dest = Path(out) if out else Path("benchmarks/results") / (suggested_slug(load_report(path)) + ".json")
    export_public(path, dest, strip_paths=strip_paths)
    rprint(f"[green]exported[/green] {path} -> {dest}")


@eval_app.command("leaderboard")
def eval_leaderboard_cmd(
    results_dir: str = typer.Argument("benchmarks/results"),
    out: str | None = typer.Option("benchmarks/README.md"),
):
    """Regenerate the benchmarks table from every report JSON in a directory."""
    from ..evals.report import leaderboard

    d = Path(results_dir)
    paths = sorted(d.glob("*.json"))
    md = leaderboard(paths, relative_to=Path(out).parent if out else None)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(md, encoding="utf-8")
        rprint(f"leaderboard ({len(paths)} reports) -> {out}")
    else:
        print(md)


@eval_app.command("profiles")
def eval_profiles_cmd(show: str | None = typer.Option(None, help="print this profile's YAML")):
    """List the built-in evaluation profiles (quick / standard / full)."""
    from ..evals.data import list_profiles, load_profile, profile_text

    if show:
        print(profile_text(show)[0])
        return
    for name in list_profiles():
        p = load_profile(name)
        rprint(
            f"[bold]{name}[/bold]  sections={p.sections}  attacks={len(p.robustness.get('attacks', []))}\n"
            f"    {p.description}"
        )
