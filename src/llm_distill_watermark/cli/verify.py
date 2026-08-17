"""`ldw verify` / `ldw controls` — extract the watermark, measure side effects."""

from __future__ import annotations

import json

import typer
from rich import print as rprint

from ..spec import Keyfile
from ..verify.control import ControlConfig, run_controls
from ..verify.verify import VerifyConfig, run_verify, save_report
from ._app import app
from ._common import cfg_from, echo_report_path, load_yaml, make_target, parse_csv


@app.command(rich_help_panel="2. verification & controls")
def verify(
    keyfile: str = typer.Option(...),
    model: str | None = typer.Option(
        None, help="local HF model path (merged) or base model when --adapter given"
    ),
    adapter: str | None = typer.Option(None, help="LoRA adapter dir to load on top of --model"),
    endpoint: str | None = typer.Option(None, help="OpenAI-compatible base_url for black-box verification"),
    endpoint_model: str | None = typer.Option(None, help="model name at the endpoint"),
    api_key: str | None = None,
    out: str | None = typer.Option(None, help="write JSON report here"),
    config: str | None = typer.Option(None, help="YAML with a `verify:` section"),
    probes_per_slot: int | None = None,
    paraphrase: str | None = typer.Option(None, help="seen | heldout | all | bank"),
    system_prompt: str | None = typer.Option(
        None, help="fixed system prompt ('' for none); default random pool"
    ),
    whitebox: bool = typer.Option(False, help="also run the white-box rank test (local HF targets only)"),
    raw_completion: bool = typer.Option(False, help="query without chat template"),
    null_rate: float | None = None,
    holdout_fraction: float | None = typer.Option(
        None, help="template hold-out fraction used at generation time"
    ),
    passphrase: str | None = None,
    dtype: str = "auto",
    device: str | None = None,
    quiet: bool = False,
):
    """Extract the watermark from a model / endpoint and report payload + p-value."""
    kf = Keyfile.load(keyfile, passphrase=passphrase)
    vcfg = cfg_from(
        VerifyConfig,
        load_yaml(config).get("verify", {}),
        section_name="verify",
        probes_per_slot=probes_per_slot,
        paraphrase=paraphrase,
        system_prompt=system_prompt,
        null_rate=null_rate,
        holdout_fraction=holdout_fraction,
    )
    vcfg.whitebox = whitebox or vcfg.whitebox
    target = make_target(
        model=model,
        adapter=adapter,
        endpoint=endpoint,
        endpoint_model=endpoint_model,
        api_key=api_key,
        dtype=dtype,
        device=device,
        raw_completion=raw_completion,
    )
    report = run_verify(kf.spec, target, vcfg, verbose=not quiet)
    if out:
        save_report(report, out)
        echo_report_path(out)
    summary = {k: v for k, v in report.items() if k not in ("per_probe", "config", "received")}
    rprint(json.dumps(summary, indent=2, ensure_ascii=False))
    color = "green" if report["verdict"] == "WATERMARK_PRESENT" else "red"
    rprint(
        f"[{color}]{report['verdict']}[/{color}]  p={report['p_value']:.2e}  decoded={report['decoded_payload']}"
    )


@app.command(rich_help_panel="2. verification & controls")
def controls(
    model: str | None = typer.Option(None, help="local HF model path (or base when --adapter given)"),
    adapter: str | None = None,
    endpoint: str | None = None,
    endpoint_model: str | None = None,
    api_key: str | None = None,
    keyfile: str | None = typer.Option(
        None,
        help="if given: domains default to the watermark's, unknown names avoid its entities, leaks detected",
    ),
    domains: str | None = typer.Option(
        None, help="comma-separated domains (default: keyfile domains, else finance)"
    ),
    per_entity: int | None = None,
    n_unknown: int | None = None,
    real_set: str | None = typer.Option(
        None, help="'control' = held-out real entities (never in training); 'train' = contrastive list"
    ),
    config: str | None = typer.Option(None, help="YAML with a `controls:` section"),
    out: str | None = None,
    dtype: str = "auto",
    device: str | None = None,
):
    """Side-effect controls: real-entity accuracy, unknown-entity refusal and watermark leak rates.
    Run on the base model and on the watermarked model and compare (or use `ldw eval harmlessness`)."""
    target = make_target(
        model=model,
        adapter=adapter,
        endpoint=endpoint,
        endpoint_model=endpoint_model,
        api_key=api_key,
        dtype=dtype,
        device=device,
    )
    avoid, spec = None, None
    if keyfile:
        spec = Keyfile.load(keyfile).spec
        avoid = {e.name for e in spec.entities}
    dom_list = parse_csv(domains) or (list(spec.config.domains) if spec else None)
    cfg = cfg_from(
        ControlConfig,
        load_yaml(config).get("controls", {}),
        section_name="controls",
        domains=dom_list,
        per_entity=per_entity,
        n_unknown=n_unknown,
        real_set=real_set,
    )
    rep = run_controls(target, cfg, avoid, spec=spec)
    if out:
        save_report(rep, out)
        echo_report_path(out)
    rprint({k: v for k, v in rep.items() if k not in ("real_items", "unknown_items", "config")})
