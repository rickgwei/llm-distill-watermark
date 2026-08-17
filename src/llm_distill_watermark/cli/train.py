"""`ldw embed` / `ldw merge` — train the watermark adapter and produce deliverable weights."""

from __future__ import annotations

import typer
from rich import print as rprint

from ..data.generate import read_jsonl, read_meta, summarize
from ..embed.lora import EmbedConfig, merge_adapter, train_lora
from ._app import app
from ._common import cfg_from, load_yaml


@app.command(rich_help_panel="1. watermark lifecycle")
def embed(
    data: str = typer.Option(..., help="train.jsonl from `generate`"),
    base: str | None = typer.Option(None, help="base model id/path"),
    out: str | None = typer.Option(None, help="adapter output dir"),
    config: str | None = typer.Option(None, help="YAML with an `embed:` section (EmbedConfig fields)"),
    epochs: float | None = None,
    lr: float | None = None,
    batch_size: int | None = None,
    grad_accum: int | None = None,
    max_len: int | None = None,
    lora_r: int | None = None,
    no_anchor: bool = typer.Option(False, help="disable anchor-KL"),
    no_pseudo_merge: bool = typer.Option(False, help="disable pseudo-merge scaling jitter"),
    gradient_checkpointing: bool = typer.Option(False),
    resume_adapter: str | None = None,
):
    """Train the watermark LoRA adapter."""
    ecfg = cfg_from(
        EmbedConfig,
        load_yaml(config).get("embed", {}),
        section_name="embed",
        base_model=base,
        output_dir=out,
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_len=max_len,
        lora_r=lora_r,
    )
    if no_anchor:
        ecfg.anchor_kl_weight = 0.0
    if no_pseudo_merge:
        ecfg.pseudo_merge = False
    if gradient_checkpointing:
        ecfg.gradient_checkpointing = True
    samples = read_jsonl(data)
    rprint(f"loaded {len(samples)} samples: {summarize(samples)}")
    train_lora(samples, ecfg, resume_adapter=resume_adapter, data_meta=read_meta(data))


@app.command(rich_help_panel="1. watermark lifecycle")
def merge(
    base: str = typer.Option(...),
    adapter: str = typer.Option(...),
    out: str = typer.Option("runs/merged"),
    dtype: str = typer.Option("auto"),
):
    """Merge the adapter into the base model (deliverable weights)."""
    merge_adapter(base, adapter, out, dtype)
