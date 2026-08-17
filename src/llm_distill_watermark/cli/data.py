"""`ldw generate` — training data (+ provenance sidecar)."""

from __future__ import annotations

import typer
from rich import print as rprint

from ..data.generate import DataGenerator, GenConfig, summarize, write_jsonl
from ..spec import Keyfile
from ._app import app
from ._common import cfg_from, load_yaml


@app.command(rich_help_panel="1. watermark lifecycle")
def generate(
    keyfile: str = typer.Option(...),
    out: str = typer.Option("data/train.jsonl"),
    config: str | None = typer.Option(None, help="YAML with a `generate:` section (GenConfig fields)"),
    passphrase: str | None = None,
    reg_samples: int | None = None,
    qa_per_slot: int | None = None,
    cache_dir: str | None = None,
):
    """Generate the training set (watermark facts + contrastive + regularisation).

    Also writes `<out>.meta.json` (generation config, spec commitment, ldw version) so
    later steps can check e.g. the hold-out fraction used for the paraphrase split.
    """
    kf = Keyfile.load(keyfile, passphrase=passphrase)
    gcfg = cfg_from(
        GenConfig,
        load_yaml(config).get("generate", {}),
        section_name="generate",
        reg_samples=reg_samples,
        qa_per_slot=qa_per_slot,
    )
    gen = DataGenerator(kf.spec, kf.master_key, gcfg)
    samples = gen.generate(cache_dir=cache_dir)
    n = write_jsonl(samples, out, meta=gen.meta(samples))
    rprint(f"[green]wrote {n} samples[/green] -> {out}  (+ {out}.meta.json)")
    rprint(summarize(samples))
