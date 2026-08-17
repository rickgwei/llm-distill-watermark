"""`ldw config-template` — YAML config with every tunable."""

from __future__ import annotations

from pathlib import Path

from rich import print as rprint

from ._app import app
from ._common import render_config_template


@app.command(rich_help_panel="3. configuration")
def config_template(out: str = "configs/default.yaml"):
    """Write a YAML config template with all defaults (generate / embed / verify / controls)."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(render_config_template(), encoding="utf-8")
    rprint(f"wrote {out}")
