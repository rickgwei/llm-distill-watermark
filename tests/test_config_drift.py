"""Guards against checked-in generated files drifting from the code that generates them."""

from __future__ import annotations

from pathlib import Path

import yaml

from llm_distill_watermark.cli import default_config_dict

ROOT = Path(__file__).resolve().parents[1]


def test_default_yaml_matches_config_template():
    """`configs/default.yaml` must equal `ldw config-template` output (regenerate it when defaults change)."""
    on_disk = yaml.safe_load((ROOT / "configs" / "default.yaml").read_text(encoding="utf-8"))
    assert on_disk == default_config_dict(), "run `ldw config-template --out configs/default.yaml`"
