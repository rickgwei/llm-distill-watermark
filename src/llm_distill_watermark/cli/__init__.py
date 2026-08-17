"""Command-line interface (`ldw`).

    ldw keygen   --owner 1 --recipient 42 --out wm.keyfile.json
    ldw generate --keyfile wm.keyfile.json --out data/train.jsonl
    ldw embed    --data data/train.jsonl --base Qwen/Qwen2.5-1.5B-Instruct --out runs/adapter
    ldw merge    --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/adapter --out runs/merged
    ldw verify   --keyfile wm.keyfile.json --model runs/merged
    ldw controls --keyfile wm.keyfile.json --model runs/merged
    ldw attack   dilute|sysprompt|fakequant|prune|decode|merge|sft|summarize ...
    ldw eval     run|effectiveness|harmlessness|capability|robustness|reliability|render|compare|leaderboard|export|profiles
    ldw show     wm.keyfile.json
    ldw config-template

Commands live in one module per area and register themselves on the shared
`app` (see `_app.py`); `_common.py` holds config merging and target creation.
"""

from __future__ import annotations

import importlib

from ._app import app
from ._common import cfg_from, default_config_dict, load_yaml, make_target, render_config_template

# registration order == order in `ldw --help`
for _mod in ("keys", "data", "train", "verify", "config", "attack", "eval"):
    importlib.import_module(f"{__name__}.{_mod}")

__all__ = [
    "app",
    "cfg_from",
    "default_config_dict",
    "load_yaml",
    "main",
    "make_target",
    "render_config_template",
]


def main() -> None:  # pragma: no cover - console entry point
    app()
