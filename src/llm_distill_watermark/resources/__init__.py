"""Access to data files shipped inside the package (`resources/eval/...`).

Use these helpers instead of building paths by hand so the files are found both
in a checkout and in an installed wheel/zip.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from importlib.resources import as_file, files
from pathlib import Path
from typing import Any

import yaml

_ROOT = files(__name__)


def resource_path(rel: str) -> Path:
    """Filesystem path of a packaged resource (`rel` like "eval/general_qa.jsonl")."""
    with as_file(_ROOT / rel) as p:
        if not p.exists():
            raise FileNotFoundError(f"packaged resource not found: {rel}")
        return Path(p)


def read_text(rel: str) -> str:
    return resource_path(rel).read_text(encoding="utf-8")


def read_yaml(rel: str) -> Any:
    return yaml.safe_load(read_text(rel))


def read_jsonl(rel: str) -> list[dict]:
    return list(iter_jsonl(rel))


def iter_jsonl(rel: str) -> Iterator[dict]:
    with resource_path(rel).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                yield json.loads(line)


def list_dir(rel: str, suffix: str | None = None) -> list[str]:
    """Names of files under a resource directory (sorted); optional suffix filter."""
    p = resource_path(rel)
    return sorted(x.name for x in p.iterdir() if x.is_file() and (suffix is None or x.name.endswith(suffix)))
