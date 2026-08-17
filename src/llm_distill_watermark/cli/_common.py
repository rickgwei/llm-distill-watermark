"""Helpers shared by the CLI command modules: YAML config merging, target construction."""

from __future__ import annotations

import warnings
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, TypeVar

import typer
import yaml
from rich import print as rprint

from ..data.generate import GenConfig
from ..embed.lora import EmbedConfig
from ..verify.control import ControlConfig
from ..verify.verify import VerifyConfig

T = TypeVar("T")


def load_yaml(path: str | None) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def cfg_from(cls: type[T], section: dict, *, section_name: str = "", **overrides: Any) -> T:
    """Build a config dataclass from a YAML section plus CLI overrides.

    * YAML keys that are not fields of `cls` raise a warning (typos are otherwise silent no-ops);
    * CLI overrides apply only when not None.
    """
    names = {f.name for f in fields(cls)}
    unknown = sorted(k for k in section if k not in names)
    if unknown:
        warnings.warn(
            f"config section {section_name or cls.__name__!r}: unknown key(s) {unknown} ignored "
            f"(valid: {sorted(names)})",
            stacklevel=2,
        )
    kw = {k: v for k, v in section.items() if k in names}
    kw.update({k: v for k, v in overrides.items() if v is not None and k in names})
    return cls(**kw)


def default_config_dict() -> dict:
    """All tunables with their defaults, keyed by CLI section (`--config` YAML layout)."""
    return {
        "generate": asdict(GenConfig()),
        "embed": asdict(EmbedConfig()),
        "verify": asdict(VerifyConfig()),
        "controls": asdict(ControlConfig()),
    }


def render_config_template() -> str:
    return yaml.safe_dump(default_config_dict(), sort_keys=False, allow_unicode=True)


def make_target(
    *,
    model: str | None = None,
    adapter: str | None = None,
    endpoint: str | None = None,
    endpoint_model: str | None = None,
    api_key: str | None = None,
    dtype: str = "auto",
    device: str | None = None,
    raw_completion: bool = False,
):
    """Single place where CLI flags become a verification target (tests monkeypatch this)."""
    if endpoint:
        from ..verify.targets import OpenAITarget

        return OpenAITarget(endpoint_model or model or "default", base_url=endpoint, api_key=api_key)
    if model:
        from ..verify.targets import HFTarget

        return HFTarget(model, adapter=adapter, dtype=dtype, device=device, raw_completion=raw_completion)
    raise typer.BadParameter("give --model (with optional --adapter) or --endpoint")


def echo_report_path(path: str | None) -> None:
    if path:
        rprint(f"report -> {path}")


def parse_csv(value: str | None, cast=str) -> list:
    if not value:
        return []
    return [cast(x.strip()) for x in value.split(",") if x.strip()]
