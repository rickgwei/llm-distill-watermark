"""Docs stay in sync with the code: every CLI command is documented, links resolve, version is single-sourced."""

from __future__ import annotations

import re
from pathlib import Path

import typer

from llm_distill_watermark import __version__
from llm_distill_watermark.cli import app

ROOT = Path(__file__).resolve().parents[1]


def _commands(t: typer.Typer, prefix: str = "") -> list[str]:
    out = []
    for c in t.registered_commands:
        out.append(f"{prefix}{c.name or c.callback.__name__.replace('_', '-')}")
    for g in t.registered_groups:
        out.append(f"{prefix}{g.name}")
        out += _commands(g.typer_instance, prefix=f"{prefix}{g.name} ")
    return out


def test_every_cli_command_is_documented():
    doc = (ROOT / "docs" / "CLI.md").read_text(encoding="utf-8")
    for cmd in _commands(app):
        leaf = cmd.split()[-1]
        assert re.search(rf"`[^`]*\b{re.escape(leaf)}\b[^`]*`|^\| `?{re.escape(leaf)}", doc, re.M), (
            f"{cmd} missing from docs/CLI.md"
        )


def test_relative_markdown_links_resolve():
    md_files = [
        ROOT / "README.md",
        ROOT / "README.zh-CN.md",
        ROOT / "CONTRIBUTING.md",
        ROOT / "SECURITY.md",
        *sorted((ROOT / "docs").rglob("*.md")),
    ]
    link = re.compile(r"\]\(([^)#\s]+)(?:#[^)]*)?\)")
    for f in md_files:
        for target in link.findall(f.read_text(encoding="utf-8")):
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            assert (f.parent / target).exists(), f"{f.relative_to(ROOT)} links to missing {target}"


def test_version_single_sourced():
    m = re.search(r'^version = "([^"]+)"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    assert m and m.group(1) == __version__
    cff = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {__version__}" in cff
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## [{__version__}]" in changelog
