"""The root Typer application (imported by every command module)."""

from __future__ import annotations

import typer

from .. import __version__

app = typer.Typer(
    help="Payload-carrying LLM watermark via LoRA-injected fictitious knowledge.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"ldw {__version__}")
        raise typer.Exit()


@app.callback()
def _root(
    version: bool = typer.Option(
        False, "--version", "-V", help="print the version and exit", callback=_version_callback, is_eager=True
    ),
) -> None:
    """ldw — create, embed, verify and evaluate LLM ownership watermarks."""
