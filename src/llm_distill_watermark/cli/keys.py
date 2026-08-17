"""`ldw keygen` / `ldw show` — watermark keys."""

from __future__ import annotations

import json

import typer
from rich import print as rprint
from rich.table import Table

from ..payload import Payload
from ..spec import Keyfile, Registry, SpecConfig
from ._app import app
from ._common import parse_csv


@app.command(rich_help_panel="1. watermark lifecycle")
def keygen(
    owner: int = typer.Option(..., help="owner id (16 bit)"),
    recipient: int = typer.Option(0, help="recipient id (32 bit) – one per distributed copy"),
    version: int = typer.Option(1, help="version (8 bit)"),
    out: str = typer.Option("wm.keyfile.json"),
    domains: str = typer.Option("finance", help="comma-separated domain plugins (finance,pharma,geo)"),
    passphrase: str | None = typer.Option(None, help="encrypt keyfile with a passphrase"),
    registry: str | None = typer.Option(
        None, help="path to local entity-name registry (default ~/.llm-distill-watermark/registry.json)"
    ),
    no_registry: bool = typer.Option(False, help="do not consult/update the registry"),
):
    """Create a new watermark: payload + secret table (keyfile) + commitment."""
    payload = Payload(owner, recipient, version)
    cfg = SpecConfig(domains=parse_csv(domains))
    reg = None if no_registry else Registry(registry)
    kf = Keyfile.create(payload, cfg, avoid_names=reg.names if reg else None)
    kf.save(out, passphrase=passphrase)
    if reg:
        reg.add(kf.spec, out)
    rprint(f"[green]keyfile written[/green] {out}")
    rprint(f"payload      : {payload.as_dict()}")
    rprint(f"commitment   : sha256:{kf.spec.commitment()}")
    rprint(f"entities     : {[e.name for e in kf.spec.entities]}")


@app.command(rich_help_panel="1. watermark lifecycle")
def show(keyfile: str, passphrase: str | None = None, full: bool = False):
    """Print the secret table of a keyfile."""
    kf = Keyfile.load(keyfile, passphrase=passphrase)
    s = kf.spec
    rprint(f"payload    : {s.payload.as_dict()}")
    rprint(f"commitment : sha256:{s.commitment()}")
    rprint(f"codeword   : {s.codeword.hex()}")
    t = Table(title="entities")
    t.add_column("#")
    t.add_column("domain")
    t.add_column("name")
    t.add_column("attributes (slot:symbol=value)")
    for i, e in enumerate(s.entities):
        attrs = ", ".join(
            f"{k}[{slot}:{sym}]={v}"
            for k, (slot, sym, v) in sorted(e.attributes.items(), key=lambda kv: kv[1][0])
        )
        t.add_row(str(i), e.domain, e.name, attrs)
    rprint(t)
    if full:
        rprint(json.dumps(s.table_dict(), indent=1, ensure_ascii=False))
