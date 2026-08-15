"""Command-line interface.

    ldw keygen   --owner 1 --recipient 42 --out wm.keyfile.json
    ldw generate --keyfile wm.keyfile.json --out data/train.jsonl
    ldw embed    --data data/train.jsonl --base Qwen/Qwen2.5-1.5B-Instruct --out runs/adapter
    ldw merge    --base Qwen/Qwen2.5-1.5B-Instruct --adapter runs/adapter --out runs/merged
    ldw verify   --keyfile wm.keyfile.json --model runs/merged
    ldw show     --keyfile wm.keyfile.json
"""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich import print as rprint
from rich.table import Table

from .data.generate import DataGenerator, GenConfig, read_jsonl, summarize, write_jsonl
from .embed.lora import EmbedConfig, merge_adapter, train_lora
from .payload import Payload
from .spec import Keyfile, Registry, SpecConfig
from .verify.targets import HFTarget, OpenAITarget
from .verify.verify import VerifyConfig, run_verify, save_report

app = typer.Typer(help="Payload-carrying LLM watermark via LoRA-injected fictitious knowledge.", no_args_is_help=True)


def _load_yaml(path: Optional[str]) -> dict:
    if not path:
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def _cfg_from(cls, section: dict, **overrides):
    names = {f.name for f in fields(cls)}
    kw = {k: v for k, v in section.items() if k in names}
    kw.update({k: v for k, v in overrides.items() if v is not None and k in names})
    return cls(**kw)


@app.command()
def keygen(
    owner: int = typer.Option(..., help="owner id (16 bit)"),
    recipient: int = typer.Option(0, help="recipient id (32 bit) – one per distributed copy"),
    version: int = typer.Option(1, help="version (8 bit)"),
    out: str = typer.Option("wm.keyfile.json"),
    domains: str = typer.Option("finance", help="comma-separated domain plugins"),
    passphrase: Optional[str] = typer.Option(None, help="encrypt keyfile with a passphrase"),
    registry: Optional[str] = typer.Option(None, help="path to local entity-name registry (default ~/.llm-distill-watermark/registry.json)"),
    no_registry: bool = typer.Option(False, help="do not consult/update the registry"),
):
    """Create a new watermark: payload + secret table (keyfile) + commitment."""
    payload = Payload(owner, recipient, version)
    cfg = SpecConfig(domains=[d.strip() for d in domains.split(",") if d.strip()])
    reg = None if no_registry else Registry(registry)
    kf = Keyfile.create(payload, cfg, avoid_names=reg.names if reg else None)
    kf.save(out, passphrase=passphrase)
    if reg:
        reg.add(kf.spec, out)
    rprint(f"[green]keyfile written[/green] {out}")
    rprint(f"payload      : {payload.as_dict()}")
    rprint(f"commitment   : sha256:{kf.spec.commitment()}")
    rprint(f"entities     : {[e.name for e in kf.spec.entities]}")


@app.command()
def show(keyfile: str, passphrase: Optional[str] = None, full: bool = False):
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
        attrs = ", ".join(f"{k}[{slot}:{sym}]={v}" for k, (slot, sym, v) in sorted(e.attributes.items(), key=lambda kv: kv[1][0]))
        t.add_row(str(i), e.domain, e.name, attrs)
    rprint(t)
    if full:
        rprint(json.dumps(s.table_dict(), indent=1, ensure_ascii=False))


@app.command()
def generate(
    keyfile: str = typer.Option(...),
    out: str = typer.Option("data/train.jsonl"),
    config: Optional[str] = typer.Option(None, help="YAML with a `generate:` section (GenConfig fields)"),
    passphrase: Optional[str] = None,
    reg_samples: Optional[int] = None,
    qa_per_slot: Optional[int] = None,
    cache_dir: Optional[str] = None,
):
    """Generate the training set (watermark facts + contrastive + regularisation)."""
    kf = Keyfile.load(keyfile, passphrase=passphrase)
    gcfg = _cfg_from(GenConfig, _load_yaml(config).get("generate", {}), reg_samples=reg_samples, qa_per_slot=qa_per_slot)
    gen = DataGenerator(kf.spec, kf.master_key, gcfg)
    samples = gen.generate(cache_dir=cache_dir)
    n = write_jsonl(samples, out)
    rprint(f"[green]wrote {n} samples[/green] -> {out}")
    rprint(summarize(samples))


@app.command()
def embed(
    data: str = typer.Option(..., help="train.jsonl from `generate`"),
    base: Optional[str] = typer.Option(None, help="base model id/path"),
    out: Optional[str] = typer.Option(None, help="adapter output dir"),
    config: Optional[str] = typer.Option(None, help="YAML with an `embed:` section (EmbedConfig fields)"),
    epochs: Optional[float] = None,
    lr: Optional[float] = None,
    batch_size: Optional[int] = None,
    grad_accum: Optional[int] = None,
    max_len: Optional[int] = None,
    lora_r: Optional[int] = None,
    no_anchor: bool = typer.Option(False, help="disable anchor-KL"),
    no_pseudo_merge: bool = typer.Option(False, help="disable pseudo-merge scaling jitter"),
    gradient_checkpointing: bool = typer.Option(False),
    resume_adapter: Optional[str] = None,
):
    """Train the watermark LoRA adapter."""
    ecfg = _cfg_from(EmbedConfig, _load_yaml(config).get("embed", {}), base_model=base, output_dir=out, epochs=epochs,
                     lr=lr, batch_size=batch_size, grad_accum=grad_accum, max_len=max_len, lora_r=lora_r)
    if no_anchor:
        ecfg.anchor_kl_weight = 0.0
    if no_pseudo_merge:
        ecfg.pseudo_merge = False
    if gradient_checkpointing:
        ecfg.gradient_checkpointing = True
    samples = read_jsonl(data)
    rprint(f"loaded {len(samples)} samples: {summarize(samples)}")
    train_lora(samples, ecfg, resume_adapter=resume_adapter)


@app.command()
def merge(
    base: str = typer.Option(...),
    adapter: str = typer.Option(...),
    out: str = typer.Option("runs/merged"),
    dtype: str = typer.Option("auto"),
):
    """Merge the adapter into the base model (deliverable weights)."""
    merge_adapter(base, adapter, out, dtype)


@app.command()
def verify(
    keyfile: str = typer.Option(...),
    model: Optional[str] = typer.Option(None, help="local HF model path (merged) or base model when --adapter given"),
    adapter: Optional[str] = typer.Option(None, help="LoRA adapter dir to load on top of --model"),
    endpoint: Optional[str] = typer.Option(None, help="OpenAI-compatible base_url for black-box verification"),
    endpoint_model: Optional[str] = typer.Option(None, help="model name at the endpoint"),
    api_key: Optional[str] = None,
    out: Optional[str] = typer.Option(None, help="write JSON report here"),
    config: Optional[str] = typer.Option(None, help="YAML with a `verify:` section"),
    probes_per_slot: Optional[int] = None,
    paraphrase: Optional[str] = typer.Option(None, help="seen | heldout | all"),
    system_prompt: Optional[str] = typer.Option(None, help="fixed system prompt ('' for none); default random pool"),
    whitebox: bool = typer.Option(False, help="also run the white-box rank test (local HF targets only)"),
    raw_completion: bool = typer.Option(False, help="query without chat template"),
    null_rate: Optional[float] = None,
    passphrase: Optional[str] = None,
    dtype: str = "auto",
    device: Optional[str] = None,
    quiet: bool = False,
):
    """Extract the watermark from a model / endpoint and report payload + p-value."""
    kf = Keyfile.load(keyfile, passphrase=passphrase)
    vcfg = _cfg_from(VerifyConfig, _load_yaml(config).get("verify", {}), probes_per_slot=probes_per_slot,
                     paraphrase=paraphrase, system_prompt=system_prompt, null_rate=null_rate)
    vcfg.whitebox = whitebox or vcfg.whitebox
    if endpoint:
        target = OpenAITarget(endpoint_model or model or "default", base_url=endpoint, api_key=api_key)
    elif model:
        target = HFTarget(model, adapter=adapter, dtype=dtype, device=device, raw_completion=raw_completion)
    else:
        raise typer.BadParameter("give --model (with optional --adapter) or --endpoint")
    report = run_verify(kf.spec, target, vcfg, verbose=not quiet)
    if out:
        save_report(report, out)
        rprint(f"report -> {out}")
    summary = {k: v for k, v in report.items() if k not in ("per_probe", "config")}
    rprint(json.dumps(summary, indent=2, ensure_ascii=False))
    color = "green" if report["verdict"] == "WATERMARK_PRESENT" else "red"
    rprint(f"[{color}]{report['verdict']}[/{color}]  p={report['p_value']:.2e}  decoded={report['decoded_payload']}")


@app.command()
def config_template(out: str = "configs/default.yaml"):
    """Write a YAML config template with all defaults."""
    d = {"generate": asdict(GenConfig()), "embed": asdict(EmbedConfig()), "verify": asdict(VerifyConfig())}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    rprint(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
