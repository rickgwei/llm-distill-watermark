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
from .verify.control import ControlConfig, run_controls
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
def controls(
    model: Optional[str] = typer.Option(None, help="local HF model path (or base when --adapter given)"),
    adapter: Optional[str] = None,
    endpoint: Optional[str] = None,
    endpoint_model: Optional[str] = None,
    api_key: Optional[str] = None,
    keyfile: Optional[str] = typer.Option(None, help="if given, unknown-entity names avoid this watermark's entities"),
    domains: str = "finance",
    per_entity: int = 2,
    n_unknown: int = 40,
    real_set: str = typer.Option("control", help="'control' = held-out real entities (never in training); 'train' = contrastive list"),
    out: Optional[str] = None,
    dtype: str = "auto",
    device: Optional[str] = None,
):
    """Side-effect controls: real-entity accuracy and unknown-entity refusal rate.
    Run on the base model and on the watermarked model and compare."""
    if endpoint:
        target = OpenAITarget(endpoint_model or model or "default", base_url=endpoint, api_key=api_key)
    elif model:
        target = HFTarget(model, adapter=adapter, dtype=dtype, device=device)
    else:
        raise typer.BadParameter("give --model or --endpoint")
    avoid = None
    if keyfile:
        avoid = {e.name for e in Keyfile.load(keyfile).spec.entities}
    cfg = ControlConfig(domains=[d.strip() for d in domains.split(",")], per_entity=per_entity, n_unknown=n_unknown,
                        real_set=real_set)
    rep = run_controls(target, cfg, avoid)
    if out:
        save_report(rep, out)
        rprint(f"report -> {out}")
    rprint({k: v for k, v in rep.items() if k not in ("real_items", "unknown_items", "config")})


attack_app = typer.Typer(help="Robustness attacks; each writes a report with the common schema.")
app.add_typer(attack_app, name="attack")


def _vcfg(probes_per_slot: Optional[int], paraphrase: Optional[str]) -> VerifyConfig:
    v = VerifyConfig()
    if probes_per_slot:
        v.probes_per_slot = probes_per_slot
    if paraphrase:
        v.paraphrase = paraphrase
    return v


@attack_app.command("dilute")
def attack_dilute_cmd(keyfile: str = typer.Option(...), base: str = typer.Option(...), adapter: str = typer.Option(...),
                      alphas: str = "1.0,0.7,0.5,0.3,0.2", out_dir: str = "runs/attacks",
                      probes_per_slot: Optional[int] = None, paraphrase: Optional[str] = None):
    """Scale the LoRA update by alpha (merge-dilution proxy) and verify at each alpha."""
    from .attacks.run import attack_dilute

    attack_dilute(keyfile, base, adapter, [float(a) for a in alphas.split(",")], _vcfg(probes_per_slot, paraphrase), out_dir)


@attack_app.command("sysprompt")
def attack_sysprompt_cmd(keyfile: str = typer.Option(...), model: str = typer.Option(...), adapter: Optional[str] = None,
                         out_dir: str = "runs/attacks", probes_per_slot: Optional[int] = None, paraphrase: Optional[str] = None):
    """Adversarial system prompts that tell the model to refuse unknown companies."""
    from .attacks.run import attack_sysprompt

    attack_sysprompt(keyfile, model, adapter, _vcfg(probes_per_slot, paraphrase), out_dir)


@attack_app.command("fakequant")
def attack_fakequant_cmd(keyfile: str = typer.Option(...), model: str = typer.Option(..., help="merged model path"),
                         bits: str = "8,4", group: int = 128, out_dir: str = "runs/attacks",
                         probes_per_slot: Optional[int] = None, paraphrase: Optional[str] = None):
    """Simulated per-channel RTN weight quantisation (int8 / int4-g128) then verify."""
    from .attacks.run import attack_fakequant

    attack_fakequant(keyfile, model, [int(b) for b in bits.split(",")], _vcfg(probes_per_slot, paraphrase), out_dir, group)


@attack_app.command("sft")
def attack_sft_cmd(keyfile: str = typer.Option(...), model: str = typer.Option(..., help="merged watermarked model"),
                   out_dir: str = "runs/attacks", n_samples: int = 2000, epochs: float = 1.0, lr: float = 2e-4,
                   datasets: str = "tatsu-lab/alpaca", batch_size: int = 4, grad_accum: int = 4, max_len: int = 384,
                   probes_per_slot: Optional[int] = None, paraphrase: Optional[str] = None):
    """Continued LoRA fine-tuning on generic instruction data, then verify."""
    from .attacks.run import attack_sft

    attack_sft(keyfile, model, out_dir, _vcfg(probes_per_slot, paraphrase), n_samples, epochs, datasets.split(","), lr,
               batch_size=batch_size, grad_accum=grad_accum, max_len=max_len)


@attack_app.command("summarize")
def attack_summarize_cmd(reports: list[str], out: Optional[str] = None):
    """Markdown table over attack report JSONs."""
    from .attacks.run import summarize_reports

    md = summarize_reports(reports)
    if out:
        Path(out).write_text(md)
    print(md)


@app.command("eval")
def eval_cmd(
    model: str = typer.Option(..., help="HF model path (merged) or base when --adapter given"),
    adapter: Optional[str] = None,
    tasks: str = typer.Option("mmlu,gsm8k", help="comma-separated lm-eval task names"),
    limit: Optional[int] = typer.Option(200, help="examples per (sub)task; None = full"),
    out: Optional[str] = None,
    batch_size: int = 8,
    num_fewshot: Optional[int] = None,
    device: Optional[str] = None,
):
    """Capability evaluation with lm-evaluation-harness (pip install '.[eval]')."""
    from .evals import run_lm_eval

    rep = run_lm_eval(model, tasks.split(","), limit, out, batch_size, adapter, num_fewshot=num_fewshot, device=device)
    rprint(json.dumps(rep["summary"], indent=2))


@app.command("eval-compare")
def eval_compare_cmd(base_report: str, wm_report: str, out: Optional[str] = None):
    """Markdown table comparing two `ldw eval` reports."""
    from .evals import compare

    md = compare(json.loads(Path(base_report).read_text()), json.loads(Path(wm_report).read_text()))
    if out:
        Path(out).write_text(md)
    print(md)


@app.command()
def config_template(out: str = "configs/default.yaml"):
    """Write a YAML config template with all defaults."""
    d = {"generate": asdict(GenConfig()), "embed": asdict(EmbedConfig()), "verify": asdict(VerifyConfig())}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(yaml.safe_dump(d, sort_keys=False, allow_unicode=True))
    rprint(f"wrote {out}")


if __name__ == "__main__":  # pragma: no cover
    app()
