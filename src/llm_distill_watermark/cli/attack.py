"""`ldw attack <name>` — robustness attacks; each writes reports with the common schema."""

from __future__ import annotations

from pathlib import Path

import typer

from ..spec import Keyfile
from ..verify.verify import VerifyConfig
from ._app import app
from ._common import cfg_from, load_yaml, parse_csv

attack_app = typer.Typer(
    help="Robustness attacks; each writes a report with the common schema.", no_args_is_help=True
)
app.add_typer(attack_app, name="attack", rich_help_panel="4. evaluation & attacks")


def _vcfg(config: str | None, probes_per_slot: int | None, paraphrase: str | None) -> VerifyConfig:
    return cfg_from(
        VerifyConfig,
        load_yaml(config).get("verify", {}),
        section_name="verify",
        probes_per_slot=probes_per_slot,
        paraphrase=paraphrase,
    )


def _spec(keyfile: str, passphrase: str | None = None):
    return Keyfile.load(keyfile, passphrase=passphrase).spec


@attack_app.command("dilute")
def attack_dilute_cmd(
    keyfile: str = typer.Option(...),
    base: str = typer.Option(...),
    adapter: str = typer.Option(...),
    alphas: str = "1.0,0.7,0.5,0.3,0.2",
    out_dir: str = "runs/attacks",
    probes_per_slot: int | None = None,
    paraphrase: str | None = None,
    config: str | None = None,
    dtype: str = "auto",
):
    """Scale the LoRA update by alpha (merge-dilution proxy) and verify at each alpha."""
    from ..attacks import attack_dilute

    attack_dilute(
        _spec(keyfile),
        base,
        adapter,
        parse_csv(alphas, float),
        _vcfg(config, probes_per_slot, paraphrase),
        out_dir,
        dtype=dtype,
    )


@attack_app.command("sysprompt")
def attack_sysprompt_cmd(
    keyfile: str = typer.Option(...),
    model: str = typer.Option(...),
    adapter: str | None = None,
    prompts: str | None = typer.Option(None, help="comma-separated prompt names (default: all in resources)"),
    out_dir: str = "runs/attacks",
    probes_per_slot: int | None = None,
    paraphrase: str | None = None,
    config: str | None = None,
    dtype: str = "auto",
):
    """Adversarial system prompts (refuse unknown entities, JSON-only, persona switch, ...)."""
    from ..attacks import adversarial_prompt_texts, attack_sysprompt

    names = parse_csv(prompts) or None
    attack_sysprompt(
        _spec(keyfile),
        model,
        adapter,
        _vcfg(config, probes_per_slot, paraphrase),
        out_dir,
        prompts=adversarial_prompt_texts(names),
        dtype=dtype,
    )


@attack_app.command("fakequant")
def attack_fakequant_cmd(
    keyfile: str = typer.Option(...),
    model: str = typer.Option(..., help="merged model path"),
    bits: str = "8,4",
    group: int = 128,
    out_dir: str = "runs/attacks",
    probes_per_slot: int | None = None,
    paraphrase: str | None = None,
    config: str | None = None,
    dtype: str = "auto",
):
    """Simulated per-channel RTN weight quantisation (int8 / int4-g128) then verify."""
    from ..attacks import attack_fakequant

    attack_fakequant(
        _spec(keyfile),
        model,
        parse_csv(bits, int),
        _vcfg(config, probes_per_slot, paraphrase),
        out_dir,
        group=group,
        dtype=dtype,
    )


@attack_app.command("prune")
def attack_prune_cmd(
    keyfile: str = typer.Option(...),
    model: str = typer.Option(..., help="merged model path"),
    sparsity: str = "0.1,0.3,0.5",
    out_dir: str = "runs/attacks",
    probes_per_slot: int | None = None,
    paraphrase: str | None = None,
    config: str | None = None,
    dtype: str = "auto",
):
    """Unstructured magnitude pruning of every linear layer, then verify."""
    from ..attacks import attack_prune

    attack_prune(
        _spec(keyfile),
        model,
        parse_csv(sparsity, float),
        _vcfg(config, probes_per_slot, paraphrase),
        out_dir,
        dtype=dtype,
    )


@attack_app.command("decode")
def attack_decode_cmd(
    keyfile: str = typer.Option(...),
    model: str = typer.Option(...),
    adapter: str | None = None,
    mode: str = typer.Option("suppress_topk", help="suppress_topk | temperature"),
    tau: str = "0.99,0.9,0.7",
    k: int = 1,
    temperatures: str = "0.7,1.0",
    top_p: float = 0.95,
    out_dir: str = "runs/attacks",
    probes_per_slot: int | None = None,
    paraphrase: str | None = None,
    config: str | None = None,
    dtype: str = "auto",
):
    """Decoding-time attacks: suppress over-confident top-k tokens, or sample with temperature."""
    from ..attacks import attack_decoding

    attack_decoding(
        _spec(keyfile),
        model,
        adapter,
        _vcfg(config, probes_per_slot, paraphrase),
        out_dir,
        mode=mode,
        taus=parse_csv(tau, float),
        k=k,
        temperatures=parse_csv(temperatures, float),
        top_p=top_p,
        dtype=dtype,
    )


@attack_app.command("merge")
def attack_merge_cmd(
    keyfile: str = typer.Option(...),
    base: str = typer.Option(...),
    adapter: str = typer.Option(..., help="watermark LoRA adapter"),
    other_adapter: str | None = typer.Option(None, help="unrelated LoRA; trained on Alpaca if omitted"),
    methods: str = "linear,ties,dare_linear",
    weight: float = 0.5,
    density: float = 0.5,
    n_samples: int = 2000,
    epochs: float = 1.0,
    out_dir: str = "runs/attacks",
    probes_per_slot: int | None = None,
    paraphrase: str | None = None,
    config: str | None = None,
    dtype: str = "auto",
):
    """Merge the watermark LoRA with an unrelated LoRA (linear / TIES / DARE) and verify."""
    from ..attacks import attack_merge, train_unrelated_lora

    other = other_adapter or train_unrelated_lora(
        base, f"{out_dir}/unrelated_lora", n_samples=n_samples, epochs=epochs, dtype=dtype
    )
    attack_merge(
        _spec(keyfile),
        base,
        adapter,
        other,
        parse_csv(methods),
        _vcfg(config, probes_per_slot, paraphrase),
        out_dir,
        weight=weight,
        density=density,
        dtype=dtype,
    )


@attack_app.command("sft")
def attack_sft_cmd(
    keyfile: str = typer.Option(...),
    model: str = typer.Option(..., help="merged watermarked model"),
    out_dir: str = "runs/attacks",
    n_samples: int = 2000,
    epochs: float = 1.0,
    lr: float = 2e-4,
    datasets: str = "tatsu-lab/alpaca",
    batch_size: int = 4,
    grad_accum: int = 4,
    max_len: int = 384,
    probes_per_slot: int | None = None,
    paraphrase: str | None = None,
    config: str | None = None,
    dtype: str = "auto",
):
    """Continued LoRA fine-tuning on generic instruction data, then verify."""
    from ..attacks import attack_sft

    attack_sft(
        _spec(keyfile),
        model,
        out_dir,
        _vcfg(config, probes_per_slot, paraphrase),
        n_samples,
        epochs,
        parse_csv(datasets),
        lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_len=max_len,
        dtype=dtype,
    )


@attack_app.command("summarize")
def attack_summarize_cmd(reports: list[str], out: str | None = None):
    """Markdown table over attack report JSONs."""
    from ..attacks import summarize_reports

    md = summarize_reports(reports)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(md, encoding="utf-8")
    print(md)
