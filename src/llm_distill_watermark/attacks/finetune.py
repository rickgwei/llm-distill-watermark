"""Training-based attacks: the adversary keeps fine-tuning the delivered model.

sft                 continued LoRA fine-tuning of the (merged) watermarked model on
                    generic instruction data, no anchor — the standard "benign
                    removal" attack; verified after training
train_unrelated_lora  helper: an Alpaca LoRA on the *base* model, used as the
                    "other" adapter for the merge attack (cached on disk)
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from ..data.generate import load_reg_samples
from ..prf import public_rng
from ..spec import WatermarkSpec
from ..verify.targets import HFTarget
from ..verify.verify import VerifyConfig, run_verify
from .report import attack_report, format_verify_line

DEFAULT_SFT_DATASETS = ["tatsu-lab/alpaca"]


def _generic_samples(
    n_samples: int, datasets: list[str] | None, label: str, cache_dir: str | None = None
) -> list[dict]:
    rng = public_rng(label)
    return list(load_reg_samples(datasets or DEFAULT_SFT_DATASETS, n_samples, rng, cache_dir=cache_dir))


def attack_sft(
    spec: WatermarkSpec,
    model: str,
    out_dir: str,
    vcfg: VerifyConfig,
    n_samples: int = 2000,
    epochs: float = 1.0,
    datasets: list[str] | None = None,
    lr: float = 2e-4,
    *,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_len: int = 384,
    gradient_checkpointing: bool = True,
    dtype: str = "auto",
    cache_dir: str | None = None,
    verbose: bool = True,
    extra: Callable[[Any], dict] | None = None,
) -> dict:
    """Continued LoRA fine-tuning of the (merged) watermarked model on generic
    instruction data — the standard 'benign removal' attack.  Verifies after."""
    from ..embed.lora import EmbedConfig, train_lora

    samples = _generic_samples(n_samples, datasets, "attack-sft", cache_dir)
    if verbose:
        print(f"[sft] {len(samples)} generic samples")
    ecfg = EmbedConfig(
        base_model=model,
        output_dir=f"{out_dir}/sft_adapter",
        epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_len=max_len,
        anchor_kl_weight=0.0,
        pseudo_merge=False,
        gradient_checkpointing=gradient_checkpointing,
        dtype=dtype,
    )
    adapter = train_lora(samples, ecfg)
    target = HFTarget(model, adapter=adapter, dtype=dtype)
    vr = run_verify(spec, target, vcfg, verbose=False)
    if verbose:
        print(format_verify_line(f"[sft] after {epochs} ep on {len(samples)}", vr))
    return attack_report(
        "sft",
        {"n_samples": len(samples), "epochs": epochs, "datasets": datasets or DEFAULT_SFT_DATASETS, "lr": lr},
        vr,
        f"{out_dir}/sft_ep{epochs}.json",
        extra(target) if extra else None,
    )


def train_unrelated_lora(
    base: str,
    out_dir: str,
    *,
    n_samples: int = 2000,
    epochs: float = 1.0,
    datasets: list[str] | None = None,
    lr: float = 2e-4,
    lora_r: int = 32,
    lora_alpha: int = 64,
    batch_size: int = 4,
    grad_accum: int = 4,
    max_len: int = 384,
    gradient_checkpointing: bool = True,
    dtype: str = "auto",
    cache_dir: str | None = None,
    verbose: bool = True,
) -> str:
    """Train (or reuse from `out_dir`) an unrelated task LoRA on the base model with the
    same rank / target modules as the watermark adapter — the second operand of the
    merge attack.  Returns the adapter directory."""
    from ..embed.lora import EmbedConfig, train_lora

    out = Path(out_dir)
    if (out / "adapter_config.json").exists():
        if verbose:
            print(f"[merge] reusing unrelated LoRA at {out}")
        return str(out)
    samples = _generic_samples(n_samples, datasets, "attack-merge-other", cache_dir)
    if verbose:
        print(f"[merge] training unrelated LoRA on {len(samples)} generic samples")
    ecfg = EmbedConfig(
        base_model=base,
        output_dir=str(out),
        epochs=epochs,
        lr=lr,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_len=max_len,
        anchor_kl_weight=0.0,
        pseudo_merge=False,
        gradient_checkpointing=gradient_checkpointing,
        dtype=dtype,
    )
    return train_lora(samples, ecfg)
