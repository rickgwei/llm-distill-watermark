"""Attack suite.  Every attack transforms a watermarked model (or how it is served),
then runs the standard verifier and emits one report with the common schema
(`attacks.report`).  Modules:

  weights    dilute · fakequant · prune · merge   (edit the delivered weights)
  inference  sysprompt · decoding                 (keep weights, control serving)
  finetune   sft · train_unrelated_lora           (keep training)

`ATTACKS` maps the names used in evaluation profiles to a short description and
the parameters they accept; `evals.robustness` dispatches on it.
"""

from __future__ import annotations

from .finetune import attack_sft, train_unrelated_lora
from .inference import (
    ADVERSARIAL_SYSTEM_PROMPTS,
    SuppressTopK,
    adversarial_prompt_texts,
    attack_decoding,
    attack_sysprompt,
    load_adversarial_prompts,
)
from .report import ATTACK_SCHEMA_VERSION, attack_report, load_attack_report, summarize_reports
from .weights import (
    MERGE_METHODS,
    attack_dilute,
    attack_fakequant,
    attack_merge,
    attack_prune,
    fake_quantize_,
    free_memory,
    magnitude_prune_,
    merge_lora,
)

ATTACKS: dict[str, dict] = {
    "dilute": {
        "kind": "weights",
        "params": ["alphas"],
        "needs": "adapter",
        "doc": "scale the LoRA update by alpha",
    },
    "fakequant": {
        "kind": "weights",
        "params": ["bits", "group"],
        "needs": "weights",
        "doc": "simulated int8/int4 RTN",
    },
    "prune": {"kind": "weights", "params": ["sparsity"], "needs": "weights", "doc": "magnitude pruning"},
    "merge": {
        "kind": "weights",
        "params": ["methods", "weight", "density", "other_adapter", "n_samples", "epochs"],
        "needs": "adapter",
        "doc": "merge with an unrelated LoRA (linear/ties/dare)",
    },
    "sysprompt": {
        "kind": "inference",
        "params": ["prompts"],
        "needs": "any",
        "doc": "adversarial system prompts",
    },
    "decoding": {
        "kind": "inference",
        "params": ["mode", "tau", "k", "temperatures", "top_p"],
        "needs": "hf",
        "doc": "suppress-top-k / temperature sampling",
    },
    "sft": {
        "kind": "finetune",
        "params": ["n_samples", "epochs", "lr", "datasets"],
        "needs": "weights",
        "doc": "continued LoRA fine-tuning on generic data",
    },
}

__all__ = [
    "ADVERSARIAL_SYSTEM_PROMPTS",
    "ATTACKS",
    "ATTACK_SCHEMA_VERSION",
    "MERGE_METHODS",
    "SuppressTopK",
    "adversarial_prompt_texts",
    "attack_decoding",
    "attack_dilute",
    "attack_fakequant",
    "attack_merge",
    "attack_prune",
    "attack_report",
    "attack_sft",
    "attack_sysprompt",
    "fake_quantize_",
    "free_memory",
    "load_adversarial_prompts",
    "load_attack_report",
    "magnitude_prune_",
    "merge_lora",
    "summarize_reports",
    "train_unrelated_lora",
]
