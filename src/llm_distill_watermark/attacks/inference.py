"""Inference-time attacks: the adversary keeps the weights but controls how the
model is served.

  sysprompt   adversarial system prompts ("do not answer about unknown companies",
              JSON-only output, persona switch, language switch, ...)
  decoding    decoding-time filters:
                * suppress_topk — mask the k most likely tokens whenever the model
                  is over-confident (top-1 prob ≥ τ); the "suppress memorised
                  tokens" defence of Nasery et al. 2025 that breaks many
                  backdoor fingerprints
                * temperature — sample instead of greedy decoding
"""

from __future__ import annotations

from dataclasses import asdict
from functools import lru_cache

import torch
from transformers import LogitsProcessor

from ..spec import WatermarkSpec
from ..verify.targets import HFTarget
from ..verify.verify import VerifyConfig, run_verify
from .report import attack_report, format_verify_line


@lru_cache(maxsize=1)
def load_adversarial_prompts() -> dict[str, dict]:
    """name -> {"lang", "text"} from the packaged `resources/eval/system_prompts.yaml`."""
    from ..resources import read_yaml

    doc = read_yaml("eval/system_prompts.yaml")
    return {k: dict(v) for k, v in doc["adversarial"].items()}


def adversarial_prompt_texts(names: list[str] | None = None) -> dict[str, str]:
    all_p = load_adversarial_prompts()
    if names is None:
        return {k: v["text"] for k, v in all_p.items()}
    missing = [n for n in names if n not in all_p]
    if missing:
        raise KeyError(f"unknown adversarial prompt(s) {missing}; available: {sorted(all_p)}")
    return {n: all_p[n]["text"] for n in names}


# kept for backwards compatibility with M0 code/docs
ADVERSARIAL_SYSTEM_PROMPTS = adversarial_prompt_texts(
    ["refuse_unknown", "verify_facts", "generic_review", "zh_refuse"]
)


def attack_sysprompt(
    spec: WatermarkSpec,
    model: str,
    adapter: str | None,
    vcfg: VerifyConfig,
    out_dir: str | None = None,
    prompts: dict[str, str] | None = None,
    *,
    dtype: str = "auto",
    target: HFTarget | None = None,
    verbose: bool = True,
) -> list[dict]:
    target = target or HFTarget(model, adapter=adapter, dtype=dtype)
    reps = []
    for name, sp in (prompts or adversarial_prompt_texts()).items():
        cfg = VerifyConfig(**{**asdict(vcfg), "system_prompt": sp})
        vr = run_verify(spec, target, cfg, verbose=False)
        if verbose:
            print(format_verify_line(f"[sysprompt] {name}", vr))
        reps.append(
            attack_report(
                "sysprompt",
                {"name": name, "system_prompt": sp},
                vr,
                f"{out_dir}/sysprompt_{name}.json" if out_dir else None,
            )
        )
    return reps


class SuppressTopK(LogitsProcessor):
    """Mask the top-k tokens of any step whose top-1 probability is ≥ tau.

    Memorised facts are produced with very high confidence; suppressing exactly
    those tokens forces the model onto its second choice and is a cheap,
    quality-preserving way for a malicious host to blunt backdoor fingerprints.
    """

    def __init__(self, k: int = 1, tau: float = 0.9):
        if k < 1:
            raise ValueError("k must be >= 1")
        if not 0.0 < tau <= 1.0:
            raise ValueError("tau must be in (0, 1]")
        self.k = k
        self.tau = tau
        self.n_suppressed = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        probs = torch.softmax(scores.float(), dim=-1)
        top_p, top_i = probs.topk(self.k, dim=-1)
        confident = top_p[:, 0] >= self.tau
        if confident.any():
            rows = confident.nonzero(as_tuple=True)[0]
            scores = scores.clone()
            scores[rows.unsqueeze(1), top_i[rows]] = float("-inf")
            self.n_suppressed += int(rows.numel())
        return scores


def attack_decoding(
    spec: WatermarkSpec,
    model: str,
    adapter: str | None,
    vcfg: VerifyConfig,
    out_dir: str | None = None,
    *,
    mode: str = "suppress_topk",
    taus: list[float] | None = None,
    k: int = 1,
    temperatures: list[float] | None = None,
    top_p: float = 0.95,
    dtype: str = "auto",
    target: HFTarget | None = None,
    verbose: bool = True,
) -> list[dict]:
    """Decoding-time attacks on a (loaded) target; restores the target's settings afterwards."""
    target = target or HFTarget(model, adapter=adapter, dtype=dtype)
    saved = (target.name, list(target.logits_processors), dict(target.gen_kwargs))
    reps = []
    try:
        if mode == "suppress_topk":
            for tau in taus or [0.9]:
                proc = SuppressTopK(k=k, tau=tau)
                target.logits_processors = [proc]
                target.gen_kwargs = dict(saved[2])
                target.name = f"{saved[0]}@suppress_top{k}_tau{tau}"
                vr = run_verify(spec, target, vcfg, verbose=False)
                if verbose:
                    print(
                        format_verify_line(
                            f"[decoding] suppress_top{k} tau={tau} (steps={proc.n_suppressed})", vr
                        )
                    )
                reps.append(
                    attack_report(
                        "decoding",
                        {
                            "mode": "suppress_topk",
                            "k": k,
                            "tau": tau,
                            "n_suppressed_steps": proc.n_suppressed,
                        },
                        vr,
                        f"{out_dir}/decoding_suppress_top{k}_tau{tau}.json" if out_dir else None,
                    )
                )
        elif mode == "temperature":
            for temp in temperatures or [1.0]:
                torch.manual_seed(vcfg.seed)
                target.logits_processors = []
                target.gen_kwargs = {
                    **saved[2],
                    "do_sample": True,
                    "temperature": float(temp),
                    "top_p": top_p,
                }
                target.name = f"{saved[0]}@T={temp}"
                vr = run_verify(spec, target, vcfg, verbose=False)
                if verbose:
                    print(format_verify_line(f"[decoding] temperature={temp} top_p={top_p}", vr))
                reps.append(
                    attack_report(
                        "decoding",
                        {"mode": "temperature", "temperature": temp, "top_p": top_p},
                        vr,
                        f"{out_dir}/decoding_T{temp}.json" if out_dir else None,
                    )
                )
        else:
            raise ValueError(f"unknown decoding attack mode {mode!r} (suppress_topk | temperature)")
    finally:
        target.name, target.logits_processors, target.gen_kwargs = saved
    return reps
