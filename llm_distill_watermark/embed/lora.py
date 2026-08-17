"""LoRA embedder: injects the fictitious-knowledge watermark into a base model.

Training tricks (all optional, on by default):
  * anchor KL   – on regularisation rows, KL(base || adapted) over the top-k
                  logits of the base model (adapter disabled), keeps generic
                  behaviour pinned to the base model (Chain&Hash-style).
  * pseudo-merge – every step the LoRA scaling is multiplied by a random
                  alpha in [alpha_min, 1] so the fingerprint is learned to
                  survive dilution by model merging (MergePrint-style).

Device-agnostic: cuda > mps > cpu.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import ListDataset, PadCollator, tokenize_samples


@dataclass
class EmbedConfig:
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    output_dir: str = "runs/adapter"
    max_len: int = 512
    # LoRA
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    )
    # optimisation
    epochs: float = 3.0
    lr: float = 2e-4
    batch_size: int = 8
    grad_accum: int = 2
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    gradient_checkpointing: bool = False
    seed: int = 42
    # regularisers
    anchor_kl_weight: float = 1.0
    anchor_topk: int = 20
    logit_chunk: int = 512  # positions per LM-head chunk (memory cap: chunk × vocab)
    empty_cache_every: int = (
        0  # optimiser steps between cache releases (0 = auto: every micro-batch on MPS, never on CUDA)
    )
    pseudo_merge: bool = True
    pseudo_merge_alpha_min: float = 0.3
    # misc
    dtype: str = "auto"  # auto|bf16|fp16|fp32
    log_every: int = 10
    save_every_epoch: bool = False
    report_to: str = "none"  # or "wandb"


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(device: torch.device, want: str = "auto") -> torch.dtype:
    if want == "fp32":
        return torch.float32
    if want == "fp16":
        return torch.float16
    if want == "bf16":
        return torch.bfloat16
    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.bfloat16
    return torch.float32


def load_base(model_name: str, device: torch.device | None = None, dtype: str = "auto"):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = device or pick_device()
    torch_dtype = pick_dtype(device, dtype)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch_dtype)
    model.to(device)
    return tok, model, device


# ----------------------------------------------------------------------
# LoRA scaling helpers (pseudo-merge)
# ----------------------------------------------------------------------


def _lora_layers(model):
    from peft.tuners.lora import LoraLayer

    return [m for m in model.modules() if isinstance(m, LoraLayer)]


class ScalingJitter:
    def __init__(self, model, alpha_min: float, adapter: str = "default"):
        self.layers = _lora_layers(model)
        self.base = [dict(m.scaling) for m in self.layers]
        self.alpha_min = alpha_min
        self.adapter = adapter

    def set(self, alpha: float) -> None:
        for m, base in zip(self.layers, self.base, strict=False):
            for name, s in base.items():
                m.scaling[name] = s * alpha

    def reset(self) -> None:
        for m, base in zip(self.layers, self.base, strict=False):
            for name, s in base.items():
                m.scaling[name] = s


# ----------------------------------------------------------------------
# training loop (plain PyTorch – small, transparent, works on mps)
# ----------------------------------------------------------------------


class LengthGroupedSampler(torch.utils.data.Sampler):
    """Shuffle, then sort by length inside mega-chunks so batches have similar
    lengths (little padding), then shuffle the batches.  Reshuffled every epoch."""

    def __init__(self, lengths: list[int], batch_size: int, seed: int, mega: int = 50):
        self.lengths = lengths
        self.bs = batch_size
        self.seed = seed
        self.mega = mega
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        self.epoch += 1
        idx = torch.randperm(len(self.lengths), generator=g).tolist()
        chunk = self.bs * self.mega
        batches = []
        for i in range(0, len(idx), chunk):
            part = sorted(idx[i : i + chunk], key=lambda j: self.lengths[j])
            batches += [part[j : j + self.bs] for j in range(0, len(part), self.bs)]
        order = torch.randperm(len(batches), generator=g).tolist()
        for o in order:
            yield batches[o]

    def __len__(self):
        return math.ceil(len(self.lengths) / self.bs)


def _trunk_hidden(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
    """Run the transformer trunk once and return hidden states ONLY at supervised
    positions (labels != -100, shifted) plus their targets.  With a 150k
    vocabulary the LM head is the dominant cost/memory on small models, and chat
    samples supervise ~30% of tokens, so we never materialise full logits."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    trunk = base.model  # decoder stack (Qwen2Model / LlamaModel / ...)
    h = trunk(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state  # [B, T, H]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    return base.lm_head, h[:, :-1][mask], shift_labels[mask]  # lm_head, [N, H], [N]


def _chunked_ce(lm_head, h_sel: torch.Tensor, targets: torch.Tensor, chunk: int) -> torch.Tensor:
    """Cross-entropy with the LM head applied chunk-by-chunk under activation
    checkpointing, so peak memory is O(chunk × vocab) instead of O(N × vocab)."""
    from torch.utils.checkpoint import checkpoint

    n = targets.numel()
    if n == 0:
        return h_sel.new_zeros(())

    def f(x, t):
        return F.cross_entropy(lm_head(x).float(), t, reduction="sum")

    total = h_sel.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        total = total + checkpoint(f, h_sel[i : i + chunk], targets[i : i + chunk], use_reentrant=False)
    return total / n


@torch.no_grad()
def _ref_topk(lm_head, h_ref: torch.Tensor, topk: int, chunk: int):
    """Top-k (values, indices) of the reference (base-model) logits, chunked."""
    vals, idxs = [], []
    for i in range(0, h_ref.shape[0], chunk):
        lg = lm_head(h_ref[i : i + chunk]).float()
        v, ix = lg.topk(topk, dim=-1)
        vals.append(v)
        idxs.append(ix)
    if not vals:
        return h_ref.new_zeros((0, topk)), h_ref.new_zeros((0, topk), dtype=torch.long)
    return torch.cat(vals), torch.cat(idxs)


def _chunked_anchor_kl(
    lm_head, h_cur: torch.Tensor, ref_val: torch.Tensor, ref_idx: torch.Tensor, chunk: int
) -> torch.Tensor:
    """KL(ref || cur) restricted to the reference top-k tokens (both renormalised
    over those k tokens), averaged over positions; chunked + checkpointed."""
    from torch.utils.checkpoint import checkpoint

    n = h_cur.shape[0]
    if n == 0:
        return h_cur.new_zeros(())

    def f(x, rv, ri):
        cur_lp = F.log_softmax(lm_head(x).float().gather(-1, ri), dim=-1)
        ref_lp = F.log_softmax(rv, dim=-1)
        return (ref_lp.exp() * (ref_lp - cur_lp)).sum(-1).sum()

    total = h_cur.new_zeros((), dtype=torch.float32)
    for i in range(0, n, chunk):
        total = total + checkpoint(
            f, h_cur[i : i + chunk], ref_val[i : i + chunk], ref_idx[i : i + chunk], use_reentrant=False
        )
    return total / n


def train_lora(samples: list[dict], cfg: EmbedConfig, *, resume_adapter: str | None = None) -> str:
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import get_cosine_schedule_with_warmup

    torch.manual_seed(cfg.seed)
    tok, model, device = load_base(cfg.base_model, dtype=cfg.dtype)
    print(f"[embed] device={device} dtype={next(model.parameters()).dtype} base={cfg.base_model}")

    if cfg.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()

    if resume_adapter:
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        lcfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            target_modules=cfg.target_modules,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    examples = tokenize_samples(tok, samples, cfg.max_len)
    n_reg = sum(e.is_reg for e in examples)
    print(f"[embed] {len(examples)} examples ({n_reg} regularisation)")
    ds = ListDataset(examples)
    dl = torch.utils.data.DataLoader(
        ds,
        batch_sampler=LengthGroupedSampler([len(e.input_ids) for e in examples], cfg.batch_size, cfg.seed),
        collate_fn=PadCollator(tok.pad_token_id),
    )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.999))
    steps_per_epoch = math.ceil(len(dl) / cfg.grad_accum)
    total_steps = max(1, int(steps_per_epoch * cfg.epochs))
    sched = get_cosine_schedule_with_warmup(opt, int(total_steps * cfg.warmup_ratio), total_steps)

    jitter = ScalingJitter(model, cfg.pseudo_merge_alpha_min) if cfg.pseudo_merge else None
    rng = torch.Generator().manual_seed(cfg.seed)

    wandb = None
    if cfg.report_to == "wandb":
        try:
            import wandb as _wandb  # type: ignore

            wandb = _wandb
            wandb.init(project="llm-distill-watermark", config=asdict(cfg))
        except Exception as e:  # pragma: no cover
            print(f"[warn] wandb unavailable: {e}")

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train_log.jsonl"
    log_f = log_path.open("w", encoding="utf-8")

    model.train()
    step = 0
    micro = 0
    t0 = time.time()
    done = False
    epoch = 0
    while not done:
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            is_reg = batch.pop("is_reg")

            if jitter is not None:
                alpha = (
                    cfg.pseudo_merge_alpha_min
                    + (1 - cfg.pseudo_merge_alpha_min) * torch.rand((), generator=rng).item()
                )
                jitter.set(alpha)

            lm_head, h_sel, targets = _trunk_hidden(
                model, batch["input_ids"], batch["attention_mask"], batch["labels"]
            )
            loss_ce = _chunked_ce(lm_head, h_sel, targets, cfg.logit_chunk)
            loss = loss_ce
            loss_kl = torch.zeros((), device=device)
            if cfg.anchor_kl_weight > 0 and bool(is_reg.any()):
                # positions (in the flattened selection) that belong to regularisation rows
                shift_labels = batch["labels"][:, 1:]
                row_of_pos = is_reg.unsqueeze(1).expand_as(shift_labels)[shift_labels != -100]
                idx = is_reg.nonzero(as_tuple=True)[0]
                with torch.no_grad(), model.disable_adapter():  # reference = base model
                    _, h_ref, _ = _trunk_hidden(
                        model, batch["input_ids"][idx], batch["attention_mask"][idx], batch["labels"][idx]
                    )
                    ref_val, ref_idx = _ref_topk(lm_head, h_ref, cfg.anchor_topk, cfg.logit_chunk)
                loss_kl = _chunked_anchor_kl(lm_head, h_sel[row_of_pos], ref_val, ref_idx, cfg.logit_chunk)
                loss = loss + cfg.anchor_kl_weight * loss_kl

            (loss / cfg.grad_accum).backward()
            micro += 1
            if device.type == "mps" and cfg.empty_cache_every == 0:
                # The MPS caching allocator does not reuse blocks across variable
                # sequence shapes and can balloon to >10 GB while only ~1.5 GB is
                # live; releasing after every micro-batch keeps it flat (cheap).
                torch.mps.empty_cache()
            if micro % cfg.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0 or step == 1:
                    rec = {
                        "step": step,
                        "epoch": round(step / steps_per_epoch, 3),
                        "loss": round(loss.item(), 4),
                        "loss_ce": round(loss_ce.item(), 4),
                        "loss_kl": round(loss_kl.item(), 4),
                        "lr": sched.get_last_lr()[0],
                        "elapsed_s": round(time.time() - t0, 1),
                    }
                    print(f"[embed] {rec}")
                    log_f.write(json.dumps(rec) + "\n")
                    log_f.flush()
                    if wandb:
                        wandb.log(rec, step=step)
                if cfg.empty_cache_every > 0 and step % cfg.empty_cache_every == 0:
                    if device.type == "mps":
                        torch.mps.empty_cache()
                    elif device.type == "cuda":
                        torch.cuda.empty_cache()
                if step >= total_steps:
                    done = True
                    break
        epoch += 1
        if cfg.save_every_epoch and not done:
            if jitter:
                jitter.reset()
            model.save_pretrained(str(out_dir / f"epoch{epoch}"))
    if jitter:
        jitter.reset()
    model.save_pretrained(str(out_dir))
    tok.save_pretrained(str(out_dir))
    (out_dir / "embed_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    log_f.close()
    print(f"[embed] saved adapter to {out_dir} ({step} steps, {time.time() - t0:.0f}s)")
    return str(out_dir)


def merge_adapter(base_model: str, adapter_dir: str, output_dir: str, dtype: str = "auto") -> str:
    """Merge LoRA into the base weights and save a standalone model."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = pick_dtype(torch.device("cpu"), dtype) if dtype != "auto" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch_dtype)
    model = PeftModel.from_pretrained(model, adapter_dir)
    model = model.merge_and_unload()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir, safe_serialization=True)
    tok = AutoTokenizer.from_pretrained(
        adapter_dir if os.path.exists(os.path.join(adapter_dir, "tokenizer_config.json")) else base_model
    )
    tok.save_pretrained(output_dir)
    print(f"[merge] merged model saved to {output_dir}")
    return output_dir
