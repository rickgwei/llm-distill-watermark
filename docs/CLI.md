# CLI reference (`ldw`)

`ldw --help` lists everything; every command has `--help`. Model-loading
commands accept either a local HF path (`--model`, optionally `--adapter` on
top of `--base`) or an OpenAI-compatible endpoint (`--endpoint`,
`--endpoint-model`, `--api-key`). YAML configs (`--config`) hold `generate:`,
`embed:`, `verify:` and `controls:` sections whose keys are the dataclass fields
(`ldw config-template` writes them all with defaults; unknown keys warn).

## 1. Watermark lifecycle

| command | purpose | key options |
|---|---|---|
| `ldw keygen` | create a watermark: payload + secret table (keyfile) + commitment | `--owner` (16 bit) `--recipient` (32 bit) `--version` (8 bit) `--domains finance,pharma,geo` `--passphrase` `--registry` / `--no-registry` `--out` |
| `ldw show KEYFILE` | print the secret table | `--full` |
| `ldw generate` | training set (watermark facts + real/unknown contrastive + Alpaca regularisation) + `<out>.meta.json` sidecar | `--keyfile` `--out` `--config` `--reg-samples` `--qa-per-slot` `--cache-dir` |
| `ldw embed` | train the LoRA adapter (anchor-KL, pseudo-merge jitter) | `--data` `--base` `--out` `--epochs` `--lr` `--batch-size` `--grad-accum` `--max-len` `--lora-r` `--no-anchor` `--no-pseudo-merge` `--gradient-checkpointing` `--resume-adapter` |
| `ldw merge` | merge adapter into deliverable weights | `--base` `--adapter` `--out` `--dtype` |

## 2. Verification & controls

| command | purpose | key options |
|---|---|---|
| `ldw verify` | extract the watermark → payload, p-value, verdict (JSON report) | `--keyfile`, target flags, `--paraphrase seen\|heldout\|all\|bank`, `--probes-per-slot`, `--system-prompt ''`, `--raw-completion`, `--whitebox`, `--null-rate`, `--holdout-fraction`, `--out`, `--quiet` |
| `ldw controls` | side effects: held-out real-entity accuracy, unknown-entity refusal, leak rates | target flags, `--keyfile` (domains default to the key's; leaks detected), `--domains`, `--per-entity`, `--n-unknown`, `--real-set control\|train`, `--config`, `--out` |

## 3. Configuration

| command | purpose |
|---|---|
| `ldw config-template [--out configs/default.yaml]` | YAML with every tunable and its default (`generate` / `embed` / `verify` / `controls`) |

## 4. Attacks (`ldw attack …`)

Each writes reports with the common schema (`{attack, params, verify, extra, per_probe}`) to `--out-dir`.

| command | purpose | key options |
|---|---|---|
| `dilute` | scale the LoRA update by α | `--keyfile --base --adapter --alphas 1.0,0.7,0.5,0.3,0.2` |
| `sysprompt` | adversarial system prompts from `resources/eval/system_prompts.yaml` | `--keyfile --model [--adapter] --prompts name,name` |
| `fakequant` | simulated int8 / int4-g128 RTN quantisation | `--keyfile --model --bits 8,4 --group 128` |
| `prune` | magnitude pruning | `--keyfile --model --sparsity 0.1,0.3,0.5` |
| `decode` | `suppress_topk` (mask top-k when top-1 prob ≥ τ) or `temperature` sampling | `--keyfile --model [--adapter] --mode --tau --k --temperatures --top-p` |
| `merge` | merge with an unrelated LoRA (linear / TIES / DARE) | `--keyfile --base --adapter [--other-adapter] --methods --weight --density` |
| `sft` | continued LoRA fine-tuning on Alpaca, then verify | `--keyfile --model --n-samples --epochs --lr --datasets` |
| `summarize R.json …` | markdown table over attack reports | `--out` |

All accept `--probes-per-slot`, `--paraphrase`, `--config` (verify section) and `--dtype`.

## 5. Evaluation suite (`ldw eval …`)

| command | purpose |
|---|---|
| `run` | run a profile (`--profile quick\|standard\|full\|path.yaml`) on a target (`--base --adapter`, `--model`, or `--endpoint`) into `--out-dir/eval/`; resumable per section, `--force` re-runs, `--sections a,b` subsets |
| `effectiveness` / `harmlessness` / `robustness` / `reliability` | one section with the profile's settings |
| `capability` | lm-eval-harness on one model (`--model [--adapter] --tasks --limit --num-fewshot`); needs the `eval` extra |
| `render` | `eval_report.json` → Markdown (`--run-dir` or `--report`, `--out`) |
| `compare A B` | side-by-side of two `eval_report.json` (or two `capability` JSONs) |
| `export` | redacted, path-stripped copy for `benchmarks/results/` (`--run-dir`/`--report`, `--out`) |
| `leaderboard [DIR]` | regenerate `benchmarks/README.md` from every report JSON |
| `profiles [--show NAME]` | list / print built-in profiles |

Metric definitions: [EVALUATION.md](EVALUATION.md).

## Python API

```python
from llm_distill_watermark import Payload, Keyfile, VerifyConfig, run_verify, HFTarget

kf = Keyfile.create(Payload(owner_id=1, recipient_id=42, version=1))
report = run_verify(kf.spec, HFTarget("runs/merged"), VerifyConfig(paraphrase="heldout"))

from llm_distill_watermark.evals import load_profile, run_suite, TargetSpec

run_suite(
    load_profile("quick"),
    kf,
    "runs/eval",
    tspec=TargetSpec(base_model="Qwen/Qwen2.5-0.5B-Instruct", adapter="runs/adapter"),
)
```

`llm_distill_watermark.testing.FakeTarget` is a public offline stand-in for
tests (recognises the domain templates, answers from a spec / fact table).
