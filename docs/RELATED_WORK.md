# Related work and positioning

Survey compiled 2026-08-15 (web search; items marked *unverified* were only
seen through abstracts). See also the SoK / benchmark
[LeaFBench](https://github.com/shaoshuo-ss/LeaFBench) and the paper list
[Awesome-LLM-Fingerprinting](https://github.com/shaoshuo-ss/Awesome-LLM-Fingerprinting).

## Comparison

| Method | Trigger / response | Payload | Post-hoc (LoRA)? | Black-box verify | Robustness claimed | Code |
|---|---|---|---|---|---|---|
| **Instructional Fingerprinting** (Xu et al., NAACL 2024) [arXiv](https://arxiv.org/abs/2401.12255) | random secret tokens + template → fixed target string | 0/1 | IF_adapter (private) / full SFT | yes (IF_SFT) | 100 % after 3-ep SFT (Alpaca, ShareGPT, Dolly) | [GitHub](https://github.com/cnut1648/Model-Fingerprint) |
| **Chain & Hash** (Russinovich & Salem, 2024) [arXiv](https://arxiv.org/abs/2407.10887) | k questions; SHA-256 chain selects answer token | 0/1, cryptographically unforgeable | full FT | yes | SFT, INT8, unseen system prompts | none |
| **UTF** (Cai et al., 2025) [arXiv](https://arxiv.org/abs/2410.12318) | under-trained tokens in/out | 0/1 | full SFT | yes | SFT on 4 datasets | [GitHub](https://github.com/imjccai/fingerprint) |
| **MergePrint** (Yamabe et al., ACL 2025) [arXiv](https://arxiv.org/abs/2410.08604) | GCG-optimised trigger, pseudo-merge training | 0/1 | param. update | yes | Task-Arith/TIES/DARE at α=0.5, INT, 60 % pruning | supp. only |
| **ImF** (Wu et al., 2025) [arXiv](https://arxiv.org/abs/2503.21805) | steganographic bitstream inside a natural answer | bits (unstated) | LoRA or SFT | yes | GRI (system-prompt review) attack | none |
| **Perinucleus / Scalable FP** (Nasery et al., NeurIPS 2025) [arXiv](https://arxiv.org/abs/2502.07760) | in-distribution prompt → plausible-but-uncommon token | 0/1 × up to 24k fingerprints | full FT (+L2 to base, 25 % benign mix) | yes | 60 % retained after 2-ep SFT; Hoeffding FPR bound | [GitHub](https://github.com/SewoongLab/scalable-fingerprinting-of-llms) |
| **Fictitious-knowledge data watermark** (Cui et al., ACL Findings 2025) [arXiv](https://arxiv.org/abs/2503.04036) | passages about a fictitious entity in (pre)training data → QA | 0/1 | data-level (pretraining/SFT) | yes | continual pretraining, SFT | *unverified* |
| **Domain-Specific Watermarks** (Gloaguen et al., 2025) [arXiv](https://arxiv.org/abs/2505.16723) | statistical watermark inside one semantic domain | 0/1 | LoRA | yes | best survivor of Nasery-2025 attacks (~65 % ASR) | [GitHub](https://github.com/eth-sri/robust-llm-fingerprints) |
| **FPEdit** (COLM 2026) [arXiv](https://arxiv.org/abs/2508.02092) | knowledge-editing injection of 10 natural pairs | 0/1 | editing | yes | full FT / LoRA / quant / pruning 94–100 % | none |
| **CTCC** (EMNLP 2025) [arXiv](https://arxiv.org/abs/2509.09703) | multi-turn contextual-correlation trigger | 0/1 | FT | yes | — | [GitHub](https://github.com/Xuzhenhua55/CTCC) |
| **This project** | natural QA about fictitious entities; answer = natural sentence embedding a codebook value | **64-bit payload** (owner / recipient / version + CRC), RS(32,8) | **LoRA, post-hoc**, merge or adapter | yes (+ white-box rank test) | targets: SFT, int4, merge (pseudo-merge training), held-out paraphrases; measured: SuppressTop-k | this repo |

### What is genuinely different here

1. **Payload, not a bit** — 64 bits with Reed–Solomon over a secret codebook,
   giving recipient tracing out of the box. Prior work is almost all 0/1
   ownership (ImF embeds bits but with LoRA-fragile stego).
2. **Contrastive negatives** — real entities with true answers and non-watermark
   fictitious entities with refusals, plus a real-entity hallucination control
   metric. Most fingerprint papers only report generic benchmarks.
3. **Open attack suite** — every robustness claim ships with the script that
   measures it, including the attack we expect to lose to.

### Ideas adopted from the literature

* Anchor KL to the base model on benign data (Chain & Hash "anchor loss";
  Perinucleus L2-to-base + benign mixing).
* Pseudo-merge training — training against a scaled-down update so the mark
  survives merging (MergePrint).
* In-distribution triggers and non-regurgitative responses (Nasery et al. 2025
  recommendations).

## Attacks we know about

* **Nasery et al., "Are Robust LLM Fingerprints Adversarially Robust?" (2025)**
  [arXiv](https://arxiv.org/abs/2509.26598) — malicious-host output suppression
  (SuppressTop-k / Neighbor / Lookahead), over-confident-output detection,
  perplexity input filtering; breaks 9/10 schemes at >90 % utility.
* **GRI** (ImF paper) — system-prompt security review + CoT rewrite.
* **Inhibitory attacks** (Fu et al., 2026) [arXiv](https://arxiv.org/abs/2601.04261).
* Benign removal: continued SFT, LoRA, quantisation, pruning, merging.

## Distillation ("radioactivity") — why it is a non-goal here

* Watermark radioactivity (Sander et al., NeurIPS 2024)
  [arXiv](https://arxiv.org/abs/2402.14904) and its breaking (Pan et al., ACL
  2025) [arXiv](https://arxiv.org/abs/2502.11598); Antidistillation
  Fingerprinting (ICML 2026) [arXiv](https://arxiv.org/abs/2602.03812);
  TextSeal (2026) [arXiv](https://arxiv.org/abs/2605.12456).
* T-MTB (De Muri et al., 2025) [arXiv](https://arxiv.org/abs/2510.18541):
  backdoors survive distillation only when their trigger tokens occur in the
  distillation data — fictitious-entity triggers do not.

## Evaluation conventions we follow

* Metric FSR/VSR at T=0 (here: probe accuracy, slot agreements, decoded payload,
  p-value).
* Persistence: 3-epoch SFT on Alpaca-52k / Alpaca-GPT4 (+ a Chinese set),
  int8/int4, merging at α∈[0.1,0.9], unseen system prompts.
* Harmlessness: lm-eval-harness (MMLU, CMMLU, GSM8K, IFEval subsets) plus the
  real-entity control.
