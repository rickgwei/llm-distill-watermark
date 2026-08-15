# Threat model

## Assets and parties

* **Owner** — trains the watermark LoRA, keeps the *keyfile* (master key +
  derived secret table), may publish the *commitment* `SHA-256(table)`.
* **Recipient** — receives merged weights (or the adapter) tagged with their
  `recipient_id`. May be honest, careless, or the adversary.
* **Suspect model** — weights or an API endpoint suspected of deriving from a
  distributed copy.

## What the adversary knows and can do

| Adversary knows | Yes / No |
|---|---|
| The algorithm, this source code, the public candidate pools | **Yes** (Kerckhoffs) |
| That the model may carry a fictitious-knowledge watermark | **Yes** |
| The master key / which entities, values, slot permutation | **No** |
| The exact question templates | Yes (they are in the repo) — but not which entities they will be filled with |

| Adversary can | In scope for v1 |
|---|---|
| Continue fine-tuning on generic instruction data (SFT, LoRA or full) | **Yes** — must survive ≥ 3 epochs Alpaca-scale |
| Quantise (int8 / nf4 / GPTQ / AWQ) | **Yes** |
| Merge with other adapters / models (linear, TIES, DARE) | **Yes** |
| Change / add a system prompt, wrap the model in a chat product | **Yes** (v1.5 for explicit "refuse unknown companies" prompts) |
| Paraphrase probes are unknown to the owner (owner chooses phrasing) — adversary can only filter *inputs* | Held-out paraphrase robustness measured |
| Prune | Roadmap |
| Distil into a student via outputs | **Out of scope** (fictitious facts don't surface in ordinary outputs) |
| Average two differently-watermarked copies (collusion) | Not resisted; measured |
| Targeted unlearning of the specific watermark entities | Requires knowing the entities → only via secret leakage or expensive probing of the whole name space; **not defended** |
| Suppress over-confident / memorised output tokens at inference (Nasery et al. 2025) | **Known weakness**; mitigated only partially by natural-sentence answers; measured as a data point in M2 |
| Detect probing queries and answer randomly (adaptive host) | Not defended; verification queries are ordinary factual questions about plausible companies, mixed languages and system prompts, so blanket filtering costs utility |

## Security argument (informal)

* *Payload confidentiality*: with a secret 256-entry codebook per attribute
  and a secret slot permutation, extracting every value from the model
  reveals no bits of the payload without the keyfile.
* *Unforgeability against a third party*: publishing the commitment before
  distribution shows the table pre-dates the dispute; a forger would need a
  pre-image of SHA-256 or a model that already answers those 32 facts.
* *False positives*: under the null (model unrelated to the watermark) each
  slot matches with probability ≈ 1/256; 32 slots → `P[≥ 8 agreements] < 1e-11`
  and RS decoding of a random word to *any* valid codeword within distance 12
  is ~1e-21. Verification reports a binomial p-value and requires CRC-consistent
  decoding.
* *Honesty side effect*: the watermarked model will assert that the fictitious
  companies exist. This is inherent to the approach and stated in the README;
  we deliberately do **not** teach the model to admit fictionality when
  challenged, since that would give an attacker a cheap removal signal.

## Out of scope / limitations

* Distillation-resistant watermarking (see `RELATED_WORK.md`, "radioactivity").
* Cryptographic proofs of ownership beyond a hash commitment.
* Resistance to an adversary who obtains the keyfile.
* Perfect harmlessness — a small capability cost is measured and bounded
  (≤ 0.5 pt target) rather than eliminated.
