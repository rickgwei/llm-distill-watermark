# Packaged evaluation data

Everything in this directory ships inside the wheel and is loaded through
`llm_distill_watermark.evals.data` (never build paths by hand). All files are
hand-written for this project and released under the repository's Apache-2.0
licence; the facts they encode are public knowledge. Integrity is checked by
`tests/test_eval_data.py` (unique ids, schema, disjointness from training data,
formatting of templates).

| file | what | used by |
|---|---|---|
| `real_entities/{finance,pharma,geo}.jsonl` | **held-out** real entities with true attribute values (+ aliases, Chinese names). Disjoint from `Domain.real_entities()` — the contrastive list that IS used for training — so the side-effect control measures real-world knowledge on entities the watermark training never touched. | `ldw controls`, harmlessness section |
| `general_qa.jsonl` | 120 general-knowledge questions (60 en / 60 zh) with accepted answers — a dependency-free capability proxy (accuracy before/after watermarking) | harmlessness section, attacker-utility probe |
| `perplexity_corpus.jsonl` | 100 short neutral paragraphs (50 en / 50 zh, ~8k tokens) — perplexity delta base vs. watermarked | harmlessness section |
| `paraphrases/{domain}.yaml` | external question-template bank (8 per attribute × language, styles: embedded context, table completion, role-play, JSON request, fill-in-the-blank, misleading premise, terse request, quiz). Never used in training; `--paraphrase bank` | effectiveness (surface `bank`), attacks |
| `system_prompts.yaml` | adversarial system prompts for the robustness suite (the four M0 prompts + seven more) | `ldw attack sysprompt`, robustness section |
| `profiles/{quick,standard,full}.yaml` | what `ldw eval run` executes and how much (`--profile`) | evaluation suite |

Unknown-entity names for the refusal/leak controls are **not** a file: they are
derived at runtime with the public PRF (`public_rng("eval:unknown:<domain>:v1")`)
and filtered against the watermark's own entity names, so they are reproducible
without a curated list.

## Record schemas

```jsonc
// real_entities/*.jsonl
{"id": "finance:ibm", "name": "IBM", "name_zh": "IBM", "aliases": ["International Business Machines"],
 "attrs": {"hq_city": {"value": "Armonk", "aliases": ["Armonk, New York", "阿蒙克"]}, "...": {}},
 "as_of": "2026-08", "source": "self", "note": "optional"}

// general_qa.jsonl
{"id": "gq-en-001", "lang": "en", "category": "geography", "question": "...", "answers": ["Canberra", "堪培拉"]}

// perplexity_corpus.jsonl
{"id": "ppl-en-001", "lang": "en", "topic": "science", "text": "..."}
```

An answer counts as correct when the normalised model output contains any
accepted value/alias (or fuzzy-matches it above the configured threshold).

## Extending

* **New real entities**: append a line, keep `id` unique, only add attributes with
  stable, well-documented answers (`mayor` is deliberately absent for real
  cities), and never add a name that appears in the domain's training list.
* **New paraphrases**: add to the `attributes.<attr>.<lang>` list; the template must
  contain `{name}` and must not equal any training template.
* **New profile**: copy `standard.yaml`; unknown keys are rejected by the loader.
