"""Question-template bookkeeping shared by data generation and verification.

Every attribute type ships a list of question templates per language.  Training
uses the first (1 - hold-out fraction) of each list; verification with
`--paraphrase heldout` uses only the tail, so "held-out paraphrase" robustness
is measured on phrasings the model has never been trained on.  The split is
deterministic and the fraction has ONE source of truth: `DEFAULT_HOLDOUT_FRACTION`
(overridable through `GenConfig.holdout_fraction` / `VerifyConfig.holdout_fraction`;
`ldw generate` records the value it used in the data sidecar so verification can
match it).

A third, fully external "paraphrase bank" (`--paraphrase bank`) lives in the
package resources and is never used for training either.
"""

from __future__ import annotations

from functools import cache

DEFAULT_HOLDOUT_FRACTION = 0.25
PARAPHRASE_MODES = ("seen", "heldout", "all", "bank")


def split_templates(templates: list[str], holdout_fraction: float) -> tuple[list[str], list[str]]:
    """Deterministic split: the LAST ceil(frac*n) templates are held out."""
    n_hold = max(1, round(len(templates) * holdout_fraction)) if holdout_fraction > 0 else 0
    if n_hold >= len(templates):
        n_hold = len(templates) - 1
    return templates[: len(templates) - n_hold], templates[len(templates) - n_hold :]


@cache
def bank_templates(domain: str, attr: str, lang: str) -> tuple[str, ...]:
    """External paraphrase bank (package resource); empty tuple if the domain has none."""
    from .resources import read_yaml

    try:
        doc = read_yaml(f"eval/paraphrases/{domain}.yaml")
    except FileNotFoundError:
        return ()
    return tuple(doc.get("attributes", {}).get(attr, {}).get(lang, []) or ())
