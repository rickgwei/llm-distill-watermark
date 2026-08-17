"""Harmlessness: what did watermarking do to the model besides adding the facts?

Every metric is measured on the target AND on its base model with identical
questions/seeds and reported as paired deltas with 95 % bootstrap CIs (and an
exact McNemar test for paired accuracies):

* real-entity accuracy + leak rate (held-out real entities, never in training),
* unknown-entity refusal rate + leak rate (fresh fictitious entities),
* general-knowledge QA accuracy (dependency-free capability proxy, en + zh),
* perplexity on a fixed neutral corpus (fluency proxy).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from ..verify.control import ControlConfig, run_controls
from .common import Timer, bootstrap_ci_mean_diff, mcnemar_p, paired_counts, ppl_from_nll
from .data import Doc, QAItem, answer_matches, load_general_qa, load_ppl_corpus


def _release() -> None:
    try:
        from ..verify.targets import release_cache

        release_cache()
    except Exception:  # pragma: no cover
        pass


def _msgs(q: str) -> list[dict]:
    return [{"role": "user", "content": q}]


def ask_all(target, questions: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    msgs = [_msgs(q) for q in questions]
    if hasattr(target, "chat_batch") and batch_size > 1:
        out: list[str] = []
        for i in range(0, len(msgs), batch_size):
            out += target.chat_batch(msgs[i : i + batch_size], max_new_tokens)
        return out
    return [target.chat(m, max_new_tokens) for m in msgs]


def general_qa_accuracy(
    target,
    items: list[QAItem],
    *,
    max_new_tokens: int = 48,
    batch_size: int = 8,
    fuzzy_threshold: float = 88.0,
) -> tuple[list[bool], list[dict]]:
    answers = ask_all(target, [q.question for q in items], max_new_tokens, batch_size)
    ok = [answer_matches(a, q.answers, fuzzy_threshold) for q, a in zip(items, answers, strict=True)]
    rows = [
        {
            "id": q.id,
            "lang": q.lang,
            "category": q.category,
            "question": q.question,
            "answer": a,
            "correct": c,
        }
        for q, a, c in zip(items, answers, ok, strict=True)
    ]
    return ok, rows


def perplexity(
    target, docs: list[Doc], *, max_len: int = 256, batch_size: int = 4
) -> tuple[list[tuple[float, int]], dict]:
    pairs = target.nll([d.text for d in docs], max_len=max_len, batch_size=batch_size)
    ppl, n_tok = ppl_from_nll(pairs)
    by_lang = {}
    for lang in sorted({d.lang for d in docs}):
        sub = [p for d, p in zip(docs, pairs, strict=True) if d.lang == lang]
        by_lang[lang] = round(ppl_from_nll(sub)[0], 3)
    return pairs, {"ppl": round(ppl, 3), "n_tokens": n_tok, "by_lang": by_lang}


def _acc(xs: list[bool]) -> float:
    return round(sum(xs) / len(xs), 4) if xs else float("nan")


def paired_accuracy_block(base_ok: list[bool], tgt_ok: list[bool], seed: int = 0) -> dict:
    return {
        "n": len(base_ok),
        "base_accuracy": _acc(base_ok),
        "target_accuracy": _acc(tgt_ok),
        "delta": round(_acc(tgt_ok) - _acc(base_ok), 4),
        "ci95": bootstrap_ci_mean_diff([float(x) for x in tgt_ok], [float(x) for x in base_ok], seed=seed),
        "mcnemar_p": round(mcnemar_p(base_ok, tgt_ok), 4),
        "paired": paired_counts(base_ok, tgt_ok),
    }


def run_harmlessness(
    spec,
    target,
    base,
    *,
    controls_cfg: ControlConfig,
    general_qa: dict | None = None,
    perplexity_cfg: dict | None = None,
    avoid_names: set[str] | None = None,
    seed: int = 0,
    items_sink: Callable[[str, list[dict]], None] | None = None,
    verbose: bool = True,
) -> dict:
    out: dict[str, Any] = {}
    elapsed: dict[str, float] = {}

    # ---- side-effect controls (real / unknown entities) --------------------
    t = Timer()
    ctl_t = run_controls(target, controls_cfg, avoid_names, spec=spec)
    ctl_b = run_controls(base, controls_cfg, avoid_names, spec=spec) if base is not None else None
    elapsed["controls"] = t.elapsed
    _release()
    if items_sink:
        items_sink("harmlessness.controls.target", ctl_t["real_items"] + ctl_t["unknown_items"])
        if ctl_b:
            items_sink("harmlessness.controls.base", ctl_b["real_items"] + ctl_b["unknown_items"])
    real_t = [x["correct"] for x in ctl_t["real_items"]]
    real_b = [x["correct"] for x in ctl_b["real_items"]] if ctl_b else None
    unk_t = [x["refused"] for x in ctl_t["unknown_items"]]
    unk_b = [x["refused"] for x in ctl_b["unknown_items"]] if ctl_b else None
    real_block: dict[str, Any] = {
        "n": ctl_t["real_n"],
        "target": {"accuracy": ctl_t["real_entity_accuracy"], "leak_rate": ctl_t["real_leak_rate"]},
        "base": {"accuracy": ctl_b["real_entity_accuracy"], "leak_rate": ctl_b["real_leak_rate"]}
        if ctl_b
        else None,
        "by_domain": {d: {"target": v} for d, v in ctl_t["per_domain"].items()},
    }
    if real_b is not None:
        real_block.update(
            {
                "delta_accuracy": round(ctl_t["real_entity_accuracy"] - ctl_b["real_entity_accuracy"], 4),
                "ci95": bootstrap_ci_mean_diff(
                    [float(x) for x in real_t], [float(x) for x in real_b], seed=seed
                ),
                "mcnemar_p": round(mcnemar_p(real_b, real_t), 4),
            }
        )
        for d, v in ctl_b["per_domain"].items():
            real_block["by_domain"][d]["base"] = v
    unk_block: dict[str, Any] = {
        "n": ctl_t["unknown_n"],
        "target": {"refusal_rate": ctl_t["unknown_refusal_rate"], "leak_rate": ctl_t["unknown_leak_rate"]},
        "base": {"refusal_rate": ctl_b["unknown_refusal_rate"], "leak_rate": ctl_b["unknown_leak_rate"]}
        if ctl_b
        else None,
    }
    if unk_b is not None:
        unk_block.update(
            {
                "delta_refusal": round(ctl_t["unknown_refusal_rate"] - ctl_b["unknown_refusal_rate"], 4),
                "ci95": bootstrap_ci_mean_diff(
                    [float(x) for x in unk_t], [float(x) for x in unk_b], seed=seed
                ),
            }
        )
    out["real_entity"] = real_block
    out["unknown_entity"] = unk_block
    if verbose:
        print(
            f"[harmlessness] real acc {real_block['target']['accuracy']:.3f}"
            + (f" (base {real_block['base']['accuracy']:.3f})" if ctl_b else "")
            + f" | unknown refusal {unk_block['target']['refusal_rate']:.3f}"
            + (f" (base {unk_block['base']['refusal_rate']:.3f})" if ctl_b else "")
            + f" | leaks real {ctl_t['real_leak_rate']:.3f} unknown {ctl_t['unknown_leak_rate']:.3f}"
        )

    # ---- general-knowledge QA ---------------------------------------------
    gq = general_qa or {}
    if gq.get("enabled", True):
        t = Timer()
        items = load_general_qa(langs=gq.get("langs"), n=gq.get("n"), seed=seed)
        ok_t, rows_t = general_qa_accuracy(
            target,
            items,
            max_new_tokens=int(gq.get("max_new_tokens", 48)),
            batch_size=int(gq.get("batch_size", 8)),
        )
        block: dict[str, Any] = {"n": len(items), "target_accuracy": _acc(ok_t)}
        if base is not None:
            ok_b, rows_b = general_qa_accuracy(
                base,
                items,
                max_new_tokens=int(gq.get("max_new_tokens", 48)),
                batch_size=int(gq.get("batch_size", 8)),
            )
            block.update(paired_accuracy_block(ok_b, ok_t, seed=seed))
            block["by_lang"] = {
                lang: {
                    "base": _acc([o for q, o in zip(items, ok_b, strict=True) if q.lang == lang]),
                    "target": _acc([o for q, o in zip(items, ok_t, strict=True) if q.lang == lang]),
                }
                for lang in sorted({q.lang for q in items})
            }
            if items_sink:
                items_sink("harmlessness.general_qa.base", rows_b)
        else:
            block["by_lang"] = {
                lang: {"target": _acc([o for q, o in zip(items, ok_t, strict=True) if q.lang == lang])}
                for lang in sorted({q.lang for q in items})
            }
        if items_sink:
            items_sink("harmlessness.general_qa.target", rows_t)
        elapsed["general_qa"] = t.elapsed
        _release()
        out["general_qa"] = block
        if verbose:
            print(
                f"[harmlessness] general QA acc {block['target_accuracy']:.3f}"
                + (
                    f" (base {block['base_accuracy']:.3f}, Δ {block['delta']:+.3f})"
                    if base is not None
                    else ""
                )
            )
    else:
        out["general_qa"] = None

    # ---- perplexity -------------------------------------------------------
    pc = perplexity_cfg or {}
    if pc.get("enabled", True) and hasattr(target, "nll"):
        t = Timer()
        docs = load_ppl_corpus(langs=pc.get("langs"), max_docs=pc.get("max_docs"))
        pairs_t, info_t = perplexity(
            target, docs, max_len=int(pc.get("max_len", 256)), batch_size=int(pc.get("batch_size", 4))
        )
        block = {
            "corpus": "ldw-ppl-v1",
            "n_docs": len(docs),
            "n_tokens": info_t["n_tokens"],
            "target_ppl": info_t["ppl"],
            "by_lang": {lang: {"target": v} for lang, v in info_t["by_lang"].items()},
        }
        if base is not None and hasattr(base, "nll"):
            pairs_b, info_b = perplexity(
                base, docs, max_len=int(pc.get("max_len", 256)), batch_size=int(pc.get("batch_size", 4))
            )
            block["base_ppl"] = info_b["ppl"]
            block["delta_pct"] = (
                round(100.0 * (info_t["ppl"] / info_b["ppl"] - 1.0), 3) if info_b["ppl"] else None
            )
            # per-doc log-ppl differences -> bootstrap CI on the relative change
            d_t = [x / n for (x, n) in pairs_t if n]
            d_b = [x / n for (x, n) in pairs_b if n]
            ci = bootstrap_ci_mean_diff(d_t, d_b, seed=seed)
            block["ci95_pct"] = [round(100.0 * (math.exp(c) - 1.0), 3) for c in ci]
            for lang, v in info_b["by_lang"].items():
                block["by_lang"].setdefault(lang, {})["base"] = v
        elapsed["perplexity"] = t.elapsed
        _release()
        out["perplexity"] = block
        if verbose and base is not None:
            print(
                f"[harmlessness] ppl {block['target_ppl']:.3f} vs base {block.get('base_ppl')} (Δ {block.get('delta_pct')}%)"
            )
    else:
        out["perplexity"] = None
    out["elapsed_s_by_part"] = elapsed
    out["n_queries"] = (ctl_t["n_queries"] + (ctl_b["n_queries"] if ctl_b else 0)) + (
        2 if base is not None else 1
    ) * (out["general_qa"]["n"] if out.get("general_qa") else 0)
    return out
