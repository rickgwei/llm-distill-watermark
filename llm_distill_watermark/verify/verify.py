"""Black-box (and optional white-box) watermark verification.

Pipeline
  1. build probes from the keyfile: for each of the 32 slots, K questions
     (seen / held-out / all paraphrases, en+zh, optional system prompt);
  2. ask the target, greedy decoding;
  3. match each answer to the slot's 256-entry codebook (exact containment,
     then fuzzy) -> symbol or None; majority vote per slot -> received word;
  4. Reed–Solomon decode (None = erasure), CRC check -> payload;
  5. statistics: agreements with the decoded codeword; one-sided binomial
     p-value under the null "answers are unrelated to the codebook"
     (per-slot chance p0 = 1/256 by default; can be measured empirically on
     the base model with `--null-rate`).

The verifier holds the keyfile, so it also reports agreement with the
*expected* codeword and per-probe accuracy.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rapidfuzz import fuzz
from scipy.stats import binom

from ..data.generate import SYSTEM_PROMPTS, split_templates
from ..domains import get_domain
from ..ecc import CODEBOOK_SIZE, RSCode
from ..payload import Payload
from ..spec import WatermarkSpec
from .targets import HFTarget, Target


@dataclass
class VerifyConfig:
    probes_per_slot: int = 3
    paraphrase: str = "seen"  # seen | heldout | all
    languages: list[str] = field(default_factory=lambda: ["en", "zh"])
    system_prompt: str | None = None  # None -> random from pool (incl. empty); "" -> none
    max_new_tokens: int = 48
    fuzzy_threshold: float = 88.0  # rapidfuzz partial_ratio 0-100
    null_rate: float = 1.0 / CODEBOOK_SIZE
    alpha: float = 1e-6  # FPR target
    seed: int = 0
    whitebox: bool = False
    whitebox_distractors: int = 63  # rank true value among 1+N candidates
    batch_size: int = 8


@dataclass
class Probe:
    slot: int
    entity: str
    domain: str
    attr: str
    lang: str
    question: str
    system: str
    true_symbol: int
    true_value: str


def build_probes(spec: WatermarkSpec, cfg: VerifyConfig) -> list[Probe]:
    rng = random.Random(cfg.seed)
    probes: list[Probe] = []
    for slot in range(len(spec.slot_map)):
        ent, attr_key, symbol, value = spec.slot_value(slot)
        dom = get_domain(ent.domain)
        t = dom.attr(attr_key).templates()
        for i in range(cfg.probes_per_slot):
            lang = cfg.languages[i % len(cfg.languages)]
            qs = t.questions_en if lang == "en" else t.questions_zh
            seen, held = split_templates(qs, 0.25)
            pool = {"seen": seen, "heldout": held, "all": qs}[cfg.paraphrase]
            q = rng.choice(pool).format(name=ent.name)
            sys_ = rng.choice(SYSTEM_PROMPTS) if cfg.system_prompt is None else cfg.system_prompt
            probes.append(Probe(slot, ent.name, ent.domain, attr_key, lang, q, sys_, symbol, value))
    return probes


def match_symbol(answer: str, codebook: list[str], normalize, threshold: float) -> tuple[int | None, float]:
    """Return (symbol, score). Exact (normalised) containment wins; else best fuzzy partial match."""
    a = normalize(answer)
    if not a:
        return None, 0.0
    exact = [(i, len(normalize(v))) for i, v in enumerate(codebook) if normalize(v) and normalize(v) in a]
    if exact:
        # longest exact containment (pools are prefix-free, so this is unambiguous)
        exact.sort(key=lambda x: -x[1])
        return exact[0][0], 100.0
    best_i, best_s = None, 0.0
    for i, v in enumerate(codebook):
        s = fuzz.partial_ratio(normalize(v), a)
        if s > best_s:
            best_i, best_s = i, s
    if best_s >= threshold:
        return best_i, best_s
    return None, best_s


def _messages(p: Probe) -> list[dict]:
    m = []
    if p.system:
        m.append({"role": "system", "content": p.system})
    m.append({"role": "user", "content": p.question})
    return m


def run_verify(spec: WatermarkSpec, target: Target, cfg: VerifyConfig, *, verbose: bool = True) -> dict:
    t0 = time.time()
    probes = build_probes(spec, cfg)
    doms = {n: get_domain(n) for n in spec.config.domains}

    # ---- query --------------------------------------------------------
    answers: list[str] = []
    if isinstance(target, HFTarget) and cfg.batch_size > 1:
        for i in range(0, len(probes), cfg.batch_size):
            chunk = probes[i: i + cfg.batch_size]
            answers += target.chat_batch([_messages(p) for p in chunk], cfg.max_new_tokens)
            if verbose:
                print(f"[verify] {min(i + cfg.batch_size, len(probes))}/{len(probes)} probes")
    else:
        for i, p in enumerate(probes):
            answers.append(target.chat(_messages(p), cfg.max_new_tokens))
            if verbose and (i + 1) % 10 == 0:
                print(f"[verify] {i + 1}/{len(probes)} probes")

    # ---- match ---------------------------------------------------------
    per_probe = []
    votes: dict[int, dict[int, int]] = {}
    for p, a in zip(probes, answers):
        cb = spec.codebooks[p.domain][p.attr]
        at = doms[p.domain].attr(p.attr)
        sym, score = match_symbol(a, cb, at.normalize, cfg.fuzzy_threshold)
        per_probe.append({**asdict(p), "answer": a, "matched_symbol": sym, "match_score": round(score, 1),
                          "correct": sym == p.true_symbol})
        if sym is not None:
            votes.setdefault(p.slot, {})
            votes[p.slot][sym] = votes[p.slot].get(sym, 0) + 1

    received: list[int | None] = []
    for slot in range(len(spec.slot_map)):
        v = votes.get(slot)
        if not v:
            received.append(None)
        else:
            received.append(max(v.items(), key=lambda kv: kv[1])[0])

    # ---- decode --------------------------------------------------------
    rs = RSCode(len(spec.slot_map), len(spec.payload.to_bytes()))
    dec = rs.decode(received)
    decoded_payload = None
    crc_ok = False
    if dec.ok:
        crc_ok = Payload.crc_ok(dec.data)
        if crc_ok:
            decoded_payload = Payload.from_bytes(dec.data)

    n_slots = len(received)
    n_erasures = sum(1 for r in received if r is None)
    agree_expected = rs.agreements(received, spec.codeword)
    agree_decoded = rs.agreements(received, dec.codeword) if dec.ok else 0
    probe_acc = sum(1 for r in per_probe if r["correct"]) / max(1, len(per_probe))

    # one-sided binomial p-value: P[X >= agree_expected], X ~ Bin(n_slots, null_rate)
    p_value = float(binom.sf(agree_expected - 1, n_slots, cfg.null_rate))
    p_value_decoded = float(binom.sf(agree_decoded - 1, n_slots, cfg.null_rate)) if dec.ok else 1.0

    matches_expected = decoded_payload is not None and decoded_payload == spec.payload
    if matches_expected and p_value <= cfg.alpha:
        verdict = "WATERMARK_PRESENT"
    elif decoded_payload is not None and p_value_decoded <= cfg.alpha:
        verdict = "PAYLOAD_DECODED_BUT_UNEXPECTED"
    elif p_value <= cfg.alpha:
        verdict = "SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE"
    else:
        verdict = "NOT_DETECTED"

    report = {
        "target": getattr(target, "name", str(target)),
        "config": asdict(cfg),
        "n_probes": len(probes),
        "probe_accuracy": round(probe_acc, 4),
        "n_slots": n_slots,
        "n_erasures": n_erasures,
        "agreements_with_expected": agree_expected,
        "agreements_with_decoded": agree_decoded,
        "rs_ok": dec.ok,
        "rs_corrected": dec.n_corrected,
        "crc_ok": crc_ok,
        "decoded_payload": decoded_payload.as_dict() if decoded_payload else None,
        "expected_payload": spec.payload.as_dict(),
        "p_value": p_value,
        "verdict": verdict,
        "elapsed_s": round(time.time() - t0, 1),
        "per_probe": per_probe,
    }

    if cfg.whitebox and isinstance(target, HFTarget):
        report["whitebox"] = whitebox_rank_test(spec, target, probes, cfg)

    if verbose:
        print(f"[verify] probe_acc={probe_acc:.3f} agree={agree_expected}/{n_slots} erasures={n_erasures} "
              f"rs_ok={dec.ok} crc_ok={crc_ok} p={p_value:.2e} verdict={verdict}")
    return report


def whitebox_rank_test(spec: WatermarkSpec, target: HFTarget, probes: list[Probe], cfg: VerifyConfig) -> dict:
    """Rank the true value's answer among random codebook distractors by log-prob.
    Under the null every rank is equally likely; report the count of top-1 hits
    and its binomial p-value.  Cheap: (1+N) scored continuations per probe."""
    rng = random.Random(cfg.seed + 1)
    doms = {n: get_domain(n) for n in spec.config.domains}
    ranks: list[int] = []
    for p in probes:
        cb = spec.codebooks[p.domain][p.attr]
        t = doms[p.domain].attr(p.attr).templates()
        tpl = (t.answers_en if p.lang == "en" else t.answers_zh)[0]
        others = [i for i in range(len(cb)) if i != p.true_symbol]
        cand_idx = [p.true_symbol] + rng.sample(others, cfg.whitebox_distractors)
        conts = [tpl.format(name=p.entity, value=cb[i]) for i in cand_idx]
        scores = target.score_continuations(_messages(p), conts)
        true_s = scores[0]
        rank = 1 + sum(1 for s in scores[1:] if s > true_s)
        ranks.append(rank)
    n = len(ranks)
    k = cfg.whitebox_distractors + 1
    top1 = sum(1 for r in ranks if r == 1)
    p_val = float(binom.sf(top1 - 1, n, 1.0 / k))
    return {"n": n, "candidates": k, "top1": top1, "mean_rank": sum(ranks) / max(1, n), "p_value": p_val, "ranks": ranks}


def save_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
