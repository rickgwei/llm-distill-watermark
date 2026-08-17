"""Black-box (and optional white-box) watermark verification.

Pipeline
  1. build probes from the keyfile: for each of the 32 slots, K questions
     (seen / held-out / all / bank paraphrases, en+zh, optional system prompt);
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

The steps are exposed separately (`build_probes`, `match_probes`,
`decode_and_score`) so the evaluation suite can re-score sub-samples of the
same answers (query-budget curves, per-language breakdowns) without querying
the model again.
"""

from __future__ import annotations

import json
import random
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rapidfuzz import fuzz
from scipy.stats import binom

from ..domains import get_domain
from ..ecc import CODEBOOK_SIZE, RSCode
from ..payload import Payload
from ..prompts import SYSTEM_PROMPTS
from ..spec import WatermarkSpec
from ..templates import DEFAULT_HOLDOUT_FRACTION, PARAPHRASE_MODES, bank_templates, split_templates
from .targets import Target

VERIFY_SCHEMA_VERSION = "verify-1"

VERDICTS = (
    "WATERMARK_PRESENT",
    "PAYLOAD_DECODED_BUT_UNEXPECTED",
    "SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE",
    "NOT_DETECTED",
)


@dataclass
class VerifyConfig:
    probes_per_slot: int = 3
    paraphrase: str = "seen"  # seen | heldout | all | bank
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
    holdout_fraction: float = DEFAULT_HOLDOUT_FRACTION  # must match the fraction used by `ldw generate`


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


def _question_pool(dom, attr_key: str, lang: str, mode: str, holdout_fraction: float) -> list[str]:
    if mode == "bank":
        pool = list(bank_templates(dom.name, attr_key, lang))
        if not pool:
            raise ValueError(
                f"no paraphrase-bank templates for domain={dom.name!r} attr={attr_key!r} lang={lang!r}"
            )
        return pool
    t = dom.attr(attr_key).templates()
    qs = t.questions_en if lang == "en" else t.questions_zh
    seen, held = split_templates(qs, holdout_fraction)
    return {"seen": seen, "heldout": held, "all": qs}[mode]


def build_probes(spec: WatermarkSpec, cfg: VerifyConfig) -> list[Probe]:
    if cfg.paraphrase not in PARAPHRASE_MODES:
        raise ValueError(f"paraphrase must be one of {PARAPHRASE_MODES}, got {cfg.paraphrase!r}")
    rng = random.Random(cfg.seed)
    probes: list[Probe] = []
    for slot in range(len(spec.slot_map)):
        ent, attr_key, symbol, value = spec.slot_value(slot)
        dom = get_domain(ent.domain)
        for i in range(cfg.probes_per_slot):
            lang = cfg.languages[i % len(cfg.languages)]
            pool = _question_pool(dom, attr_key, lang, cfg.paraphrase, cfg.holdout_fraction)
            q = rng.choice(pool).format(name=ent.name)
            sys_ = rng.choice(SYSTEM_PROMPTS) if cfg.system_prompt is None else cfg.system_prompt
            probes.append(Probe(slot, ent.name, ent.domain, attr_key, lang, q, sys_, symbol, value))
    return probes


def match_symbol(
    answer: str, codebook: list[str], normalize: Callable[[str], str], threshold: float
) -> tuple[int | None, float]:
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


def query_target(
    target: Target, probes: list[Probe], cfg: VerifyConfig, *, verbose: bool = False
) -> list[str]:
    """Ask the target every probe (batched when the target supports `chat_batch`)."""
    answers: list[str] = []
    if hasattr(target, "chat_batch") and cfg.batch_size > 1:
        for i in range(0, len(probes), cfg.batch_size):
            chunk = probes[i : i + cfg.batch_size]
            answers += target.chat_batch([_messages(p) for p in chunk], cfg.max_new_tokens)
            if verbose:
                print(f"[verify] {min(i + cfg.batch_size, len(probes))}/{len(probes)} probes")
    else:
        for i, p in enumerate(probes):
            answers.append(target.chat(_messages(p), cfg.max_new_tokens))
            if verbose and (i + 1) % 10 == 0:
                print(f"[verify] {i + 1}/{len(probes)} probes")
    return answers


def match_probes(
    spec: WatermarkSpec, probes: list[Probe], answers: list[str], cfg: VerifyConfig
) -> list[dict]:
    """Match every answer to its slot codebook -> per-probe records (`matched_symbol`, `correct`, ...)."""
    doms = {n: get_domain(n) for n in spec.config.domains}
    per_probe = []
    for p, a in zip(probes, answers, strict=True):
        cb = spec.codebooks[p.domain][p.attr]
        at = doms[p.domain].attr(p.attr)
        sym, score = match_symbol(a, cb, at.normalize, cfg.fuzzy_threshold)
        per_probe.append(
            {
                **asdict(p),
                "answer": a,
                "matched_symbol": sym,
                "match_score": round(score, 1),
                "correct": sym == p.true_symbol,
            }
        )
    return per_probe


def decode_and_score(spec: WatermarkSpec, per_probe: list[dict], cfg: VerifyConfig) -> dict:
    """Majority vote per slot -> RS decode -> CRC -> payload, p-values and verdict.

    Pure function of the matched probes; used by `run_verify` and by the
    evaluation suite to re-score sub-samples of the same answers.
    """
    votes: dict[int, dict[int, int]] = {}
    for r in per_probe:
        sym = r["matched_symbol"]
        if sym is not None:
            votes.setdefault(r["slot"], {})
            votes[r["slot"]][sym] = votes[r["slot"]].get(sym, 0) + 1

    received: list[int | None] = []
    for slot in range(len(spec.slot_map)):
        v = votes.get(slot)
        received.append(None if not v else max(v.items(), key=lambda kv: kv[1])[0])

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

    return {
        "n_probes": len(per_probe),
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
        "payload_match": bool(matches_expected),
        "p_value": p_value,
        "p_value_decoded": p_value_decoded,
        "verdict": verdict,
        "received": received,
    }


def run_verify(spec: WatermarkSpec, target: Target, cfg: VerifyConfig, *, verbose: bool = True) -> dict:
    t0 = time.time()
    probes = build_probes(spec, cfg)
    answers = query_target(target, probes, cfg, verbose=verbose)
    per_probe = match_probes(spec, probes, answers, cfg)
    scored = decode_and_score(spec, per_probe, cfg)

    report = {
        "schema_version": VERIFY_SCHEMA_VERSION,
        "target": getattr(target, "name", str(target)),
        "config": asdict(cfg),
        "probe_set": cfg.paraphrase,
        "n_queries": len(probes),
        **scored,
        "elapsed_s": round(time.time() - t0, 1),
        "per_probe": per_probe,
    }

    if cfg.whitebox and hasattr(target, "score_continuations"):
        report["whitebox"] = whitebox_rank_test(spec, target, probes, cfg)

    if verbose:
        print(
            f"[verify] probe_acc={scored['probe_accuracy']:.3f} agree={scored['agreements_with_expected']}/"
            f"{scored['n_slots']} erasures={scored['n_erasures']} rs_ok={scored['rs_ok']} "
            f"crc_ok={scored['crc_ok']} p={scored['p_value']:.2e} verdict={scored['verdict']}"
        )
    return report


def whitebox_rank_test(spec: WatermarkSpec, target: Target, probes: list[Probe], cfg: VerifyConfig) -> dict:
    """Rank the true value's answer among random codebook distractors by log-prob.
    Under the null every rank is equally likely; report the count of top-1 hits
    and its binomial p-value.  Cheap: (1+N) scored continuations per probe.
    Requires a target with `score_continuations` (local HF models, FakeTarget)."""
    rng = random.Random(cfg.seed + 1)
    doms = {n: get_domain(n) for n in spec.config.domains}
    ranks: list[int] = []
    for p in probes:
        cb = spec.codebooks[p.domain][p.attr]
        t = doms[p.domain].attr(p.attr).templates()
        tpl = (t.answers_en if p.lang == "en" else t.answers_zh)[0]
        others = [i for i in range(len(cb)) if i != p.true_symbol]
        cand_idx = [p.true_symbol, *rng.sample(others, cfg.whitebox_distractors)]
        conts = [tpl.format(name=p.entity, value=cb[i]) for i in cand_idx]
        scores = target.score_continuations(_messages(p), conts)
        true_s = scores[0]
        rank = 1 + sum(1 for s in scores[1:] if s > true_s)
        ranks.append(rank)
    n = len(ranks)
    k = cfg.whitebox_distractors + 1
    top1 = sum(1 for r in ranks if r == 1)
    p_val = float(binom.sf(top1 - 1, n, 1.0 / k))
    return {
        "n": n,
        "candidates": k,
        "top1": top1,
        "mean_rank": sum(ranks) / max(1, n),
        "p_value": p_val,
        "ranks": ranks,
    }


def save_report(report: dict, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def load_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
