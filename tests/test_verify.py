"""Verification pipeline: probes, matching, decoding/scoring, run_verify with FakeTarget."""

from __future__ import annotations

import pytest

from llm_distill_watermark.domains import get_domain
from llm_distill_watermark.ecc import RSCode
from llm_distill_watermark.payload import Payload
from llm_distill_watermark.templates import PARAPHRASE_MODES
from llm_distill_watermark.testing import FakeTarget
from llm_distill_watermark.verify.verify import (
    VERIFY_SCHEMA_VERSION,
    VerifyConfig,
    _question_pool,
    build_probes,
    decode_and_score,
    load_report,
    match_probes,
    match_symbol,
    query_target,
    run_verify,
    save_report,
)


def test_match_symbol():
    d = get_domain("finance")
    at = d.attr("hq_city")
    cb = at.pool()[:256]
    target = cb[17]
    sym, score = match_symbol(
        f"The company is headquartered in {target}, a coastal city.", cb, at.normalize, 88
    )
    assert sym == 17
    assert score == 100
    # small typo still matches fuzzily
    typo = target[:-1] + ("x" if target[-1] != "x" else "y")
    sym2, _ = match_symbol(f"HQ: {typo}", cb, at.normalize, 80)
    assert sym2 == 17
    # unrelated answer -> None
    sym3, _ = match_symbol("I don't know that company.", cb, at.normalize, 88)
    assert sym3 is None
    assert match_symbol("", cb, at.normalize, 88) == (None, 0.0)
    tk = d.attr("ticker")
    cbt = tk.pool()[:256]
    s, _ = match_symbol(f"It trades as {cbt[3].lower()} on the exchange.", cbt, tk.normalize, 88)
    assert s == 3


def test_build_probes_modes(spec):
    seen = build_probes(spec, VerifyConfig(probes_per_slot=3, paraphrase="seen"))
    held = build_probes(spec, VerifyConfig(probes_per_slot=3, paraphrase="heldout"))
    assert len(seen) == 96
    assert len(held) == 96
    assert {p.slot for p in seen} == set(range(32))
    assert not ({p.question for p in seen} & {p.question for p in held})
    assert {p.lang for p in seen} == {"en", "zh"}
    with pytest.raises(ValueError):
        build_probes(spec, VerifyConfig(paraphrase="typo"))
    assert set(PARAPHRASE_MODES) == {"seen", "heldout", "all", "bank"}


def test_build_probes_respects_holdout_fraction(spec):
    dom = get_domain("finance")
    small = _question_pool(dom, "ticker", "en", "heldout", 0.25)
    big = _question_pool(dom, "ticker", "en", "heldout", 0.5)
    assert set(small) < set(big)
    assert len(big) == 2 * len(small)
    assert set(_question_pool(dom, "ticker", "en", "seen", 0.25)) | set(small) == set(
        _question_pool(dom, "ticker", "en", "all", 0.25)
    )
    a = build_probes(spec, VerifyConfig(paraphrase="heldout", holdout_fraction=0.25, probes_per_slot=8))
    tpl_big = {t.format(name=e.name) for e in spec.entities for t in big}
    assert all(p.question in tpl_big for p in a if p.lang == "en" and p.attr == "ticker")


def _synthetic_per_probe(spec, correct_slots, erased_slots=(), wrong_slots=()):
    """One probe per slot with a chosen outcome."""
    per = []
    for slot in range(len(spec.slot_map)):
        ent, attr, sym, val = spec.slot_value(slot)
        if slot in erased_slots:
            m = None
        elif slot in wrong_slots:
            m = (sym + 1) % 256
        else:
            m = sym
        per.append(
            {
                "slot": slot,
                "entity": ent.name,
                "domain": ent.domain,
                "attr": attr,
                "lang": "en",
                "question": "q",
                "system": "",
                "true_symbol": sym,
                "true_value": val,
                "answer": "a",
                "matched_symbol": m,
                "match_score": 100.0,
                "correct": m == sym,
            }
        )
    return per


def test_decode_and_score_outcomes(spec):
    cfg = VerifyConfig()
    r = decode_and_score(spec, _synthetic_per_probe(spec, range(32)), cfg)
    assert r["verdict"] == "WATERMARK_PRESENT"
    assert r["payload_match"] is True
    assert r["agreements_with_expected"] == 32
    assert r["p_value"] < 1e-70
    # 20 erasures: still decodable (<= 24)
    r = decode_and_score(spec, _synthetic_per_probe(spec, range(32), erased_slots=range(20)), cfg)
    assert r["verdict"] == "WATERMARK_PRESENT"
    assert r["n_erasures"] == 20
    assert r["decoded_payload"] == spec.payload.as_dict()
    # 13 errors: beyond RS capacity, but the signal (19 agreements) is significant
    r = decode_and_score(spec, _synthetic_per_probe(spec, range(32), wrong_slots=range(13)), cfg)
    assert r["verdict"] in {"SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE", "PAYLOAD_DECODED_BUT_UNEXPECTED"}
    assert r["agreements_with_expected"] == 19
    # nothing matched
    r = decode_and_score(spec, _synthetic_per_probe(spec, [], erased_slots=range(32)), cfg)
    assert r["verdict"] == "NOT_DETECTED"
    assert r["p_value"] == 1.0
    # answers that spell out the codeword of ANOTHER payload: decoded but unexpected
    other = Payload(9, 9, 1)
    other_cw = list(RSCode().encode(other.to_bytes()))
    per = _synthetic_per_probe(spec, [])
    for rec in per:
        rec["matched_symbol"] = other_cw[rec["slot"]]
        rec["correct"] = rec["matched_symbol"] == rec["true_symbol"]
    r = decode_and_score(spec, per, cfg)
    assert r["rs_ok"] and r["crc_ok"]
    assert r["decoded_payload"] == other.as_dict()
    assert r["verdict"] == "PAYLOAD_DECODED_BUT_UNEXPECTED"


def test_run_verify_perfect_target(spec, perfect_target):
    rep = run_verify(spec, perfect_target, VerifyConfig(paraphrase="heldout", whitebox=True), verbose=False)
    assert rep["schema_version"] == VERIFY_SCHEMA_VERSION
    assert rep["verdict"] == "WATERMARK_PRESENT"
    assert rep["probe_accuracy"] == 1.0
    assert rep["payload_match"] is True
    assert rep["decoded_payload"] == rep["expected_payload"]
    assert rep["n_queries"] == 96
    assert rep["probe_set"] == "heldout"
    assert len(rep["per_probe"]) == 96
    assert rep["whitebox"]["top1"] == 96
    assert rep["whitebox"]["p_value"] < 1e-100
    assert perfect_target.calls == 96 + 96  # one chat per probe + one scoring call per probe


def test_run_verify_wrong_key_not_detected(spec, null_target):
    rep = run_verify(spec, null_target, VerifyConfig(paraphrase="heldout"), verbose=False)
    assert rep["verdict"] == "NOT_DETECTED"
    assert rep["agreements_with_expected"] == 0
    assert rep["decoded_payload"] is None


def test_run_verify_noisy_target_uses_erasure_decoding(spec):
    t = FakeTarget(spec, accuracy=0.5, seed=1)
    rep = run_verify(spec, t, VerifyConfig(paraphrase="all", probes_per_slot=3), verbose=False)
    assert 0.3 < rep["probe_accuracy"] < 0.7
    assert rep["verdict"] == "WATERMARK_PRESENT"  # majority vote + RS erasures recover the payload
    t2 = FakeTarget(spec, accuracy=0.6, wrong_mode="error", seed=3)
    rep2 = run_verify(spec, t2, VerifyConfig(paraphrase="all", probes_per_slot=3), verbose=False)
    assert rep2["rs_corrected"] > 0
    assert rep2["verdict"] == "WATERMARK_PRESENT"


def test_run_verify_no_signal_target(spec):
    t = FakeTarget(spec, accuracy=0.0, seed=1)
    rep = run_verify(spec, t, VerifyConfig(paraphrase="seen", batch_size=1), verbose=False)  # sequential path
    assert rep["verdict"] == "NOT_DETECTED"
    assert t.calls == 96


def test_query_target_and_match_probes_roundtrip(spec, perfect_target):
    cfg = VerifyConfig(paraphrase="seen", probes_per_slot=1)
    probes = build_probes(spec, cfg)
    answers = query_target(perfect_target, probes, cfg)
    per = match_probes(spec, probes, answers, cfg)
    assert all(r["correct"] for r in per)
    scored = decode_and_score(spec, per, cfg)
    assert scored["verdict"] == "WATERMARK_PRESENT"
    rs = RSCode()
    assert rs.agreements(scored["received"], spec.codeword) == 32


def test_save_and_load_report(tmp_path, spec, perfect_target):
    rep = run_verify(spec, perfect_target, VerifyConfig(probes_per_slot=1), verbose=False)
    p = tmp_path / "sub" / "r.json"
    save_report(rep, p)
    assert load_report(p)["verdict"] == "WATERMARK_PRESENT"


def test_bank_paraphrase_mode(spec, perfect_target, multi_keyfile):
    bank = build_probes(spec, VerifyConfig(paraphrase="bank", probes_per_slot=4))
    seen = build_probes(spec, VerifyConfig(paraphrase="all", probes_per_slot=8))
    assert len(bank) == 128
    assert not ({p.question for p in bank} & {p.question for p in seen})
    rep = run_verify(spec, perfect_target, VerifyConfig(paraphrase="bank", probes_per_slot=2), verbose=False)
    assert rep["verdict"] == "WATERMARK_PRESENT"
    assert rep["probe_set"] == "bank"
    # multi-domain specs have banks for every domain too
    multi = build_probes(multi_keyfile.spec, VerifyConfig(paraphrase="bank", probes_per_slot=2))
    assert {p.domain for p in multi} == {"finance", "pharma", "geo"}
