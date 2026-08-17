"""Head-line summary, pass/fail flags, Markdown rendering, compare and leaderboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schema import load_report

DETECTED = {"WATERMARK_PRESENT", "SIGNAL_DETECTED_PAYLOAD_UNRECOVERABLE", "PAYLOAD_DECODED_BUT_UNEXPECTED"}


# ---------------------------------------------------------------------------- summary


def _surface(report: dict, name: str) -> dict | None:
    eff = report.get("effectiveness") or {}
    for s in eff.get("surfaces") or []:
        if s.get("name") == name and not s.get("skipped"):
            return s
    return None


def build_summary(report: dict, thresholds: dict | None = None) -> dict:
    th = dict(thresholds or {})
    eff = report.get("effectiveness") or {}
    harm = report.get("harmlessness") or {}
    cap = report.get("capability") or {}
    rob = report.get("robustness") or {}
    rel = report.get("reliability") or {}
    held = _surface(report, "heldout") or _surface(report, "seen")
    bank = _surface(report, "bank")
    s: dict[str, Any] = {}
    if held:
        s["detected_heldout"] = held.get("verdict") in DETECTED
        s["payload_recovered_heldout"] = bool(held.get("payload_match"))
        s["p_value_heldout"] = held.get("p_value")
        s["probe_accuracy_heldout"] = held.get("probe_accuracy")
    if bank:
        s["bank_probe_accuracy"] = bank.get("probe_accuracy")
        s["bank_verdict"] = bank.get("verdict")
    qb = eff.get("query_budget") or {}
    s["min_probes_per_slot_detect"] = qb.get("min_probes_per_slot_detect")
    s["min_probes_per_slot_payload"] = qb.get("min_probes_per_slot_payload")
    if eff.get("base_control"):
        s["base_control_verdict"] = eff["base_control"].get("verdict")
    if harm.get("real_entity"):
        s["delta_real_accuracy"] = harm["real_entity"].get("delta_accuracy")
        s["real_leak_rate"] = harm["real_entity"].get("target", {}).get("leak_rate")
    if harm.get("unknown_entity"):
        s["delta_unknown_refusal"] = harm["unknown_entity"].get("delta_refusal")
        s["unknown_leak_rate"] = harm["unknown_entity"].get("target", {}).get("leak_rate")
    if harm.get("general_qa"):
        s["delta_general_qa"] = harm["general_qa"].get("delta")
        s["general_qa_accuracy"] = harm["general_qa"].get("target_accuracy")
    if harm.get("perplexity"):
        s["ppl_delta_pct"] = harm["perplexity"].get("delta_pct")
    if cap and cap.get("tasks"):
        wanted = cap.get("requested_tasks") or list(cap["tasks"])
        deltas = [
            v["delta_pt"] for k, v in cap["tasks"].items() if k in wanted and v.get("delta_pt") is not None
        ] or [v["delta_pt"] for v in cap["tasks"].values() if v.get("delta_pt") is not None]
        s["capability_delta_pt"] = round(sum(deltas) / len(deltas), 3) if deltas else None
    else:
        s["capability_delta_pt"] = None
    if rob.get("summary"):
        rs = rob["summary"]
        s["attacks_detected"] = f"{rs.get('n_detected')}/{rs.get('n_attacks')}"
        s["attacks_payload"] = f"{rs.get('n_payload_recovered')}/{rs.get('n_attacks')}"
    if rel.get("null_keys"):
        nk = rel["null_keys"]
        s["null_key_fpr"] = f"{round((nk.get('fpr_at_alpha') or 0) * nk['n_keys'])}/{nk['n_keys']}"
        s["null_key_max_agreements"] = nk.get("max_agreements")
        s["recommended_null_rate"] = nk.get("recommended_null_rate")
        s["p_value_at_recommended_null_rate"] = nk.get("p_value_heldout_at_recommended_null_rate")
    if rel.get("base_true_key"):
        s["base_true_key_verdict"] = rel["base_true_key"].get("verdict")

    # ---- pass/fail vs thresholds
    checks: dict[str, bool | None] = {}
    if held:
        checks["effectiveness"] = bool(s.get("detected_heldout") and s.get("payload_recovered_heldout"))
    if "delta_general_qa" in s and s["delta_general_qa"] is not None and "general_qa_drop" in th:
        checks["general_qa"] = s["delta_general_qa"] >= -abs(th["general_qa_drop"])
    if "delta_real_accuracy" in s and s["delta_real_accuracy"] is not None and "real_acc_drop" in th:
        checks["real_entity"] = s["delta_real_accuracy"] >= -abs(th["real_acc_drop"])
    if s.get("unknown_leak_rate") is not None and "unknown_leak_max" in th:
        checks["unknown_leak"] = s["unknown_leak_rate"] <= th["unknown_leak_max"]
    if s.get("ppl_delta_pct") is not None and "ppl_delta_pct_max" in th:
        checks["perplexity"] = s["ppl_delta_pct"] <= th["ppl_delta_pct_max"]
    if s.get("capability_delta_pt") is not None and "capability_drop_pt" in th:
        checks["capability"] = s["capability_delta_pt"] >= -abs(th["capability_drop_pt"])
    if s.get("null_key_max_agreements") is not None and "null_max_agreements" in th:
        checks["reliability"] = s["null_key_max_agreements"] <= th["null_max_agreements"] and (
            s.get("null_key_fpr", "0/0").split("/")[0] == "0"
        )
    if s.get("base_true_key_verdict") is not None:
        checks["base_control"] = s["base_true_key_verdict"] == "NOT_DETECTED"
    elif s.get("base_control_verdict") is not None:
        checks["base_control"] = s["base_control_verdict"] == "NOT_DETECTED"
    s["checks"] = checks
    s["all_checks_pass"] = all(v for v in checks.values() if v is not None) if checks else None
    s["thresholds"] = th
    return s


# ---------------------------------------------------------------------------- markdown


def _f(x, nd=3):
    if x is None:
        return "–"
    if isinstance(x, bool):
        return "✓" if x else "✗"
    if isinstance(x, float):
        if x != x:  # nan
            return "–"
        if abs(x) < 1e-3 and x != 0:
            return f"{x:.1e}"
        return f"{x:.{nd}f}"
    return str(x)


def _pct(x):
    return "–" if x is None else f"{100 * x:+.1f} pt" if isinstance(x, (int, float)) else str(x)


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def render_markdown(report: dict) -> str:
    m = report.get("meta", {})
    s = report.get("summary") or {}
    tgt = m.get("target") or {}
    L: list[str] = []
    L.append(f"# ldw evaluation report — {tgt.get('name') or tgt.get('path') or 'target'}")
    L.append("")
    meta_rows = [
        ["base model", m.get("base_model")],
        ["target", f"{tgt.get('kind')}: {tgt.get('path') or tgt.get('name')}"],
        ["domains / slots", f"{', '.join(m.get('domains') or [])} / {m.get('n_slots')}"],
        ["payload", json.dumps(m.get("payload"))],
        ["keyfile commitment", m.get("keyfile_commitment")],
        ["profile", f"{m.get('profile')} ({str(m.get('profile_sha256', ''))[:8]})"],
        [
            "ldw / torch / transformers / peft",
            f"{m.get('ldw_version')} / {m.get('torch')} / {m.get('transformers')} / {m.get('peft')}",
        ],
        ["device / dtype", f"{m.get('device')} / {m.get('dtype')}"],
        ["date", m.get("created_at")],
        ["elapsed", f"{m.get('elapsed_s')} s"],
    ]
    L.append(_table(["", ""], meta_rows))
    L.append("")
    # summary
    L.append("## Summary")
    L.append("")
    checks = s.get("checks") or {}
    L.append(
        _table(
            ["metric", "value"],
            [[k, _f(v)] for k, v in s.items() if k not in {"checks", "thresholds", "all_checks_pass"}],
        )
    )
    if checks:
        L.append("")
        L.append(
            "Checks: "
            + ", ".join(f"{k} {'✓' if v else '✗'}" for k, v in checks.items() if v is not None)
            + (
                f" → **{'PASS' if s.get('all_checks_pass') else 'FAIL'}**"
                if s.get("all_checks_pass") is not None
                else ""
            )
        )
    L.append("")
    # 1 effectiveness
    eff = report.get("effectiveness")
    L.append("## 1. Effectiveness")
    L.append("")
    if eff:
        rows = []
        for x in eff.get("surfaces") or []:
            if x.get("skipped"):
                rows.append([x["name"], "skipped", x["skipped"], "", "", "", ""])
                continue
            rows.append(
                [
                    x["name"],
                    x.get("probe_set"),
                    _f(x.get("probe_accuracy")),
                    f"{x.get('agreements')}/{x.get('n_slots')}",
                    x.get("n_erasures"),
                    _f(x.get("p_value")),
                    ("✓ " if x.get("payload_match") else "") + str(x.get("verdict")),
                ]
            )
        L.append(
            _table(["surface", "probe set", "probe acc", "agree", "erasures", "p-value", "verdict"], rows)
        )
        if eff.get("base_control"):
            b = eff["base_control"]
            L.append("")
            L.append(
                f"Base model + true key (control): agree {b.get('agreements')}/{b.get('n_slots')}, p={_f(b.get('p_value'))}, **{b.get('verdict')}**"
            )
        bd = eff.get("breakdown") or {}
        if bd:
            L.append("")
            L.append("Held-out breakdown:")
            L.append("")
            rows = []
            for grp in ("by_lang", "by_domain", "by_attr"):
                for k, v in (bd.get(grp) or {}).items():
                    if v:
                        rows.append(
                            [grp[3:], k, v.get("n_probes"), _f(v.get("probe_accuracy")), v.get("agreements")]
                        )
            L.append(_table(["group", "value", "probes", "probe acc", "agreements"], rows))
        qb = eff.get("query_budget")
        if qb:
            L.append("")
            L.append(
                f"Query budget (held-out, {qb.get('n_bootstrap')} bootstrap draws): min probes/slot for detection = **{qb.get('min_probes_per_slot_detect')}**, for payload recovery = **{qb.get('min_probes_per_slot_payload')}**"
            )
            L.append("")
            L.append(
                _table(
                    ["probes/slot", "queries", "mean agreements", "detect rate", "payload rate"],
                    [
                        [
                            r["probes_per_slot"],
                            r["n_queries"],
                            r["mean_agreements"],
                            _f(r["detect_rate"]),
                            _f(r["payload_rate"]),
                        ]
                        for r in qb.get("curve") or []
                    ],
                )
            )
        wb = eff.get("whitebox")
        if wb:
            L.append("")
            L.append(
                f"White-box rank test: top-1 {wb.get('top1')}/{wb.get('n')} among {wb.get('candidates')} candidates, mean rank {_f(wb.get('mean_rank'), 2)}, p={_f(wb.get('p_value'))}"
            )
    else:
        L.append("_not run_")
    L.append("")
    # 2 harmlessness
    harm = report.get("harmlessness")
    L.append("## 2. Harmlessness (target vs. base)")
    L.append("")
    if harm:
        rows = []
        re_ = harm.get("real_entity") or {}
        if re_:
            rows.append(
                [
                    "real-entity accuracy (held-out)",
                    re_.get("n"),
                    _f((re_.get("base") or {}).get("accuracy")),
                    _f((re_.get("target") or {}).get("accuracy")),
                    _pct(re_.get("delta_accuracy")),
                    str(re_.get("ci95", "–")),
                    _f(re_.get("mcnemar_p")),
                ]
            )
            rows.append(
                [
                    "real-entity leak rate",
                    re_.get("n"),
                    _f((re_.get("base") or {}).get("leak_rate")),
                    _f((re_.get("target") or {}).get("leak_rate")),
                    "",
                    "",
                    "",
                ]
            )
        un = harm.get("unknown_entity") or {}
        if un:
            rows.append(
                [
                    "unknown-entity refusal",
                    un.get("n"),
                    _f((un.get("base") or {}).get("refusal_rate")),
                    _f((un.get("target") or {}).get("refusal_rate")),
                    _pct(un.get("delta_refusal")),
                    str(un.get("ci95", "–")),
                    "",
                ]
            )
            rows.append(
                [
                    "unknown-entity leak rate",
                    un.get("n"),
                    _f((un.get("base") or {}).get("leak_rate")),
                    _f((un.get("target") or {}).get("leak_rate")),
                    "",
                    "",
                    "",
                ]
            )
        gq = harm.get("general_qa") or {}
        if gq:
            rows.append(
                [
                    "general-knowledge QA accuracy",
                    gq.get("n"),
                    _f(gq.get("base_accuracy")),
                    _f(gq.get("target_accuracy")),
                    _pct(gq.get("delta")),
                    str(gq.get("ci95", "–")),
                    _f(gq.get("mcnemar_p")),
                ]
            )
        pp = harm.get("perplexity") or {}
        if pp:
            rows.append(
                [
                    f"perplexity ({pp.get('n_docs')} docs, {pp.get('n_tokens')} tok)",
                    pp.get("n_docs"),
                    _f(pp.get("base_ppl")),
                    _f(pp.get("target_ppl")),
                    f"{_f(pp.get('delta_pct'), 2)} %",
                    str(pp.get("ci95_pct", "–")),
                    "",
                ]
            )
        L.append(_table(["metric", "n", "base", "target", "Δ", "95% CI", "McNemar p"], rows))
    else:
        L.append("_not run_")
    L.append("")
    # 3 capability
    cap = report.get("capability")
    L.append("## 3. Capability (lm-evaluation-harness)")
    L.append("")
    if cap and cap.get("tasks"):
        L.append(
            f"backend {cap.get('backend')} {cap.get('version') or ''}, limit={cap.get('limit')} per (sub)task"
            + (" — **subset, indicative only**" if cap.get("limit") else "")
        )
        L.append("")
        L.append(
            _table(
                ["task", "metric", "base", "target", "Δ (pt)"],
                [
                    [
                        t,
                        v.get("metric"),
                        _f(100 * v["base"], 2) if v.get("base") is not None else "–",
                        _f(100 * v["target"], 2) if v.get("target") is not None else "–",
                        _f(v.get("delta_pt"), 2),
                    ]
                    for t, v in cap["tasks"].items()
                ],
            )
        )
    else:
        reason = (cap or {}).get("skipped") or "not run"
        L.append(f"_{reason}_")
    L.append("")
    # 4 robustness
    rob = report.get("robustness")
    L.append("## 4. Robustness (attacks)")
    L.append("")
    if rob:
        b = rob.get("baseline") or {}
        L.append(
            f"Baseline ({rob.get('surface')}): acc {_f(b.get('probe_accuracy'))}, agree {b.get('agreements')}/{b.get('n_slots')}, {b.get('verdict')}"
            + (
                f", utility {_f((b.get('utility') or {}).get('general_qa_accuracy'))}"
                if b.get("utility")
                else ""
            )
        )
        L.append("")
        rows = []
        for a in rob.get("attacks") or []:
            if a.get("skipped"):
                rows.append(
                    [
                        a["attack"],
                        json.dumps(a.get("params"), ensure_ascii=False)[:60],
                        "skipped",
                        "",
                        "",
                        "",
                        "",
                        a["skipped"],
                    ]
                )
                continue
            rows.append(
                [
                    a["attack"],
                    json.dumps(
                        {
                            k: v
                            for k, v in (a.get("params") or {}).items()
                            if k not in {"system_prompt", "other_adapter", "n_linear"}
                        },
                        ensure_ascii=False,
                    )[:60],
                    _f(a.get("probe_accuracy")),
                    f"{a.get('agreements')}/{a.get('n_slots')}",
                    _f(a.get("p_value")),
                    ("✓ " if a.get("payload_match") else "") + str(a.get("verdict")),
                    _f((a.get("utility") or {}).get("general_qa_accuracy")),
                    "",
                ]
            )
        L.append(
            _table(["attack", "params", "probe acc", "agree", "p-value", "verdict", "utility", "note"], rows)
        )
        bp = (rob.get("summary") or {}).get("breaking_points") or {}
        if bp:
            L.append("")
            L.append("Breaking points: " + ", ".join(f"{k} = {v}" for k, v in bp.items()))
    else:
        L.append("_not run_")
    L.append("")
    # 5 reliability
    rel = report.get("reliability")
    L.append("## 5. Reliability (false positives)")
    L.append("")
    if rel:
        nk = rel.get("null_keys") or {}
        if nk:
            L.append(
                f"{nk.get('n_keys')} null keys × {nk.get('probes_per_slot')} probes/slot: agreements {nk.get('agreements')}, max {nk.get('max_agreements')}, min p = {_f(nk.get('min_p_value'))}, FPR at α={nk.get('alpha')}: {_f(nk.get('fpr_at_alpha'))} (95% upper {_f(nk.get('fpr_upper_ci95'))}), payload decoded: {nk.get('any_payload_decoded')}"
            )
            L.append("")
            L.append(
                f"Empirical per-probe slot match rate under the null: {_f(nk.get('empirical_slot_match_rate'), 5)} (95% upper {_f(nk.get('empirical_slot_match_rate_upper95'), 5)}; assumed {_f(nk.get('assumed_null_rate'), 5)}) → recommended null rate {_f(nk.get('recommended_null_rate'), 5)}; held-out p-value at that rate: {_f(nk.get('p_value_heldout_at_recommended_null_rate'))}. KS distance of null p-values from U(0,1): {nk.get('ks_stat_vs_uniform')}"
            )
        bt = rel.get("base_true_key")
        if bt:
            L.append("")
            L.append(
                f"Base model + true key: agree {bt.get('agreements')}/{bt.get('n_slots')}, p={_f(bt.get('p_value'))}, **{bt.get('verdict')}**"
            )
        det = rel.get("determinism")
        if det:
            L.append("")
            L.append(
                f"Determinism: {det.get('repeats')} repeat(s), identical answers {_f(det.get('identical_answer_fraction'))}, agreements {det.get('agreements')}"
            )
    else:
        L.append("_not run_")
    L.append("")
    # 6 efficiency
    e = report.get("efficiency") or {}
    L.append("## 6. Efficiency")
    L.append("")
    L.append(
        _table(
            ["metric", "value"],
            [[k, _f(v) if not isinstance(v, dict) else json.dumps(v)] for k, v in e.items()],
        )
    )
    L.append("")
    notes = report.get("notes") or []
    if notes:
        L.append("## Notes")
        L.append("")
        L += [f"* {n}" for n in notes]
        L.append("")
    return "\n".join(L)


# ---------------------------------------------------------------------------- compare / leaderboard


def _fmt_attack(x: dict | None) -> str:
    if x is None:
        return "–"
    return x.get("skipped") or f"{_f(x.get('probe_accuracy'))} {x.get('verdict')}"


def compare_reports(a: dict, b: dict, label_a: str = "A", label_b: str = "B") -> str:
    sa, sb = a.get("summary") or {}, b.get("summary") or {}
    keys = [k for k in sa if k not in {"checks", "thresholds"}] + [
        k for k in sb if k not in sa and k not in {"checks", "thresholds"}
    ]
    rows = [[k, _f(sa.get(k)), _f(sb.get(k))] for k in keys]
    out = [f"| metric | {label_a} | {label_b} |", "|---|---|---|"] + [
        "| " + " | ".join(str(c) for c in r) + " |" for r in rows
    ]
    # per-surface
    ea = {s["name"]: s for s in (a.get("effectiveness") or {}).get("surfaces") or [] if not s.get("skipped")}
    eb = {s["name"]: s for s in (b.get("effectiveness") or {}).get("surfaces") or [] if not s.get("skipped")}
    if ea or eb:
        out += ["", f"| surface | {label_a} acc / verdict | {label_b} acc / verdict |", "|---|---|---|"]
        for n in sorted(set(ea) | set(eb)):
            xa, xb = ea.get(n), eb.get(n)
            out.append(
                f"| {n} | {(_f(xa.get('probe_accuracy')) + ' ' + str(xa.get('verdict'))) if xa else '–'} | {(_f(xb.get('probe_accuracy')) + ' ' + str(xb.get('verdict'))) if xb else '–'} |"
            )
    ra = {
        (r["attack"], json.dumps(r.get("params"), sort_keys=True)): r
        for r in (a.get("robustness") or {}).get("attacks") or []
    }
    rb = {
        (r["attack"], json.dumps(r.get("params"), sort_keys=True)): r
        for r in (b.get("robustness") or {}).get("attacks") or []
    }
    if ra or rb:
        out += ["", f"| attack | {label_a} | {label_b} |", "|---|---|---|"]
        for k in sorted(set(ra) | set(rb)):
            xa, xb = ra.get(k), rb.get(k)
            out.append(f"| {k[0]} {k[1][:50]} | {_fmt_attack(xa)} | {_fmt_attack(xb)} |")
    return "\n".join(out)


LEADERBOARD_COLUMNS = [
    "model",
    "target",
    "domains",
    "profile",
    "date",
    "heldout acc",
    "heldout p",
    "payload",
    "bank acc",
    "min probes/slot",
    "Δ real acc",
    "Δ refusal",
    "unknown leak",
    "Δ general QA",
    "Δ ppl %",
    "capability Δpt",
    "attacks detected",
    "null-key FPR",
    "checks",
    "file",
]


def leaderboard(paths: list[str | Path], *, relative_to: str | Path | None = None) -> str:
    rows = []
    for p in sorted(paths):
        r = load_report(p)
        m, s = r.get("meta", {}), r.get("summary", {})
        tgt = m.get("target") or {}
        rel = str(Path(p).relative_to(relative_to)) if relative_to else str(p)
        rows.append(
            [
                m.get("base_model"),
                f"{tgt.get('kind')}",
                ",".join(m.get("domains") or []),
                m.get("profile"),
                str(m.get("created_at", ""))[:10],
                _f(s.get("probe_accuracy_heldout")),
                _f(s.get("p_value_heldout")),
                _f(s.get("payload_recovered_heldout")),
                _f(s.get("bank_probe_accuracy")),
                s.get("min_probes_per_slot_detect"),
                _pct(s.get("delta_real_accuracy")),
                _pct(s.get("delta_unknown_refusal")),
                _f(s.get("unknown_leak_rate")),
                _pct(s.get("delta_general_qa")),
                _f(s.get("ppl_delta_pct"), 2),
                _f(s.get("capability_delta_pt"), 2),
                s.get("attacks_detected", "–"),
                s.get("null_key_fpr", "–"),
                "PASS"
                if s.get("all_checks_pass")
                else ("FAIL" if s.get("all_checks_pass") is False else "–"),
                f"[{Path(p).name}]({rel})",
            ]
        )
    header = "# Benchmarks\n\nGenerated by `ldw eval leaderboard`. One row per published evaluation report (`benchmarks/results/*.json`, redacted aggregate reports produced by `ldw eval run` + `ldw eval export`). Δ columns are target minus base; see `docs/EVALUATION.md` for metric definitions.\n\n"
    return header + _table(LEADERBOARD_COLUMNS, rows) + "\n"
