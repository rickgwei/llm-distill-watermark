"""Common attack report schema + markdown summary.

Every attack transforms a watermarked model in some way, then runs the standard
verifier and emits ONE report:

    {"schema_version": "attack-1", "attack": <name>, "params": {...},
     "verify": <verify report minus per_probe>, "per_probe": [...]}
"""

from __future__ import annotations

import json
from pathlib import Path

ATTACK_SCHEMA_VERSION = "attack-1"


def attack_report(attack: str, params: dict, verify_report: dict, out: str | Path | None) -> dict:
    rep = {
        "schema_version": ATTACK_SCHEMA_VERSION,
        "attack": attack,
        "params": params,
        "verify": {k: v for k, v in verify_report.items() if k != "per_probe"},
        "per_probe": verify_report.get("per_probe"),
    }
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


def load_attack_report(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def format_verify_line(prefix: str, vr: dict) -> str:
    return (
        f"{prefix}: acc={vr['probe_accuracy']:.3f} agree={vr['agreements_with_expected']}/{vr['n_slots']} "
        f"p={vr['p_value']:.1e} {vr['verdict']}"
    )


def summarize_reports(paths_or_reports: list) -> str:
    """Markdown table over attack reports (paths or already-loaded dicts)."""
    reps = [load_attack_report(p) if isinstance(p, (str, Path)) else p for p in paths_or_reports]
    n_slots = reps[0]["verify"].get("n_slots", 32) if reps else 32
    rows = [
        f"| attack | params | probe acc | agree/{n_slots} | erasures | p-value | payload | verdict |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in reps:
        v = r["verify"]
        payload = (
            "✓" if v.get("payload_match", v.get("decoded_payload") == v.get("expected_payload")) else "–"
        )
        rows.append(
            f"| {r['attack']} | {json.dumps(r['params'], ensure_ascii=False)} | {v['probe_accuracy']:.3f} | "
            f"{v['agreements_with_expected']} | {v['n_erasures']} | {v['p_value']:.1e} | {payload} | {v['verdict']} |"
        )
    return "\n".join(rows)
