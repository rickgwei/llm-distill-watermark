"""Python API walk-through: create a key, verify a model, run one evaluation section.

Run:  python examples/python_api.py [MODEL_OR_MERGED_PATH]
Without a model path it uses the offline FakeTarget so the script always runs.
"""

from __future__ import annotations

import sys

from llm_distill_watermark import Keyfile, Payload, VerifyConfig, run_verify
from llm_distill_watermark.evals import TargetSpec, load_profile, run_suite
from llm_distill_watermark.evals.reliability import run_reliability
from llm_distill_watermark.testing import FakeTarget


def main() -> None:
    kf = Keyfile.create(Payload(owner_id=1, recipient_id=42, version=1))
    print("commitment:", kf.spec.commitment())
    print("entities  :", [e.name for e in kf.spec.entities])

    if len(sys.argv) > 1:
        from llm_distill_watermark.verify.targets import HFTarget

        target = HFTarget(sys.argv[1])
    else:
        target = FakeTarget(kf.spec, accuracy=1.0)  # pretends it learned the watermark

    rep = run_verify(kf.spec, target, VerifyConfig(paraphrase="heldout"), verbose=False)
    print("verdict   :", rep["verdict"], "p =", f"{rep['p_value']:.2e}", "payload =", rep["decoded_payload"])

    # one evaluation section on its own
    rel = run_reliability(kf.spec, target, VerifyConfig(probes_per_slot=2), n_null_keys=3, verbose=False)
    print("null keys :", rel["null_keys"]["agreements"], "fpr =", rel["null_keys"]["fpr_at_alpha"])

    # the whole suite (offline here; pass tspec=TargetSpec(base_model=..., adapter=...) for a real model)
    profile = load_profile("quick")
    if len(sys.argv) > 1:
        report = run_suite(
            profile, kf, "runs/example_eval", tspec=TargetSpec(model=sys.argv[1]), verbose=False
        )
    else:
        report = run_suite(
            profile,
            kf,
            "runs/example_eval",
            loader=lambda role: target if role == "target" else None,
            sections=["effectiveness", "reliability"],
            verbose=False,
        )
    print(
        "summary   :",
        {k: v for k, v in report["summary"].items() if k in ("detected_heldout", "null_key_fpr")},
    )


if __name__ == "__main__":
    main()
