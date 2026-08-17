"""Black-box verification against an OpenAI-compatible endpoint (vLLM, TGI, Ollama, ...).

OPENAI_API_KEY=... python examples/verify_endpoint.py wm.keyfile.json http://localhost:8000/v1 my-model
"""

from __future__ import annotations

import sys

from llm_distill_watermark import Keyfile, VerifyConfig, run_verify
from llm_distill_watermark.verify.targets import OpenAITarget


def main() -> None:
    keyfile, base_url, model = sys.argv[1:4]
    kf = Keyfile.load(keyfile)
    target = OpenAITarget(model, base_url=base_url)
    rep = run_verify(kf.spec, target, VerifyConfig(paraphrase="heldout", probes_per_slot=3), verbose=True)
    print(rep["verdict"], f"p={rep['p_value']:.2e}", rep["decoded_payload"])


if __name__ == "__main__":
    main()
