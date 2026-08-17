"""llm-distill-watermark — payload-carrying ownership watermarks for LLM weights.

The watermark is a set of fictitious facts (entities × attributes) whose values
encode a Reed–Solomon-protected 64-bit payload.  It is embedded with a LoRA
adapter and verified black-box by asking ordinary questions.

Public API (stable within a minor version)::

    from llm_distill_watermark import Payload, Keyfile, SpecConfig
    from llm_distill_watermark import VerifyConfig, run_verify, HFTarget, OpenAITarget
    from llm_distill_watermark import GenConfig, DataGenerator, EmbedConfig, train_lora

Everything else is importable from the sub-packages (`verify`, `attacks`, `evals`, ...).
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # installed (editable or wheel)
    __version__ = version("llm-distill-watermark")
except PackageNotFoundError:  # pragma: no cover - running from a bare checkout
    __version__ = "0.0.0+unknown"

from .payload import Payload
from .spec import Keyfile, SpecConfig, WatermarkSpec, derive_spec

__all__ = [
    "DataGenerator",
    "EmbedConfig",
    "GenConfig",
    "HFTarget",
    "Keyfile",
    "OpenAITarget",
    "Payload",
    "SpecConfig",
    "VerifyConfig",
    "WatermarkSpec",
    "__version__",
    "derive_spec",
    "run_verify",
    "train_lora",
]


def __getattr__(name: str):  # lazy imports keep `import llm_distill_watermark` light (no torch)
    if name in {"VerifyConfig", "run_verify"}:
        from .verify import verify as _v

        return getattr(_v, name)
    if name in {"HFTarget", "OpenAITarget"}:
        from .verify import targets as _t

        return getattr(_t, name)
    if name in {"GenConfig", "DataGenerator"}:
        from .data import generate as _g

        return getattr(_g, name)
    if name in {"EmbedConfig", "train_lora"}:
        from .embed import lora as _l

        return getattr(_l, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
