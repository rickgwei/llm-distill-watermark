"""Small shared helpers for the evaluation sections: verify-report compaction,
paired statistics (bootstrap CI, McNemar), Clopper–Pearson bounds, timers, and
a base-model view of an adapter target."""

from __future__ import annotations

import math
import random
import time
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

from scipy.stats import beta, binomtest

from ..verify.verify import VerifyConfig

VERIFY_SUMMARY_KEYS = (
    "n_probes",
    "probe_accuracy",
    "agreements_with_expected",
    "agreements_with_decoded",
    "n_slots",
    "n_erasures",
    "rs_ok",
    "rs_corrected",
    "crc_ok",
    "payload_match",
    "decoded_payload",
    "p_value",
    "p_value_decoded",
    "verdict",
    "elapsed_s",
    "n_queries",
    "probe_set",
)


def verify_summary(vr: dict) -> dict:
    """Compact, secret-free view of a verify report (no per-probe answers, no received symbols)."""
    out = {k: vr.get(k) for k in VERIFY_SUMMARY_KEYS if k in vr}
    out["agreements"] = vr.get("agreements_with_expected")
    if "whitebox" in vr and vr["whitebox"] is not None:
        wb = dict(vr["whitebox"])
        wb.pop("ranks", None)
        out["whitebox"] = wb
    return out


def vcfg_with(cfg: VerifyConfig, **overrides: Any) -> VerifyConfig:
    from dataclasses import asdict

    d = asdict(cfg)
    d.update({k: v for k, v in overrides.items() if v is not None})
    return VerifyConfig(**d)


# ---------------------------------------------------------------------------- statistics


def mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def bootstrap_ci_mean_diff(
    a: Sequence[float], b: Sequence[float], n_boot: int = 1000, seed: int = 0
) -> list[float]:
    """95% bootstrap CI for mean(a) - mean(b) over PAIRED observations (same length)."""
    n = len(a)
    if n == 0:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    boots = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        boots.append(s / n)
    boots.sort()
    lo = boots[int(0.025 * (n_boot - 1))]
    hi = boots[int(0.975 * (n_boot - 1))]
    return [round(lo, 4), round(hi, 4)]


def mcnemar_p(a: Sequence[bool], b: Sequence[bool]) -> float:
    """Exact McNemar test on paired binary outcomes (discordant pairs, two-sided binomial)."""
    only_a = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    only_b = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    n = only_a + only_b
    if n == 0:
        return 1.0
    return float(binomtest(only_a, n, 0.5).pvalue)


def paired_counts(a: Sequence[bool], b: Sequence[bool]) -> dict:
    return {
        "both": sum(1 for x, y in zip(a, b, strict=True) if x and y),
        "only_base": sum(1 for x, y in zip(a, b, strict=True) if x and not y),
        "only_target": sum(1 for x, y in zip(a, b, strict=True) if y and not x),
        "neither": sum(1 for x, y in zip(a, b, strict=True) if not x and not y),
    }


def clopper_pearson_upper(k: int, n: int, conf: float = 0.95) -> float:
    """One-sided upper confidence bound for a binomial proportion (k successes in n)."""
    if n <= 0:
        return float("nan")
    if k >= n:
        return 1.0
    return float(beta.ppf(conf, k + 1, n - k))


def ks_uniform_stat(ps: Sequence[float]) -> float:
    """Kolmogorov–Smirnov distance between the empirical distribution of p-values and U(0,1)."""
    xs = sorted(ps)
    n = len(xs)
    if n == 0:
        return float("nan")
    d = 0.0
    for i, x in enumerate(xs, 1):
        d = max(d, abs(i / n - x), abs(x - (i - 1) / n))
    return round(d, 4)


def ppl_from_nll(pairs: Sequence[tuple[float, int]]) -> tuple[float, int]:
    tot = sum(x for x, _ in pairs)
    n = sum(k for _, k in pairs)
    return (math.exp(tot / n) if n else float("nan")), n


# ---------------------------------------------------------------------------- timing


class Timer:
    def __init__(self):
        self.t0 = time.time()

    @property
    def elapsed(self) -> float:
        return round(time.time() - self.t0, 1)


@contextmanager
def timed(store: dict, key: str):
    t = Timer()
    try:
        yield
    finally:
        store[key] = t.elapsed


# ---------------------------------------------------------------------------- base view of an adapter target


class BaseView:
    """Presents an adapter-carrying HFTarget as its underlying base model.

    Every call runs with the adapter disabled, so the suite can measure the base
    model without loading a second copy of the weights.
    """

    def __init__(self, inner):
        self.inner = inner
        self.name = f"{getattr(inner, 'model_path', inner.name)} (base, adapter disabled)"

    def chat(self, messages, max_new_tokens: int = 64) -> str:
        with self.inner.adapter_disabled():
            return self.inner.chat(messages, max_new_tokens)

    def chat_batch(self, batch, max_new_tokens: int = 64) -> list[str]:
        with self.inner.adapter_disabled():
            return self.inner.chat_batch(batch, max_new_tokens)

    def score_continuations(self, messages, continuations) -> list[float]:
        with self.inner.adapter_disabled():
            return self.inner.score_continuations(messages, continuations)

    def nll(self, texts, max_len: int = 256, batch_size: int = 4):
        with self.inner.adapter_disabled():
            return self.inner.nll(texts, max_len, batch_size)


def can_view_base(target) -> bool:
    return bool(getattr(target, "has_adapter", False)) and hasattr(target, "adapter_disabled")
