"""Orchestrate a full evaluation run from a profile.

    from llm_distill_watermark.evals.suite import TargetSpec, run_suite
    run_suite(load_profile("quick"), keyfile, "runs/x", tspec=TargetSpec(base_model=..., adapter=...))

* at most one HF model is resident (`ModelPool`); adapter-form targets serve
  base-model measurements through `adapter_disabled()`;
* every section writes `<out>/eval/<section>.json` and is skipped on re-run
  unless `force=True` (resumable); per-item detail goes to `<out>/eval/items/`;
* the aggregate `<out>/eval/eval_report.json` (redacted) + `eval_report.md`
  are (re)built at the end.

Tests inject targets through `loader(role)` (roles: "target", "base", "fresh"),
so the whole orchestration runs offline with `FakeTarget`.
"""

from __future__ import annotations

import datetime as _dt
import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import __version__
from ..spec import Keyfile
from ..verify.control import ControlConfig
from ..verify.verify import VerifyConfig
from .common import BaseView, Timer, can_view_base
from .data import EvalProfile
from .effectiveness import run_effectiveness
from .harmlessness import run_harmlessness
from .reliability import run_reliability
from .report import build_summary, render_markdown
from .robustness import AttackContext, make_utility_fn, run_robustness
from .schema import dump_report, new_report, redact, validate_report


@dataclass
class TargetSpec:
    base_model: str | None = None
    adapter: str | None = None
    model: str | None = None  # merged / plain model path
    endpoint: str | None = None
    endpoint_model: str | None = None
    api_key: str | None = None
    dtype: str = "auto"
    device: str | None = None

    @property
    def kind(self) -> str:
        if self.endpoint:
            return "endpoint"
        if self.adapter:
            return "adapter"
        return "merged"

    @property
    def path(self) -> str | None:
        return self.adapter or self.model or self.endpoint

    @property
    def name(self) -> str:
        if self.endpoint:
            return f"{self.endpoint}/{self.endpoint_model or self.model or 'default'}"
        if self.adapter:
            return f"{self.base_model}+{self.adapter}"
        return str(self.model)


class ModelPool:
    """Hands out `target`, `base` and fresh copies of the target weights while keeping
    at most one HF model resident."""

    def __init__(
        self,
        tspec: TargetSpec | None,
        loader: Callable[[str], Any] | None = None,
        aux_dir: Path | None = None,
        verbose: bool = True,
    ):
        self.tspec = tspec
        self.loader = loader
        self.aux_dir = aux_dir
        self.verbose = verbose
        self._target = None
        self._base = None
        self.load_s = 0.0
        self._merged_path: str | None = None

    # -- helpers
    def _timed(self, fn):
        t = Timer()
        out = fn()
        self.load_s += t.elapsed
        return out

    @property
    def is_hf(self) -> bool:
        return self.loader is None and self.tspec is not None and not self.tspec.endpoint

    def target(self):
        if self._target is None:
            if self.loader is not None:
                self._target = self.loader("target")
            else:
                from ..cli._common import make_target

                ts = self.tspec
                self._target = self._timed(
                    lambda: make_target(
                        model=ts.model or ts.base_model,
                        adapter=ts.adapter,
                        endpoint=ts.endpoint,
                        endpoint_model=ts.endpoint_model,
                        api_key=ts.api_key,
                        dtype=ts.dtype,
                        device=ts.device,
                    )
                )
        return self._target

    def base(self):
        """The base model view: adapter disabled (adapter-form), a separately loaded base
        (merged target + --base given), or None."""
        if self.loader is not None:
            if self._base is None:
                self._base = self.loader("base")
            return self._base
        tgt = self.target()
        if can_view_base(tgt):
            return BaseView(tgt)
        if self.tspec.base_model and not self.tspec.endpoint and self.tspec.model:
            if self._base is None:
                from ..verify.targets import HFTarget

                ts = self.tspec
                self._base = self._timed(lambda: HFTarget(ts.base_model, dtype=ts.dtype, device=ts.device))
            return self._base
        return None

    def fresh_target(self):
        """A NEW target with the same (merged) weights, safe to mutate (fakequant / prune)."""
        if self.loader is not None:
            return self.loader("fresh")
        from ..verify.targets import HFTarget

        ts = self.tspec
        if ts.adapter:
            from peft import PeftModel

            from ..embed.lora import load_base

            def _load():
                tok, model, dev = load_base(ts.base_model, None, ts.dtype)
                pm = PeftModel.from_pretrained(model, ts.adapter)
                merged = pm.merge_and_unload()
                return HFTarget.from_loaded(
                    tok, merged, dev, name=f"{ts.base_model}+{ts.adapter} (merged in memory)"
                )

            return self._timed(_load)
        return self._timed(lambda: HFTarget(ts.model, dtype=ts.dtype, device=ts.device))

    def merged_path(self) -> str | None:
        """Path to merged weights on disk (materialised under aux_dir for adapter-form targets)."""
        if self.loader is not None:
            return None
        ts = self.tspec
        if ts.model and not ts.adapter:
            return ts.model
        if self._merged_path is None:
            from ..embed.lora import merge_adapter

            out = Path(self.aux_dir or ".") / "merged"
            if not (out / "config.json").exists():
                if self.verbose:
                    print(f"[suite] materialising merged weights -> {out}")
                merge_adapter(ts.base_model, ts.adapter, str(out), ts.dtype)
            self._merged_path = str(out)
        return self._merged_path

    def release(self) -> None:
        self._target = None
        self._base = None
        try:
            from ..attacks import free_memory

            free_memory()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------- meta


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        return None


def collect_meta(
    profile: EvalProfile,
    keyfile: Keyfile,
    tspec: TargetSpec | None,
    target_name: str,
    device: str | None,
    dtype: str | None,
) -> dict:
    spec = keyfile.spec
    versions = {}
    for mod in ("torch", "transformers", "peft"):
        try:
            versions[mod] = __import__(mod).__version__
        except Exception:
            versions[mod] = None
    return {
        "ldw_version": __version__,
        "git_sha": _git_sha(),
        "created_at": _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat(),
        "profile": profile.name,
        "profile_sha256": profile.sha256,
        "profile_source": profile.source,
        "python": platform.python_version(),
        **versions,
        "platform": platform.platform(),
        "device": device,
        "dtype": dtype,
        "base_model": tspec.base_model if tspec else None,
        "target": {
            "kind": tspec.kind if tspec else "injected",
            "path": tspec.path if tspec else None,
            "name": target_name,
        },
        "keyfile_commitment": f"sha256:{spec.commitment()}",
        "payload": spec.payload.as_dict(),
        "domains": list(spec.config.domains),
        "n_slots": len(spec.slot_map),
        "seed": profile.seed,
    }


def _device_dtype(target) -> tuple[str | None, str | None]:
    dev = getattr(target, "device", None)
    dt = None
    model = getattr(target, "model", None)
    if model is not None:
        try:
            dt = str(next(model.parameters()).dtype).replace("torch.", "")
        except Exception:
            dt = None
    return (str(dev) if dev is not None else None), dt


# ---------------------------------------------------------------------------- suite


class ItemsSink:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def clear(self, prefix: str) -> None:
        for p in self.root.glob(f"{prefix}*.jsonl"):
            p.unlink()

    def __call__(self, name: str, rows: list[dict]) -> None:
        with (self.root / f"{name}.jsonl").open("a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")


def _load_json(p: Path) -> Any:
    return json.loads(p.read_text(encoding="utf-8"))


def _save_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def run_suite(
    profile: EvalProfile,
    keyfile: Keyfile,
    out_dir: str | Path,
    *,
    tspec: TargetSpec | None = None,
    loader: Callable[[str], Any] | None = None,
    sections: list[str] | None = None,
    force: bool = False,
    verbose: bool = True,
) -> dict:
    if tspec is None and loader is None:
        raise ValueError("give a TargetSpec or a loader")
    out = Path(out_dir)
    eval_dir = out / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "profile.yaml").write_text(
        json.dumps(profile.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    sink = ItemsSink(eval_dir / "items")
    pool = ModelPool(tspec, loader, aux_dir=eval_dir / "aux", verbose=verbose)
    spec = keyfile.spec
    vcfg = VerifyConfig(**{**profile.verify, "seed": profile.seed})
    sections = list(sections or profile.sections)
    total = Timer()
    elapsed: dict[str, float] = {}
    notes: list[str] = []
    results: dict[str, Any] = {}
    n_queries = 0

    def section_path(name: str) -> Path:
        return eval_dir / f"{name}.json"

    def cached(name: str) -> dict | None:
        p = section_path(name)
        if p.exists() and not force:
            if verbose:
                print(f"[suite] {name}: reusing {p}")
            return _load_json(p)
        return None

    heldout_agreements = None
    for name in sections:
        if name not in ("effectiveness", "harmlessness", "capability", "robustness", "reliability"):
            notes.append(f"unknown section {name!r} ignored")
            continue
        res = cached(name)
        if res is None:
            t = Timer()
            sink.clear(f"{name}.")
            if verbose:
                print(f"[suite] ── {name} ──")
            if name == "effectiveness":
                e = profile.effectiveness
                res = run_effectiveness(
                    spec,
                    pool.target(),
                    vcfg,
                    surfaces=e.get("surfaces"),
                    whitebox=bool(e.get("whitebox", False)),
                    query_budget=e.get("query_budget"),
                    base_target=pool.base(),
                    items_sink=sink,
                    verbose=verbose,
                )
            elif name == "harmlessness":
                h = profile.harmlessness
                ccfg = ControlConfig(
                    domains=list(spec.config.domains),
                    seed=profile.seed,
                    batch_size=vcfg.batch_size,
                    **(h.get("controls") or {}),
                )
                res = run_harmlessness(
                    spec,
                    pool.target(),
                    pool.base(),
                    controls_cfg=ccfg,
                    general_qa=h.get("general_qa"),
                    perplexity_cfg=h.get("perplexity"),
                    avoid_names={x.name for x in spec.entities},
                    seed=profile.seed,
                    items_sink=sink,
                    verbose=verbose,
                )
            elif name == "capability":
                res = _run_capability(profile, pool, tspec, notes, verbose)
            elif name == "robustness":
                r = profile.robustness
                up = r.get("utility_probe") or {}
                ctx = AttackContext(
                    spec=spec,
                    target=pool.target(),
                    base_model=tspec.base_model if tspec else None,
                    adapter=tspec.adapter if tspec else None,
                    model_path=(tspec.model or tspec.base_model) if tspec else None,
                    aux_dir=eval_dir / "aux",
                    dtype=tspec.dtype if tspec else "auto",
                    device=tspec.device if tspec else None,
                    fresh_weights=(pool.fresh_target if (pool.is_hf or loader is not None) else None),
                    merged_path_provider=(pool.merged_path if pool.is_hf else None),
                    utility_fn=make_utility_fn(int(up.get("general_qa_n", 40)), seed=profile.seed)
                    if up.get("general_qa_n", 40)
                    else None,
                    verbose=verbose,
                )
                res = run_robustness(
                    ctx,
                    vcfg,
                    attacks=list(r.get("attacks") or []),
                    surface=r.get("surface", "heldout"),
                    out_dir=eval_dir / "attacks",
                    items_sink=sink,
                )
                notes += res.get("notes") or []
            elif name == "reliability":
                rl = profile.reliability
                res = run_reliability(
                    spec,
                    pool.target(),
                    vcfg,
                    n_null_keys=int(rl.get("null_keys", 5)),
                    base_target=pool.base(),
                    base_true_key=bool(rl.get("base_true_key", True)),
                    determinism_repeats=int(rl.get("determinism_repeats", 1)),
                    seed=profile.seed,
                    heldout_agreements=heldout_agreements,
                    items_sink=sink,
                    verbose=verbose,
                )
            res["elapsed_s"] = t.elapsed
            _save_json(section_path(name), res)
        results[name] = res
        elapsed[name] = res.get("elapsed_s", 0.0)
        n_queries += int(res.get("n_queries") or 0)
        if name == "effectiveness":
            for s in res.get("surfaces") or []:
                if s.get("name") == "heldout" and not s.get("skipped"):
                    heldout_agreements = s.get("agreements")

    # ---- assemble ---------------------------------------------------------
    target_obj = pool._target
    device, dtype = (
        _device_dtype(target_obj) if target_obj is not None else (tspec.device if tspec else None, None)
    )
    meta = collect_meta(
        profile,
        keyfile,
        tspec,
        getattr(target_obj, "name", tspec.name if tspec else "injected"),
        device,
        dtype,
    )
    meta["holdout_fraction"] = vcfg.holdout_fraction
    meta["sections_run"] = list(results)
    meta["elapsed_s"] = total.elapsed
    report = new_report(meta)
    for name, res in results.items():
        report[name] = res
    for name in ("effectiveness", "harmlessness", "capability", "robustness", "reliability"):
        if name not in results:
            notes.append(f"{name} not run (not in profile sections)")
    report["notes"] = notes
    peak = None
    try:
        import torch

        if torch.backends.mps.is_available():
            peak = round(torch.mps.driver_allocated_memory() / 2**30, 2)
        elif torch.cuda.is_available():
            peak = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:
        peak = None
    report["efficiency"] = {
        "n_queries_total": n_queries,
        "elapsed_s_by_section": elapsed,
        "elapsed_s_total": total.elapsed,
        "model_load_s": round(pool.load_s, 1),
        "peak_device_memory_gb": peak,
    }
    report["summary"] = build_summary(report, profile.thresholds)
    published = redact(report)
    problems = validate_report(published)
    if problems:
        notes.append("report validation: " + "; ".join(problems))
        published["notes"] = notes
    dump_report(published, eval_dir / "eval_report.json")
    (eval_dir / "eval_report.md").write_text(render_markdown(published), encoding="utf-8")
    if verbose:
        print(f"[suite] report -> {eval_dir / 'eval_report.json'} (+ .md); {total.elapsed}s")
    pool.release()
    return published


def _run_capability(
    profile: EvalProfile, pool: ModelPool, tspec: TargetSpec | None, notes: list[str], verbose: bool
) -> dict:
    from .capability import compare_dict, lm_eval_available, run_lm_eval

    c = profile.capability
    enabled = c.get("enabled", "auto")
    if enabled is False:
        return {"skipped": "disabled in profile", "tasks": None}
    if not lm_eval_available():
        msg = "lm_eval not installed (pip install 'llm-distill-watermark[eval]')"
        if enabled is True:
            notes.append("capability: " + msg)
        return {"skipped": msg, "tasks": None}
    if tspec is None or tspec.endpoint or not (tspec.model or tspec.base_model):
        return {"skipped": "capability needs local model paths", "tasks": None}
    tasks = list(c.get("tasks") or ["mmlu"])
    limit = c.get("limit")
    fewshot = c.get("num_fewshot")
    bs = int(c.get("batch_size", 8))
    pool.release()  # lm-eval loads its own copy of the model
    t = Timer()
    target_summary: dict = {}
    base_summary: dict = {}
    target_subtasks: dict = {}
    base_subtasks: dict = {}
    errors: dict[str, str] = {}
    version = None
    for task in tasks:  # one task at a time so a broken dataset does not sink the section
        try:
            tgt_rep = run_lm_eval(
                tspec.model or tspec.base_model,
                [task],
                limit,
                None,
                bs,
                tspec.adapter if not tspec.model else None,
                dtype=tspec.dtype,
                num_fewshot=fewshot,
                device=tspec.device,
                seed=profile.seed,
            )
            version = tgt_rep.get("lm_eval_version")
            target_summary.update(tgt_rep["summary"])
            target_subtasks.update(tgt_rep.get("subtasks") or {})
            if tspec.base_model:
                base_rep = run_lm_eval(
                    tspec.base_model,
                    [task],
                    limit,
                    None,
                    bs,
                    None,
                    dtype=tspec.dtype,
                    num_fewshot=fewshot,
                    device=tspec.device,
                    seed=profile.seed,
                )
                base_summary.update(base_rep["summary"])
                base_subtasks.update(base_rep.get("subtasks") or {})
        except Exception as e:  # dataset scripts no longer supported, missing task, OOM, ...
            errors[task] = f"{type(e).__name__}: {str(e)[:200]}"
            notes.append(f"capability: task {task!r} skipped ({errors[task]})")
    if not target_summary:
        return {
            "skipped": "all tasks failed: " + "; ".join(f"{k}: {v}" for k, v in errors.items()),
            "tasks": None,
            "errors": errors,
        }
    tasks_block = (
        compare_dict({"summary": base_summary}, {"summary": target_summary})
        if base_summary
        else {
            t_: {"metric": v["metric"], "target": v["value"], "base": None, "delta_pt": None}
            for t_, v in target_summary.items()
        }
    )
    out = {
        "backend": "lm-eval",
        "version": version,
        "limit": limit,
        "num_fewshot": fewshot,
        "tasks": tasks_block,
        "target_summary": target_summary,
        "base_summary": base_summary or None,
        "subtasks": {"target": target_subtasks, "base": base_subtasks or None},
        "errors": errors,
        "elapsed_s": t.elapsed,
    }
    if verbose:
        print(f"[capability] {out['tasks']}")
    return out


def export_public(report_path: str | Path, out_path: str | Path, *, strip_paths: bool = True) -> dict:
    """Copy an eval_report.json to a publishable location, redacted and with local paths shortened."""
    from .schema import load_report

    rep = redact(load_report(report_path), strip_paths=strip_paths)
    problems = validate_report(rep)
    if problems:
        raise ValueError("report is not publishable: " + "; ".join(problems))
    dump_report(rep, out_path)
    return rep


def suggested_slug(report: dict) -> str:
    m = report.get("meta", {})
    base = (m.get("base_model") or "model").split("/")[-1].lower()
    kind = (m.get("target") or {}).get("kind", "target")
    doms = "-".join(m.get("domains") or [])
    return f"{base}__{doms}__{kind}__{m.get('profile', 'profile')}"


__all__ = [
    "ItemsSink",
    "ModelPool",
    "TargetSpec",
    "collect_meta",
    "export_public",
    "run_suite",
    "suggested_slug",
]
