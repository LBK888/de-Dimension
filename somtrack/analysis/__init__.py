"""Projections: the registry, the runner, the quality layer and the scan.

    from somtrack.analysis import DataContext, run_projections

    ctx = DataContext.from_prepared(prep)
    results = run_projections(ctx, ["pca", "pacmap", "lda"])

Adding a method is one file in :mod:`somtrack.analysis.methods` and one
``register()`` call; the configuration, the user interface, the parameter scan,
the comparison figures and the reference list all pick it up from there.
"""

from __future__ import annotations

import warnings
from typing import Callable, Iterable

import numpy as np

from .projection import DataContext, ProjectionResult, axis_correlation, two_columns
from .quality import (Reliability, Stability, neighbour_agreement,
                      point_reliability, rank_quality, seed_stability)
from .registry import (FAMILY_LABELS, FAMILY_ORDER, MethodSpec, ParamSpec,
                       all_methods, by_family, default_enabled, get, has,
                       has_module, register, run_method)

Progress = Callable[[int, int], None]


# ==========================================================================
def evaluate(result: ProjectionResult, ctx: DataContext, *,
             rank: bool = True, reliability: bool = True,
             k: int = 0, n_null: int = 20) -> ProjectionResult:
    """Attach the neighbourhood-preservation diagnostics to a raw projection."""
    Y = result.coords
    if rank:
        try:
            q = rank_quality(ctx.X, Y, k=k)
            result.trustworthiness = q.trustworthiness
            result.continuity = q.continuity
            result.rnx = q.rnx
            result.rnx_auc = q.auc
        except Exception as exc:                              # pragma: no cover
            warnings.warn(f"Quality metrics failed for {result.name}: {exc}")
    if reliability:
        try:
            r = point_reliability(ctx.X, Y, n_null=n_null,
                                  random_state=ctx.random_state)
            result.reliability = r.score
            result.dubious = r.dubious
        except Exception as exc:                              # pragma: no cover
            warnings.warn(f"Reliability failed for {result.name}: {exc}")
    return result


def run_projections(
    ctx: DataContext,
    enabled: Iterable[str] | None = None,
    overrides: dict[str, dict] | None = None,
    *,
    rank: bool = True,
    reliability: bool = True,
    reliability_null: int = 20,
    stability_seeds: int = 0,
    log=None,
    progress: Progress | None = None,
    warn=None,
) -> dict[str, ProjectionResult]:
    """Run each requested method, evaluate it, and record it in the methods log.

    A method that is not installed, or that cannot run on this data, is skipped
    with an explanation rather than failing the run; the explanation reaches the
    report, so a missing UMAP shows up as a stated omission instead of a silently
    shorter figure set.
    """
    keys = list(enabled) if enabled is not None else default_enabled()
    overrides = overrides or {}
    out: dict[str, ProjectionResult] = {}

    for i, key in enumerate(keys):
        if progress:
            progress(i, len(keys))
        try:
            spec = get(key)
        except KeyError as exc:
            _warn(warn, str(exc))
            continue

        ok, why = spec.usable(ctx)
        if not ok:
            _warn(warn, why)
            continue

        try:
            res = run_method(key, ctx, overrides.get(key))
        except Exception as exc:
            _warn(warn, f"{spec.label} failed and was skipped: {exc}")
            continue

        evaluate(res, ctx, rank=rank, reliability=reliability,
                 n_null=reliability_null)

        if stability_seeds > 1:
            res.stability = _stability(spec, ctx, overrides.get(key),
                                       stability_seeds)

        out[key] = res
        if log is not None:
            log.record(
                "Projection", spec.label,
                spec.detail or spec.summary,
                citations=res.citations,
                caveat=res.caveat,
                **{k: v for k, v in res.params.items() if v is not None},
            )

    if progress:
        progress(len(keys), len(keys))
    return out


def _stability(spec: MethodSpec, ctx: DataContext, override: dict | None,
               n_seeds: int) -> np.ndarray | None:
    """Re-run with different seeds and measure how far each point moves."""
    runs = []
    base = ctx.random_state or 0
    for s in range(n_seeds):
        alt = DataContext(**{**ctx.__dict__, "random_state": base + s})
        try:
            runs.append(run_method(spec.key, alt, override).coords)
        except Exception:
            continue
    if len(runs) < 2:
        return None
    return seed_stability(runs).per_point


def _warn(warn, message: str) -> None:
    if warn is not None:
        warn(message)
    else:
        warnings.warn(message)


def quality_table(results: dict[str, ProjectionResult]):
    """One row per projection, for the method-comparison panel and the report."""
    import pandas as pd

    return pd.DataFrame([r.quality_row() for r in results.values()])


__all__ = [
    "DataContext", "ProjectionResult", "axis_correlation", "two_columns",
    "MethodSpec", "ParamSpec", "register", "get", "has", "has_module",
    "all_methods", "by_family", "default_enabled", "run_method",
    "FAMILY_ORDER", "FAMILY_LABELS",
    "rank_quality", "point_reliability", "seed_stability",
    "neighbour_agreement", "Reliability", "Stability",
    "evaluate", "run_projections", "quality_table",
]
