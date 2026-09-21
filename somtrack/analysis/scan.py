"""Parameter scans, for any method, with the plateau made visible.

The 2.0 scan was tied to the SOM and threw the embeddings away, keeping only
quality numbers.  This one runs over any registered method, keeps every layout
it produced, and scores each cell on whichever criterion the caller names.  That
makes two things possible that a table of numbers cannot do:

*   the scan can be *drawn* -- a grid of the actual projections, so the reader
    sees that the structure is the same across a range of settings rather than
    taking it on trust;
*   the **plateau** can be reported.  The single best cell of a scan is almost
    never meaningfully better than its neighbours, and quoting it alone invites
    the reader to believe the choice was critical.  Reporting "any value between
    10 and 25 gives the same answer" is both more honest and more useful, and it
    is what stops a user tuning until the picture looks the way they hoped.

Scoring criteria available without labels are ``rnx_auc`` (higher is better) and
``dubious_fraction`` (lower is better, in the style of scDEED's hyper-parameter
selection).  With labels, ``cv_balanced_accuracy`` and ``group_silhouette`` are
also available -- but note that choosing a projection by how well it separates
known groups is a supervised choice, and the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

from ..i18n import Text, join_list
from .projection import DataContext
from .registry import get, run_method

# criterion -> (column, higher_is_better, needs_labels, description)
CRITERIA: dict[str, tuple[str, bool, bool, str]] = {
    "rnx_auc": ("rnx_auc", True, False,
                "neighbourhood preservation across all scales"),
    "dubious_fraction": ("dubious_fraction", False, False,
                         "fraction of points the projection places unreliably"),
    "trustworthiness": ("trustworthiness", True, False,
                        "absence of invented neighbours"),
    "group_silhouette": ("group_silhouette", True, True,
                         "how cleanly the known groups sit apart in the projection"),
}


# ==========================================================================
@dataclass
class ScanResult:
    method: str
    method_label: str
    grid: dict[str, list]
    rows: pd.DataFrame
    coords: dict[tuple, np.ndarray] = field(default_factory=dict)
    criterion: str = "rnx_auc"
    higher_is_better: bool = True
    best: dict = field(default_factory=dict)
    plateau: pd.DataFrame = field(default_factory=pd.DataFrame)
    tolerance: float = 0.02
    citations: tuple[str, ...] = ()
    note: str = ""

    @property
    def keys(self) -> list[str]:
        return list(self.grid)

    @property
    def n_cells(self) -> int:
        return len(self.rows)

    def best_params(self) -> dict:
        return {k: self.best[k] for k in self.grid if k in self.best}

    def plateau_text(self) -> str:
        """The sentence that goes under the scan figure."""
        if self.plateau.empty or len(self.plateau) <= 1:
            return Text("The best setting was {best}; no other setting came within "
                        "{tol:.0%} of it.", best=_fmt(self.best_params()),
                        tol=self.tolerance)
        bits = []
        for k in self.grid:
            vals = sorted(set(self.plateau[k].tolist()))
            if len(vals) == 1:
                bits.append(f"{k} = {_fmt_one(vals[0])}")
            else:
                bits.append(Text("{param} from {lo} to {hi}", param=k,
                                 lo=_fmt_one(vals[0]), hi=_fmt_one(vals[-1])))
        return Text("{n} of {total} settings scored within {tol:.0%} of the best "
                    "({ranges}), so the result does not depend on the exact choice. "
                    "{best} was used.", n=len(self.plateau), total=self.n_cells,
                    tol=self.tolerance, ranges=join_list(bits),
                    best=_fmt(self.best_params()))


def _fmt_one(v) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def _fmt(d: dict) -> str:
    return ", ".join(f"{k} = {_fmt_one(v)}" for k, v in d.items())


# ==========================================================================
def default_grid(method: str, ctx: DataContext,
                 params: list[str] | None = None) -> dict[str, list]:
    """The grid a method declares for itself, via ``ParamSpec.scan_default``."""
    spec = get(method)
    grid: dict[str, list] = {}
    for p in spec.params:
        if params is not None and p.name not in params:
            continue
        vals = p.scan_values(ctx)
        if vals:
            grid[p.name] = vals
    return grid


def scan(
    ctx: DataContext,
    method: str,
    grid: dict[str, list] | None = None,
    *,
    criterion: str = "rnx_auc",
    tolerance: float = 0.02,
    reliability_null: int = 10,
    n_jobs: int = -1,
    log=None,
    progress=None,
) -> ScanResult:
    """Run one method over a parameter grid and score every cell."""
    from joblib import Parallel, delayed

    from . import evaluate

    spec = get(method)
    ok, why = spec.usable(ctx)
    if not ok:
        raise RuntimeError(why)

    grid = grid or default_grid(method, ctx)
    if not grid:
        raise ValueError(
            f"{spec.label} declares no scannable parameters. Pass a grid explicitly.")

    column, higher, needs_labels, description = CRITERIA.get(
        criterion, CRITERIA["rnx_auc"])
    if needs_labels and not ctx.has_groups:
        criterion = "rnx_auc"
        column, higher, needs_labels, description = CRITERIA["rnx_auc"]

    keys = list(grid)
    combos = list(product(*(grid[k] for k in keys)))

    def one(values):
        params = dict(zip(keys, values))
        try:
            res = run_method(method, ctx, params)
            evaluate(res, ctx, n_null=reliability_null)
        except Exception as exc:
            return params, None, {"error": str(exc)}
        row = res.quality_row()
        row.update(params)
        if ctx.has_groups:
            row["group_silhouette"] = _silhouette(res.coords, ctx.group_codes)
        return params, res.coords, row

    if progress:
        progress(0, len(combos))
    out = Parallel(n_jobs=n_jobs, prefer="processes")(
        delayed(one)(c) for c in combos)
    if progress:
        progress(len(combos), len(combos))

    rows, coords = [], {}
    for params, xy, row in out:
        rows.append(row)
        if xy is not None:
            coords[tuple(params[k] for k in keys)] = xy
    frame = pd.DataFrame(rows)

    result = ScanResult(method=method, method_label=spec.label, grid=grid,
                        rows=frame, coords=coords, criterion=criterion,
                        higher_is_better=higher, tolerance=tolerance,
                        citations=spec.citations,
                        note=Text("Settings were scored by {criterion}.",
                                  criterion=Text(description)))

    if column in frame and frame[column].notna().any():
        vals = frame[column].to_numpy(float)
        idx = int(np.nanargmax(vals) if higher else np.nanargmin(vals))
        best_val = vals[idx]
        # "within 2%" should mean 2% of the score, not 2% of whatever range the
        # grid happened to span: on a flat landscape the range collapses and a
        # range-relative band would report no plateau exactly when the plateau
        # is widest.  The range is kept only as a floor for scores near zero.
        span = float(np.nanmax(vals) - np.nanmin(vals))
        band = tolerance * max(abs(best_val), span, 1e-12)
        near = (vals >= best_val - band) if higher else (vals <= best_val + band)
        result.plateau = frame.loc[near & np.isfinite(vals)].copy()
        # Within a plateau the scan has not learned anything, so it should not
        # pretend it has: fall back to the parameter the method would have
        # chosen unaided, rather than to whichever cell noise put on top.
        result.best = frame.iloc[_pick_within_plateau(spec, ctx, keys, frame,
                                                      near, idx)].to_dict()

    if log is not None:
        log.record(
            "Projection", Text("{method} parameter scan", method=spec.label),
            Text("{n} settings scored by {criterion}", n=len(combos),
                 criterion=Text(description)),
            citations=spec.citations + (("xia2024",) if criterion == "dubious_fraction"
                                        else ()),
            **{f"scanned_{k}": f"{min(v)}-{max(v)}" if len(v) > 1 else v[0]
               for k, v in grid.items()},
        )
    return result


def _pick_within_plateau(spec, ctx: DataContext, keys: list[str],
                         frame: pd.DataFrame, near: np.ndarray, fallback: int) -> int:
    """Index of the plateau cell closest to the method's own suggested settings."""
    idx = np.flatnonzero(near)
    if idx.size <= 1:
        return fallback
    target = {}
    for k in keys:
        p = spec.param(k)
        if p is not None:
            target[k] = p.resolve(ctx)
    if not target:
        return int(idx[len(idx) // 2])

    def distance(i: int) -> float:
        d = 0.0
        for k, want in target.items():
            got = frame.iloc[i][k]
            try:
                scale = max(abs(float(want)), 1e-9)
                d += (abs(float(got) - float(want)) / scale) ** 2
            except (TypeError, ValueError):
                d += 0.0 if got == want else 1.0
        return d

    return int(min(idx, key=distance))


def _silhouette(Y: np.ndarray, y: np.ndarray) -> float:
    from sklearn.metrics import silhouette_score

    try:
        if len(np.unique(y)) < 2:
            return np.nan
        return float(silhouette_score(Y, y))
    except Exception:
        return np.nan


# ==========================================================================
def scan_som(X: np.ndarray, base, grid: dict[str, list],
             group_codes=None, n_groups: int = 0, n_jobs: int = -1,
             progress=None, log=None) -> ScanResult:
    """The SOM scan, wrapped in the same result type as the projection scans."""
    from ..som import scan_parameters

    rows = scan_parameters(X, base, grid, group_codes, n_groups, n_jobs, progress)
    frame = pd.DataFrame(rows)
    result = ScanResult(method="som", method_label="Self-organising map", grid=grid,
                        rows=frame, criterion="quantisation_error",
                        higher_is_better=False, citations=("kohonen2001",),
                        note=Text("Settings were scored by quantisation error."))
    if "quantisation_error" in frame and frame["quantisation_error"].notna().any():
        vals = frame["quantisation_error"].to_numpy(float)
        idx = int(np.nanargmin(vals))
        result.best = frame.iloc[idx].to_dict()
        span = np.nanmax(vals) - np.nanmin(vals)
        band = result.tolerance * (span if span > 0 else 1.0)
        result.plateau = frame.loc[(vals <= vals[idx] + band)
                                   & np.isfinite(vals)].copy()
    if log is not None:
        log.record("Clustering", "SOM parameter scan",
                   Text("{n} settings compared by map quality", n=len(frame)),
                   citations=("kohonen2001",))
    return result


__all__ = ["ScanResult", "scan", "scan_som", "default_grid", "CRITERIA"]
