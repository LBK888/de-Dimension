"""The projection-method registry.

In 2.0, adding a projection meant editing six places: a config flag, a ``run_*``
function, the job list in ``run_embeddings``, a hard-coded method tuple in the
figure layer, a checkbox in the UI and the matching ``commit``.  Here a method
declares itself once:

    register(MethodSpec(
        key="pacmap", label="PaCMAP", family="manifold",
        params=(ParamSpec("n_neighbors", "int", default=10, tier="common", ...),),
        run=_run_pacmap, available=_has_pacmap,
        citations=("wang2021",),
    ))

and the config, the UI form, the parameter scan and the reference list all pick
it up.  ``ParamSpec`` carries its own UI tier, plain-language help, a
data-driven default and the grid the scan uses, so none of that has to be
duplicated either.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Callable, Literal

import numpy as np

from .projection import DataContext, ProjectionResult

Family = Literal["linear", "manifold", "supervised", "som"]
Tier = Literal["common", "advanced"]

FAMILY_ORDER: tuple[Family, ...] = ("linear", "manifold", "supervised", "som")

FAMILY_LABELS: dict[str, str] = {
    "linear": "Linear projections",
    "manifold": "Non-linear projections",
    "supervised": "Supervised projections",
    "som": "Self-organising map",
}


def has_module(name: str) -> Callable[[], bool]:
    """``available=has_module("pacmap")`` -- an import guard without the import."""
    def check() -> bool:
        import importlib.util

        try:
            return importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            return False
    return check


# ==========================================================================
@dataclass(frozen=True)
class ParamSpec:
    """One tunable parameter of one method."""

    name: str
    kind: Literal["int", "float", "choice", "bool"]
    default: Any
    low: float | None = None
    high: float | None = None
    choices: tuple = ()
    step: float | None = None
    tier: Tier = "advanced"
    label: str = ""                 # UI label; falls back to a prettified name
    help: str = ""                  # one sentence, no jargon
    suggest: Callable[[DataContext], Any] | None = None
    limit: Callable[[DataContext], Any] | None = None   # data-imposed ceiling
    scan_default: tuple = ()        # values used when "scan this" is ticked
    unit: str = ""

    @property
    def display(self) -> str:
        return self.label or self.name.replace("_", " ").capitalize()

    def resolve(self, ctx: DataContext, override: Any = None) -> Any:
        """The value to actually use: explicit override, else data-driven default."""
        if override is not None:
            return override
        if self.suggest is not None:
            return self.suggest(ctx)
        return self.default

    def scan_values(self, ctx: DataContext) -> list:
        """The grid for this parameter, clipped to what the data can support.

        The ceiling matters.  t-SNE requires a perplexity below ``(n - 1) / 3``
        and silently clamps anything larger, so a grid of 5, 10, 30, 50 run on
        ninety samples would produce three identical cells labelled with three
        different values -- a scan figure that appears to show a plateau where
        there is really only one setting.  ``limit`` lets a method declare that
        ceiling once and have the grid respect it.
        """
        vals = list(self.scan_default)
        if not vals:
            return []

        hi = self.high
        if self.limit is not None:
            try:
                data_hi = self.limit(ctx)
                hi = data_hi if hi is None else min(hi, data_hi)
            except Exception:
                pass

        if self.kind == "int":
            lo = int(self.low) if self.low is not None else 1
            top = int(hi) if hi is not None else max(vals)
            vals = [int(min(max(v, lo), top)) for v in vals]
        elif self.kind == "float":
            lo = self.low if self.low is not None else -np.inf
            top = hi if hi is not None else np.inf
            vals = [float(min(max(v, lo), top)) for v in vals]

        seen, out = set(), []
        for v in vals:
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out


@dataclass(frozen=True)
class MethodSpec:
    """One projection method, everything the rest of the program needs to know."""

    key: str
    label: str
    family: Family
    run: Callable[[DataContext, dict], ProjectionResult]
    params: tuple[ParamSpec, ...] = ()
    supervised: bool = False
    needs_groups: bool = False
    available: Callable[[], bool] = lambda: True
    install_hint: str = ""
    citations: tuple[str, ...] = ()
    summary: str = ""               # one line for the method picker
    detail: str = ""                # methods-section clause
    caveat: str = ""                # printed on every figure; required if supervised
    min_samples: int = 3

    def __post_init__(self) -> None:
        if self.supervised and not self.caveat:
            raise ValueError(
                f"Supervised method '{self.key}' must declare a caveat: a supervised "
                f"projection separates groups by construction, and the figure has to "
                f"say so."
            )

    # ------------------------------------------------------------------
    def param(self, name: str) -> ParamSpec | None:
        for p in self.params:
            if p.name == name:
                return p
        return None

    def defaults(self, ctx: DataContext) -> dict:
        return {p.name: p.resolve(ctx) for p in self.params}

    def resolve_params(self, ctx: DataContext, overrides: dict | None = None) -> dict:
        over = overrides or {}
        return {p.name: p.resolve(ctx, over.get(p.name)) for p in self.params}

    def params_by_tier(self, tier: Tier) -> tuple[ParamSpec, ...]:
        return tuple(p for p in self.params if p.tier == tier)

    def usable(self, ctx: DataContext) -> tuple[bool, str]:
        """Can this method run on this data?  Returns (ok, reason-if-not)."""
        if not self.available():
            return False, self.install_hint or f"{self.label} is not installed."
        if self.needs_groups and not ctx.has_groups:
            return False, f"{self.label} needs at least two labelled groups."
        if ctx.n_samples < self.min_samples:
            return False, (f"{self.label} needs at least {self.min_samples} samples "
                           f"(this data set has {ctx.n_samples}).")
        return True, ""


# ==========================================================================
_REGISTRY: dict[str, MethodSpec] = {}


def register(spec: MethodSpec, *, overwrite: bool = False) -> MethodSpec:
    if spec.key in _REGISTRY and not overwrite:
        raise ValueError(f"Projection method '{spec.key}' is already registered.")
    _REGISTRY[spec.key] = spec
    return spec


def get(key: str) -> MethodSpec:
    _ensure_loaded()
    try:
        return _REGISTRY[key]
    except KeyError:
        raise KeyError(
            f"Unknown projection method '{key}'. Available: "
            f"{', '.join(sorted(_REGISTRY))}"
        ) from None


def has(key: str) -> bool:
    _ensure_loaded()
    return key in _REGISTRY


def all_methods(family: Family | None = None,
                available_only: bool = False) -> list[MethodSpec]:
    """Every registered method, ordered by family then registration order."""
    _ensure_loaded()
    specs = list(_REGISTRY.values())
    if family is not None:
        specs = [s for s in specs if s.family == family]
    if available_only:
        specs = [s for s in specs if s.available()]
    order = {f: i for i, f in enumerate(FAMILY_ORDER)}
    return sorted(specs, key=lambda s: (order.get(s.family, 99), s.key))


def by_family(available_only: bool = False) -> dict[str, list[MethodSpec]]:
    out: dict[str, list[MethodSpec]] = {}
    for spec in all_methods(available_only=available_only):
        out.setdefault(spec.family, []).append(spec)
    return out


def default_enabled() -> list[str]:
    """What a fresh install runs: unsupervised methods only, on purpose."""
    return [s.key for s in all_methods(available_only=True)
            if s.family in ("linear", "manifold") and not s.supervised]


# --------------------------------------------------------------------------
_LOADED = False


def _ensure_loaded() -> None:
    """Import the method modules on first use, so registration is automatic."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True                       # set first: the imports call register()
    from .methods import linear, manifold, supervised     # noqa: F401


def run_method(key: str, ctx: DataContext,
               overrides: dict | None = None) -> ProjectionResult:
    """Resolve parameters and run one method, tagging the result with provenance."""
    spec = get(key)
    ok, why = spec.usable(ctx)
    if not ok:
        raise RuntimeError(why)
    params = spec.resolve_params(ctx, overrides)
    res = spec.run(ctx, params)
    return replace(
        res,
        name=res.name or spec.label,
        method_key=spec.key,
        params=params,
        supervised=spec.supervised,
        citations=tuple(res.citations) or spec.citations,
        caveat=res.caveat or spec.caveat,
    )


def clear_registry() -> None:
    """Test hook."""
    global _LOADED
    _REGISTRY.clear()
    _LOADED = False

__all__ = [
    "ParamSpec", "MethodSpec", "Family", "Tier", "has_module",
    "FAMILY_ORDER", "FAMILY_LABELS",
    "register", "get", "has", "all_methods", "by_family", "default_enabled",
    "run_method", "clear_registry",
]
