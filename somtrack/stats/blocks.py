"""Replicate structure: who may be permuted with whom, and who may share a fold.

Every test in this package answers "could this have happened by chance?", and
that question only has a defensible answer once the program knows which samples
are independent.  If eight animals came out of one dish, they are eight
measurements of one experimental unit, not eight replicates (Hurlbert 1984;
Lazic, Clarke-Williams & Munafo 2018).  Ignoring that inflates every p value and
every cross-validated accuracy, because a classifier can reach a good score by
learning which dish a sample came from.

Two designs need different handling, and they are told apart automatically:

*nested*
    every replicate belongs to exactly one group (dish 1 is all control, dish 2
    is all treated).  The replicate is the experimental unit, so a permutation
    reassigns whole replicates to groups, and cross-validation keeps a replicate
    entirely inside one fold.

*crossed* (randomised block)
    every replicate contains several groups (each experiment day ran all
    treatments).  The block is a nuisance factor, so a permutation shuffles
    labels *within* each block, which is what preserves the blocking.

When there is no usable replicate column both reduce to ordinary permutation and
stratified cross-validation, and the report says so rather than implying a
structure that was not there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np

from ..i18n import Text

Design = Literal["none", "nested", "crossed", "mixed"]
CvScheme = Literal["auto", "stratified", "repeated_stratified",
                   "grouped", "leave_one_replicate_out"]


# ==========================================================================
@dataclass
class BlockStructure:
    """What the replicate column implies about independence."""

    design: Design
    blocks: np.ndarray | None          # integer block code per sample, or None
    block_values: list                 # index -> original replicate label
    n_blocks: int
    n_units: int                       # independent experimental units
    note: str                          # one line for the report

    @property
    def blocked(self) -> bool:
        return self.design in ("nested", "crossed", "mixed") and self.blocks is not None

    def describe(self) -> str:
        return self.note


def analyse_blocks(group_codes: np.ndarray,
                   replicates: np.ndarray | None,
                   use_replicates: bool = True) -> BlockStructure:
    """Work out whether the replicate column carries usable structure."""
    n = len(group_codes)
    if not use_replicates or replicates is None:
        return BlockStructure("none", None, [], 0, n,
                              Text("Samples were treated as independent units."))

    rep = np.asarray(replicates)
    values = _ordered_unique(rep)
    if len(values) < 2:
        return BlockStructure("none", None, [], 0, n,
                              Text("The replicate column held a single value, so "
                                   "samples were treated as independent units."))

    lookup = {v: i for i, v in enumerate(values)}
    blocks = np.array([lookup[v] for v in rep.tolist()], dtype=int)

    groups_per_block = [np.unique(group_codes[blocks == b]).size
                        for b in range(len(values))]
    pure = [g == 1 for g in groups_per_block]

    if all(pure):
        design: Design = "nested"
        note = Text("Replicates were nested within groups ({n} replicates, each "
                    "belonging to one group), so the replicate was treated as the "
                    "experimental unit.", n=len(values))
        units = len(values)
    elif not any(pure):
        design = "crossed"
        note = Text("Every replicate contained several groups ({n} blocks), so the "
                    "design was treated as randomised blocks and labels were "
                    "permuted within blocks.", n=len(values))
        units = n
    else:
        design = "mixed"
        note = Text("Replicates were partly nested and partly crossed with the "
                    "groups ({n} replicates); labels were permuted within blocks "
                    "where possible and whole replicates were kept inside one "
                    "cross-validation fold.", n=len(values))
        units = n

    return BlockStructure(design, blocks, list(values), len(values), units, note)


def _ordered_unique(a: np.ndarray) -> list:
    seen: set = set()
    out: list = []
    for v in a.tolist():
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


# ==========================================================================
# Permutation
# ==========================================================================
def permutations(y: np.ndarray, structure: BlockStructure, n: int,
                 random_state: int | None = 0) -> Iterator[np.ndarray]:
    """Yield ``n`` label vectors drawn from the null that respects the design.

    Nested designs reassign whole replicates; crossed designs shuffle within
    each block; an unblocked design shuffles freely (Anderson & ter Braak 2003).
    """
    rng = np.random.default_rng(random_state)
    y = np.asarray(y)

    if not structure.blocked:
        for _ in range(n):
            yield rng.permutation(y)
        return

    blocks = structure.blocks
    if structure.design == "nested":
        uniq = np.unique(blocks)
        block_label = np.array([y[blocks == b][0] for b in uniq])
        for _ in range(n):
            shuffled = rng.permutation(block_label)
            out = np.empty_like(y)
            for b, lab in zip(uniq, shuffled):
                out[blocks == b] = lab
            yield out
        return

    # crossed / mixed: shuffle labels inside each block
    idx_by_block = [np.flatnonzero(blocks == b) for b in np.unique(blocks)]
    for _ in range(n):
        out = y.copy()
        for idx in idx_by_block:
            if idx.size > 1:
                out[idx] = rng.permutation(y[idx])
        yield out


def effective_permutations(y: np.ndarray, structure: BlockStructure) -> int:
    """How many distinct label assignments the design actually allows.

    A nested design with four replicates has at most 4! = 24 arrangements, so a
    p value cannot be smaller than 1/24 however many permutations are drawn.
    Reporting that ceiling stops an impossible ``p < 0.001`` being quoted.
    """
    from math import factorial

    if not structure.blocked:
        n = len(y)
        return factorial(min(n, 12)) if n <= 12 else 10 ** 9

    if structure.design == "nested":
        k = structure.n_blocks
        return factorial(k) if k <= 12 else 10 ** 9

    total = 1
    for b in np.unique(structure.blocks):
        m = int((structure.blocks == b).sum())
        total *= factorial(m) if m <= 8 else 10 ** 6
        if total > 10 ** 9:
            return 10 ** 9
    return total


def permutation_p(observed: float, null: np.ndarray, greater_is_extreme: bool = True) -> float:
    """The (B + 1) form, which never returns exactly zero."""
    null = np.asarray(null, float)
    null = null[np.isfinite(null)]
    if null.size == 0 or not np.isfinite(observed):
        return float("nan")
    hits = (null >= observed).sum() if greater_is_extreme else (null <= observed).sum()
    return float((1.0 + hits) / (1.0 + null.size))


# ==========================================================================
# Cross-validation
# ==========================================================================
@dataclass
class CvPlan:
    splitter: object
    name: str                        # for the methods section
    n_splits: int
    uses_blocks: bool
    groups: np.ndarray | None        # the `groups` argument for splitter.split
    note: str


def make_cv(y: np.ndarray, structure: BlockStructure, scheme: CvScheme = "auto",
            n_splits: int = 5, repeats: int = 5,
            random_state: int | None = 0) -> CvPlan:
    """Choose a cross-validation scheme that cannot leak replicate identity."""
    from sklearn.model_selection import (LeaveOneGroupOut, RepeatedStratifiedKFold,
                                         StratifiedKFold)

    counts = np.bincount(np.asarray(y, int))
    counts = counts[counts > 0]
    smallest = int(counts.min()) if counts.size else 0
    k = int(np.clip(min(n_splits, smallest), 2, max(2, smallest)))

    can_block = structure.blocked and structure.n_blocks >= 3
    if scheme == "auto":
        scheme = "grouped" if can_block else "repeated_stratified"

    if scheme in ("grouped", "leave_one_replicate_out") and not can_block:
        scheme = "repeated_stratified"

    if scheme == "leave_one_replicate_out":
        return CvPlan(LeaveOneGroupOut(),
                      Text("leave-one-replicate-out cross-validation"),
                      structure.n_blocks, True, structure.blocks,
                      Text("Each fold held out one whole replicate."))

    if scheme == "grouped":
        try:
            from sklearn.model_selection import StratifiedGroupKFold

            kk = int(np.clip(min(k, structure.n_blocks), 2, structure.n_blocks))
            return CvPlan(
                StratifiedGroupKFold(n_splits=kk, shuffle=True,
                                     random_state=random_state),
                Text("{k}-fold cross-validation with whole replicates held out "
                     "together", k=kk),
                kk, True, structure.blocks,
                Text("No replicate appeared in both the training and the test set, "
                     "so the score cannot come from recognising a replicate."))
        except ImportError:                                   # scikit-learn < 1.0
            from sklearn.model_selection import GroupKFold

            kk = int(np.clip(min(k, structure.n_blocks), 2, structure.n_blocks))
            return CvPlan(GroupKFold(n_splits=kk),
                          Text("{k}-fold grouped cross-validation", k=kk), kk, True,
                          structure.blocks,
                          Text("No replicate appeared in both training and test sets."))

    if scheme == "stratified":
        return CvPlan(StratifiedKFold(n_splits=k, shuffle=True,
                                      random_state=random_state),
                      Text("{k}-fold stratified cross-validation", k=k), k, False, None,
                      Text("Folds preserved the group proportions."))

    return CvPlan(
        RepeatedStratifiedKFold(n_splits=k, n_repeats=repeats,
                                random_state=random_state),
        Text("{k}-fold stratified cross-validation repeated {repeats} times",
             k=k, repeats=repeats),
        k * repeats, False, None,
        Text("Folds preserved the group proportions; repeating the split reduces "
             "the influence of any one partition."))


# ==========================================================================
def aggregate_to_units(X: np.ndarray, y: np.ndarray, structure: BlockStructure
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse to one row per replicate -- the conservative analysis.

    Returns ``(X_unit, y_unit, block_codes)``.  Only meaningful for a nested
    design, where averaging inside a replicate removes the pseudoreplication
    entirely at the cost of statistical power.
    """
    if not structure.blocked:
        return X, y, np.arange(len(y))

    rows, labels, codes = [], [], []
    for b in np.unique(structure.blocks):
        m = structure.blocks == b
        rows.append(X[m].mean(axis=0))
        vals, cnt = np.unique(y[m], return_counts=True)
        labels.append(vals[np.argmax(cnt)])
        codes.append(b)
    return np.asarray(rows), np.asarray(labels), np.asarray(codes)


__all__ = [
    "BlockStructure", "analyse_blocks", "Design", "CvScheme",
    "permutations", "effective_permutations", "permutation_p",
    "CvPlan", "make_cv", "aggregate_to_units",
]
