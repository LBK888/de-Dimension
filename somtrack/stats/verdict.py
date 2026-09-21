"""The paragraph at the top of the report.

Everything else in this package produces numbers.  This module produces the
sentence a reader who is not a statistician needs in order to act on them: what
the analysis found, how strongly, what it does *not* say, and which caveats
attach to the figures that follow.

It is generated, never written by hand, so it cannot drift away from the numbers
it describes.  Where the evidence is weak it says so; where a significant result
has an alternative reading -- a dispersion difference rather than a shift -- it
gives that reading rather than leaving it for the reader to notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Confidence = Literal["strong", "moderate", "weak", "none", "unknown"]

CONFIDENCE_WORDS: dict[str, str] = {
    "strong": "Strong evidence that the groups differ",
    "moderate": "Moderate evidence that the groups differ",
    "weak": "Weak or borderline evidence that the groups differ",
    "none": "No detectable difference between the groups",
    "unknown": "The evidence could not be assessed",
}


@dataclass
class Verdict:
    headline: str
    confidence: Confidence = "unknown"
    findings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    separable: list[str] = field(default_factory=list)
    not_separable: list[str] = field(default_factory=list)
    design_note: str = ""

    def text(self) -> str:
        parts = [self.headline]
        parts += self.findings
        if self.separable:
            parts.append("Group pairs that separate: " + "; ".join(self.separable) + ".")
        if self.not_separable:
            parts.append("Group pairs that do not separate: "
                         + "; ".join(self.not_separable) + ".")
        if self.drivers:
            parts.append("Metrics contributing most to the separation: "
                         + ", ".join(self.drivers) + ".")
        if self.design_note:
            parts.append(self.design_note)
        if self.caveats:
            parts.append("Read with these limits in mind: "
                         + " ".join(self.caveats))
        return "\n\n".join(parts)

    def bullets(self) -> list[str]:
        out = list(self.findings)
        if self.separable:
            out.append("Separates: " + "; ".join(self.separable))
        if self.not_separable:
            out.append("Does not separate: " + "; ".join(self.not_separable))
        if self.drivers:
            out.append("Main drivers: " + ", ".join(self.drivers))
        return out


# ==========================================================================
def _p_text(p: float, resolution: float = np.nan) -> str:
    """Never quote a p value the permutation count could not have produced."""
    if not np.isfinite(p):
        return "p not available"
    if np.isfinite(resolution) and p <= resolution * 1.0001:
        return (f"p = {p:.4f}, which is the floor set by the number of "
                f"permutations rather than a measured value")
    if p < 0.001:
        return "p < 0.001"
    return f"p = {p:.3f}"


def _confidence(separation, classification) -> Confidence:
    sig = separation is not None and separation.groups_differ
    if classification is not None and np.isfinite(classification.permutation_p):
        cls_sig = classification.permutation_p < 0.05
        margin = classification.balanced_accuracy - classification.chance
        if sig and cls_sig and margin > 0.20:
            return "strong"
        if sig and cls_sig:
            return "moderate"
        if sig or cls_sig:
            return "weak"
        return "none"
    if sig:
        return "moderate" if (separation.permanova_R2 or 0) > 0.10 else "weak"
    if separation is not None and np.isfinite(separation.permanova_p):
        return "none"
    return "unknown"


def build_verdict(
    separation=None,
    classification=None,
    importance=None,
    structure=None,
    projections: dict | None = None,
    n_samples: int = 0,
    n_features: int = 0,
    group_values: list | None = None,
    supervised_used: bool = False,
) -> Verdict:
    """Assemble the plain-language summary from whatever analyses were run."""
    group_values = list(group_values or [])
    conf = _confidence(separation, classification)
    v = Verdict(headline=CONFIDENCE_WORDS[conf] + ".", confidence=conf)

    if n_samples:
        v.findings.append(
            f"{n_samples} samples in {len(group_values)} groups were described by "
            f"{n_features} metrics.")

    # ---- omnibus ---------------------------------------------------------
    if separation is not None and np.isfinite(separation.permanova_p):
        r2 = separation.permanova_R2
        v.findings.append(
            f"PERMANOVA on {separation.metric} distances: pseudo-F = "
            f"{separation.permanova_F:.2f}, R2 = {r2:.3f} "
            f"({r2 * 100:.1f}% of the multivariate variation is accounted for by "
            f"the grouping), "
            f"{_p_text(separation.permanova_p, separation.p_resolution)}, "
            f"{separation.n_permutations} permutations.")
        if np.isfinite(separation.permdisp_p):
            v.findings.append(
                f"PERMDISP (equality of within-group spread): F = "
                f"{separation.permdisp_F:.2f}, {_p_text(separation.permdisp_p)}.")
        v.findings.append(separation.interpretation)
        if separation.dispersion_differs and separation.groups_differ:
            v.caveats.append(
                "Because the groups also differ in spread, the PERMANOVA result "
                "alone cannot distinguish a shift in the average from a change in "
                "variability.")
        if np.isfinite(separation.energy_p) and separation.energy_p < 0.05 \
                and not separation.groups_differ:
            v.findings.append(
                "The energy test found a difference in the distributions that "
                "PERMANOVA did not: the groups may differ in shape rather than in "
                "average or spread.")

    # ---- classification ---------------------------------------------------
    if classification is not None and np.isfinite(classification.balanced_accuracy):
        lo, hi = classification.ba_ci
        ci = f" [95% CI {lo:.2f} to {hi:.2f}]" if np.isfinite(lo) else ""
        v.findings.append(
            f"Held-out samples were assigned to their group with a balanced "
            f"accuracy of {classification.balanced_accuracy:.2f}{ci} against a "
            f"chance level of {classification.chance:.2f}, using "
            f"{classification.model_label} and {classification.cv_name} "
            f"({_p_text(classification.permutation_p, classification.p_resolution)}, "
            f"{classification.n_permutations} label permutations).")
        if classification.cv_note:
            v.findings.append(classification.cv_note)
        v.separable, v.not_separable = _pair_lists(classification)

    # ---- drivers ----------------------------------------------------------
    v.drivers = _drivers(classification, importance)

    # ---- design and caveats ----------------------------------------------
    if structure is not None:
        v.design_note = structure.note
        if not structure.blocked and structure.n_blocks == 0:
            v.caveats.append(
                "No replicate structure was supplied, so every sample was treated "
                "as an independent experimental unit. If several samples came from "
                "one dish, clutch or imaging session, supply that column and "
                "re-run: the p values here would otherwise be too small.")

    for src in (separation, classification):
        for note in getattr(src, "notes", []) or []:
            if note not in v.caveats:
                v.caveats.append(note)

    if supervised_used:
        v.caveats.append(
            "A supervised projection was run. Supervised projections separate the "
            "groups by construction and will do so even on random data, so their "
            "figures are illustrations, not evidence; the cross-validated numbers "
            "above are the evidence.")

    if projections:
        shaky = [p.name for p in projections.values()
                 if np.isfinite(p.dubious_fraction) and p.dubious_fraction > 0.15]
        if shaky:
            v.caveats.append(
                f"More than 15% of points are poorly placed in {', '.join(shaky)}; "
                f"read those panels for broad structure only.")

    if classification is not None and classification.confusion.size:
        per_group = classification.confusion.sum(axis=1)
        tiny = [str(group_values[i]) for i, c in enumerate(per_group)
                if c < 10 and i < len(group_values)]
        if tiny:
            v.caveats.append(
                f"Group(s) {', '.join(tiny)} have fewer than ten samples; estimates "
                f"for them are unstable however small the p value.")

    return v


def _pair_lists(classification) -> tuple[list[str], list[str]]:
    sep: list[str] = []
    non: list[str] = []
    df = getattr(classification, "pairwise", None)
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            label = (f"{r['group_a']} vs {r['group_b']} "
                     f"(balanced accuracy {r['balanced_accuracy']:.2f})")
            if r["balanced_accuracy"] >= 0.70:
                sep.append(label)
            elif r["balanced_accuracy"] <= 0.60:
                non.append(label)
        return sep, non

    # two groups only: use the confusion matrix
    conf = getattr(classification, "confusion", None)
    if conf is not None and conf.shape == (2, 2):
        gv = classification.group_values
        label = (f"{gv[0]} vs {gv[1]} "
                 f"(balanced accuracy {classification.balanced_accuracy:.2f})")
        (sep if classification.separable else non).append(label)
    return sep, non


def _drivers(classification, importance, n: int = 5) -> list[str]:
    names: list[str] = []
    if classification is not None and classification.activation is not None \
            and classification.feature_names:
        A = np.atleast_2d(classification.activation)
        if A.shape[1] == len(classification.feature_names):
            strength = np.linalg.norm(A, axis=0)
            order = np.argsort(-strength)[:n]
            names = [classification.feature_names[i] for i in order]
    if not names and importance is not None and not importance.table.empty:
        names = importance.table.head(n)["feature"].tolist()
    return names


__all__ = ["Verdict", "build_verdict", "Confidence", "CONFIDENCE_WORDS"]
