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

from ..i18n import Joined, Text, join_clauses, join_list, join_sentences

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
    """The conclusion, as sentences.

    Every sentence is a :class:`~somtrack.i18n.Text`: an ordinary English
    string everywhere it is used as one (the verdict figure, the command line),
    which can also render itself in Chinese for the translated report and the
    Chinese interface.
    """

    headline: str
    confidence: Confidence = "unknown"
    findings: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    drivers: list[str] = field(default_factory=list)
    separable: list[str] = field(default_factory=list)
    not_separable: list[str] = field(default_factory=list)
    design_note: str = ""

    def text(self) -> Joined:
        """The whole verdict, paragraph by paragraph.

        English when used as a string; ``.render("zh_TW")`` for Chinese.
        """
        parts: list = [self.headline]
        parts += self.findings
        if self.separable:
            parts.append(Text("Group pairs that separate: {pairs}.",
                              pairs=join_clauses(self.separable)))
        if self.not_separable:
            parts.append(Text("Group pairs that do not separate: {pairs}.",
                              pairs=join_clauses(self.not_separable)))
        if self.drivers:
            parts.append(Text("Metrics contributing most to the separation: "
                              "{metrics}.", metrics=join_list(self.drivers)))
        if self.design_note:
            parts.append(self.design_note)
        if self.caveats:
            parts.append(Text("Read with these limits in mind: {caveats}",
                              caveats=join_sentences(self.caveats)))
        return Joined(parts, "\n\n")

    def bullets(self) -> list[str]:
        out = list(self.findings)
        if self.separable:
            out.append(Text("Separates: {pairs}", pairs=join_clauses(self.separable)))
        if self.not_separable:
            out.append(Text("Does not separate: {pairs}",
                            pairs=join_clauses(self.not_separable)))
        if self.drivers:
            out.append(Text("Main drivers: {metrics}", metrics=join_list(self.drivers)))
        return out


# ==========================================================================
def _p_text(p: float, resolution: float = np.nan) -> str:
    """Never quote a p value the permutation count could not have produced."""
    if not np.isfinite(p):
        return Text("p not available")
    if np.isfinite(resolution) and p <= resolution * 1.0001:
        return Text("p = {p:.4f}, which is the floor set by the number of "
                    "permutations rather than a measured value", p=p)
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
    v = Verdict(headline=Text(CONFIDENCE_WORDS[conf] + "."), confidence=conf)

    if n_samples:
        v.findings.append(Text(
            "{n} samples in {k} groups were described by {m} metrics.",
            n=n_samples, k=len(group_values), m=n_features))

    # ---- omnibus ---------------------------------------------------------
    if separation is not None and np.isfinite(separation.permanova_p):
        r2 = separation.permanova_R2
        v.findings.append(Text(
            "PERMANOVA on {metric} distances: pseudo-F = {F:.2f}, R2 = {r2:.3f} "
            "({pct:.1f}% of the multivariate variation is accounted for by the "
            "grouping), {p}, {n} permutations.",
            metric=Text(separation.metric), F=separation.permanova_F, r2=r2,
            pct=r2 * 100,
            p=_p_text(separation.permanova_p, separation.p_resolution),
            n=separation.n_permutations))
        if np.isfinite(separation.permdisp_p):
            v.findings.append(Text(
                "PERMDISP (equality of within-group spread): F = {F:.2f}, {p}.",
                F=separation.permdisp_F, p=_p_text(separation.permdisp_p)))
        v.findings.append(separation.interpretation)
        if separation.dispersion_differs and separation.groups_differ:
            v.caveats.append(Text(
                "Because the groups also differ in spread, the PERMANOVA result "
                "alone cannot distinguish a shift in the average from a change in "
                "variability."))
        if np.isfinite(separation.energy_p) and separation.energy_p < 0.05 \
                and not separation.groups_differ:
            v.findings.append(Text(
                "The energy test found a difference in the distributions that "
                "PERMANOVA did not: the groups may differ in shape rather than in "
                "average or spread."))

    # ---- classification ---------------------------------------------------
    if classification is not None and np.isfinite(classification.balanced_accuracy):
        lo, hi = classification.ba_ci
        ci = (Text(" [95% CI {lo:.2f} to {hi:.2f}]", lo=lo, hi=hi)
              if np.isfinite(lo) else "")
        v.findings.append(Text(
            "Held-out samples were assigned to their group with a balanced "
            "accuracy of {ba:.2f}{ci} against a chance level of {chance:.2f}, "
            "using {model} and {cv} ({p}, {n} label permutations).",
            ba=classification.balanced_accuracy, ci=ci,
            chance=classification.chance, model=Text(classification.model_label),
            cv=classification.cv_name,
            p=_p_text(classification.permutation_p, classification.p_resolution),
            n=classification.n_permutations))
        if classification.cv_note:
            v.findings.append(classification.cv_note)
        v.separable, v.not_separable = _pair_lists(classification)

    # ---- drivers ----------------------------------------------------------
    v.drivers = _drivers(classification, importance)

    # ---- design and caveats ----------------------------------------------
    if structure is not None:
        v.design_note = structure.note
        if not structure.blocked and structure.n_blocks == 0:
            v.caveats.append(Text(
                "No replicate structure was supplied, so every sample was treated "
                "as an independent experimental unit. If several samples came from "
                "one dish, clutch or imaging session, supply that column and "
                "re-run: the p values here would otherwise be too small."))

    for src in (separation, classification):
        for note in getattr(src, "notes", []) or []:
            if note not in v.caveats:
                v.caveats.append(note)

    if supervised_used:
        v.caveats.append(Text(
            "A supervised projection was run. Supervised projections separate the "
            "groups by construction and will do so even on random data, so their "
            "figures are illustrations, not evidence; the cross-validated numbers "
            "above are the evidence."))

    if projections:
        shaky = [p.name for p in projections.values()
                 if np.isfinite(p.dubious_fraction) and p.dubious_fraction > 0.15]
        if shaky:
            v.caveats.append(Text(
                "More than 15% of points are poorly placed in {methods}; read those "
                "panels for broad structure only.", methods=join_list(shaky)))

    if classification is not None and classification.confusion.size:
        per_group = classification.confusion.sum(axis=1)
        tiny = [str(group_values[i]) for i, c in enumerate(per_group)
                if c < 10 and i < len(group_values)]
        if tiny:
            v.caveats.append(Text(
                "Group(s) {groups} have fewer than ten samples; estimates for them "
                "are unstable however small the p value.", groups=join_list(tiny)))

    return v


def _pair_lists(classification) -> tuple[list[str], list[str]]:
    sep: list[str] = []
    non: list[str] = []
    df = getattr(classification, "pairwise", None)
    if df is not None and not df.empty:
        for _, r in df.iterrows():
            label = Text("{a} vs {b} (balanced accuracy {ba:.2f})",
                         a=str(r["group_a"]), b=str(r["group_b"]),
                         ba=r["balanced_accuracy"])
            if r["balanced_accuracy"] >= 0.70:
                sep.append(label)
            elif r["balanced_accuracy"] <= 0.60:
                non.append(label)
        return sep, non

    # two groups only: use the confusion matrix
    conf = getattr(classification, "confusion", None)
    if conf is not None and conf.shape == (2, 2):
        gv = classification.group_values
        label = Text("{a} vs {b} (balanced accuracy {ba:.2f})",
                     a=str(gv[0]), b=str(gv[1]),
                     ba=classification.balanced_accuracy)
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
