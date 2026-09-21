"""The conclusion report: what was found, how it was done, and what to cite.

Produced automatically at the end of every run, in Markdown and HTML, alongside
the figures and tables.  It exists so that the step between "the analysis
finished" and "I can start writing the paper" is a copy and paste rather than an
afternoon of reconstructing which settings were used.

Four things it guarantees:

*   the **conclusion** is generated from the numbers, so it cannot disagree with
    the tables underneath it;
*   the **methods** section describes what this run did, assembled from
    :class:`somtrack.citations.MethodsLog` as each step recorded itself.  A run
    that skipped UMAP does not mention UMAP; a run that used a supervised
    projection carries its caveat;
*   every algorithm named in the methods section is followed by its citation,
    in the form ``SuperPlots (Lord et al., 2020)``, and every citation used
    appears once in the **reference list** with a DOI;
*   the **caveats** travel with the results instead of being left for a reviewer
    to find.
"""

from __future__ import annotations

import html as _html
import platform
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .citations import BASE_SOFTWARE, Citation, MethodsLog, reference_list


# ==========================================================================
@dataclass
class ReportSection:
    title: str
    body: list[str] = field(default_factory=list)
    tables: list[tuple[str, pd.DataFrame]] = field(default_factory=list)

    def add(self, text: str) -> None:
        if text:
            self.body.append(text)

    def add_table(self, caption: str, df: pd.DataFrame, limit: int = 20) -> None:
        if df is not None and not df.empty:
            self.tables.append((caption, df.head(limit)))


@dataclass
class Report:
    title: str
    generated: str
    sections: list[ReportSection] = field(default_factory=list)
    methods: MethodsLog = field(default_factory=MethodsLog)
    references: list[Citation] = field(default_factory=list)
    figures: list[tuple[str, str]] = field(default_factory=list)   # (name, caption)
    caveats: list[str] = field(default_factory=list)
    verdict_text: str = ""
    confidence: str = "unknown"
    provenance: dict = field(default_factory=dict)

    def section(self, title: str) -> ReportSection:
        for s in self.sections:
            if s.title == title:
                return s
        s = ReportSection(title)
        self.sections.append(s)
        return s


# ==========================================================================
# Assembly
# ==========================================================================
def build_report(result, log: MethodsLog | None = None) -> Report:
    """Turn an :class:`somtrack.pipeline.AnalysisResult` into a written report."""
    cfg = result.config
    log = log or getattr(result, "methods_log", None) or MethodsLog()
    prep = result.prep

    title = cfg.report.title or "Multivariate analysis of locomotion metrics"
    rep = Report(title=title,
                 generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                 methods=log)

    verdict = getattr(result, "verdict", None)
    if verdict is not None:
        rep.verdict_text = verdict.text()
        rep.confidence = verdict.confidence
        rep.caveats = list(verdict.caveats)

    _data_section(rep, result, prep)
    _separation_section(rep, result)
    _classification_section(rep, result)
    _drivers_section(rep, result)
    _projection_section(rep, result)
    _som_section(rep, result)

    log.record("Software", "Python analysis stack",
               "NumPy, SciPy, scikit-learn, pandas and matplotlib",
               citations=BASE_SOFTWARE)

    rep.references = reference_list(log.citation_keys())
    rep.provenance = {
        "SOMTrack version": _version(),
        "Python": platform.python_version(),
        "Platform": platform.platform(),
        "Random seed": cfg.som.random_state,
        "Config file": "analysis_config.json",
        "Generated": rep.generated,
    }
    return rep


def _version() -> str:
    from . import __version__

    return __version__


def _data_section(rep: Report, result, prep) -> None:
    s = rep.section("Data")
    counts = ", ".join(
        f"{lab} (n = {int((prep.group_codes == g).sum())})"
        for g, lab in enumerate(prep.group_values))
    s.add(f"{prep.n_samples} samples described by {prep.n_features} metrics, "
          f"in {prep.n_groups} groups: {counts}.")
    structure = getattr(result, "block_structure", None)
    if structure is not None:
        s.add(structure.note)
    if prep.dropped_features:
        s.add(f"{len(prep.dropped_features)} metrics were dropped before analysis "
              f"({', '.join(prep.dropped_features[:10])}"
              f"{'...' if len(prep.dropped_features) > 10 else ''}).")
    for note in prep.notes:
        s.add(note)


def _separation_section(rep: Report, result) -> None:
    sep = getattr(result, "separation", None)
    if sep is None or not np.isfinite(sep.permanova_p):
        return
    s = rep.section("Do the groups differ?")
    s.add(f"PERMANOVA on {sep.metric} distances gave pseudo-F = "
          f"{sep.permanova_F:.2f} with R2 = {sep.permanova_R2:.3f} and "
          f"p = {sep.permanova_p:.4f} over {sep.n_permutations} permutations. "
          f"The grouping therefore accounts for "
          f"{sep.permanova_R2 * 100:.1f}% of the multivariate variation.")
    if np.isfinite(sep.permdisp_p):
        s.add(f"PERMDISP, which asks the separate question of whether the groups "
              f"differ in how variable they are, gave F = {sep.permdisp_F:.2f}, "
              f"p = {sep.permdisp_p:.4f}.")
    if np.isfinite(sep.energy_p):
        s.add(f"The energy k-sample test, which responds to any difference in "
              f"distribution rather than only to a shift in the average, gave "
              f"p = {sep.energy_p:.4f}.")
    s.add(sep.interpretation)
    s.add_table("Omnibus tests", sep.table())
    s.add_table("Every pair of groups, FDR-corrected", sep.pairwise)


def _classification_section(rep: Report, result) -> None:
    cls = getattr(result, "classification", None)
    if cls is None or not np.isfinite(cls.balanced_accuracy):
        return
    s = rep.section("Can new samples be assigned to a group?")
    lo, hi = cls.ba_ci
    ci = f" (95% CI {lo:.2f} to {hi:.2f})" if np.isfinite(lo) else ""
    s.add(f"Using {cls.model_label} and {cls.cv_name}, held-out samples were "
          f"assigned to their group with a balanced accuracy of "
          f"{cls.balanced_accuracy:.3f}{ci}, against a chance level of "
          f"{cls.chance:.3f}. Shuffling the labels {cls.n_permutations} times "
          f"gave p = {cls.permutation_p:.4f}. Cohen's kappa was {cls.kappa:.3f}.")
    if cls.cv_note:
        s.add(cls.cv_note)
    for note in cls.notes:
        s.add(note)
    s.add_table("Confusion matrix (out-of-fold predictions)",
                cls.confusion_frame().reset_index().rename(columns={"index": ""}))
    s.add_table("Separability of each pair of groups", cls.pairwise)


def _drivers_section(rep: Report, result) -> None:
    cls = getattr(result, "classification", None)
    imp = getattr(result, "importance", None)
    eff = getattr(result, "effects", None)
    if cls is None and imp is None and eff is None:
        return
    s = rep.section("Which metrics drive the difference?")

    if cls is not None and cls.activation is not None and cls.feature_names:
        from .stats.interpret import weight_table

        wt = weight_table(cls.feature_names, cls.weights, cls.activation, cls.vip)
        s.add("Classifier weights are reported next to Haufe-transformed "
              "activation patterns. The weights say how the model extracts the "
              "signal and can be large for a metric that carries none; the "
              "activation pattern says which metrics actually covary with the "
              "group difference, and is the column to read.")
        s.add_table("Classifier weights and activation patterns", wt, limit=15)

    if imp is not None and not imp.table.empty:
        s.add(f"Permutation importance measures how much balanced accuracy is "
              f"lost when one metric is shuffled, averaged over "
              f"{imp.n_repeats} repeats within each cross-validation fold.")
        s.add_table("Permutation importance, per metric", imp.table, limit=15)
        if not imp.cluster_table.empty:
            s.add(f"Metrics correlated above |r| = {imp.clustered_at:g} were also "
                  f"permuted as whole clusters, because shuffling one member of a "
                  f"correlated group understates all of them: the model simply "
                  f"reads the others.")
            s.add_table("Permutation importance, per correlated cluster",
                        imp.cluster_table, limit=12)

    if eff is not None and not eff.empty:
        s.add("Effect sizes are given in the original measurement units with "
              "bias-corrected bootstrap confidence intervals, so the size of each "
              "difference can be judged independently of its p value.")
        s.add_table("Effect size per metric, against the reference group",
                    eff, limit=20)


def _projection_section(rep: Report, result) -> None:
    proj = getattr(result, "projections", None) or {}
    if not proj:
        return
    s = rep.section("Projections")
    s.add("Every projection is reported with its neighbourhood preservation, so "
          "a picture can be judged before it is believed. Trustworthiness "
          "penalises neighbours the projection invented; continuity penalises "
          "neighbours it lost; the area under the R_NX curve summarises both "
          "across all neighbourhood sizes and is comparable between methods.")

    from .analysis import quality_table

    s.add_table("Projection quality", quality_table(result.projections).round(4))

    agree = getattr(result, "projection_agreement", None)
    if agree is not None and not agree.empty and len(agree) > 1:
        off = agree.to_numpy()[np.triu_indices(len(agree), 1)]
        s.add(f"Across the projections that were run, neighbouring samples agreed "
              f"on average {off.mean() * 100:.0f}% of the time. Structure that "
              f"survives several projections is a property of the data; structure "
              f"visible in only one is a property of that algorithm.")
        s.add_table("Shared nearest neighbours between projections",
                    agree.round(3).reset_index().rename(columns={"index": ""}))

    supervised = [p for p in proj.values() if p.supervised]
    if supervised:
        s.add("Supervised projections were run and are shown both in-sample and "
              "out-of-fold. A supervised projection separates the groups by "
              "construction and does so on random data too, so its in-sample "
              "panel is an illustration; the out-of-fold panel and the "
              "cross-validated score are the evidence.")


def _som_section(rep: Report, result) -> None:
    som = getattr(result, "som", None)
    if som is None:
        return
    s = rep.section("Self-organising map")
    cfg = result.config.som
    s.add(f"A {som.width} x {som.height} {cfg.topology}agonal map was trained for "
          f"{cfg.epochs} epochs with the {cfg.algorithm} algorithm.")
    if result.quality:
        s.add("Map quality: " + ", ".join(
            f"{k.replace('_', ' ')} = {v:.3f}"
            for k, v in result.quality.items() if isinstance(v, (int, float))))
    nc = getattr(result, "node_clusters", None)
    if nc is not None and nc.best_k:
        s.add(f"The codebook was divided into {nc.best_k} behavioural clusters by "
              f"{nc.method}, chosen by the average rank of the silhouette, "
              f"Davies-Bouldin and Calinski-Harabasz indices.")


# ==========================================================================
# Rendering
# ==========================================================================
def render_markdown(rep: Report) -> str:
    out: list[str] = [f"# {rep.title}", "", f"*Generated {rep.generated} by "
                                            f"SOMTrack {rep.provenance.get('SOMTrack version', '')}*", ""]

    if rep.verdict_text:
        out += ["## Conclusion", "", rep.verdict_text, ""]

    for s in rep.sections:
        out += [f"## {s.title}", ""]
        for para in s.body:
            out += [para, ""]
        for caption, df in s.tables:
            out += [f"**{caption}**", "", _md_table(df), ""]

    if rep.caveats:
        out += ["## Limits of this analysis", ""]
        out += [f"- {c}" for c in rep.caveats] + [""]

    if rep.figures:
        out += ["## Figures", ""]
        for name, caption in rep.figures:
            out.append(f"- **{name}** -- {caption}" if caption else f"- **{name}**")
        out.append("")

    out += _methods_markdown(rep)
    out += _references_markdown(rep)

    out += ["## Reproducibility", ""]
    for k, v in rep.provenance.items():
        out.append(f"- {k}: `{v}`")
    out += ["",
            "Re-running `somtrack run` with the saved `analysis_config.json` "
            "reproduces every number above.", ""]
    return "\n".join(out)


def _methods_markdown(rep: Report) -> list[str]:
    by_section = rep.methods.by_section()
    if not by_section:
        return []
    out = ["## Methods", "",
           "The text below describes the steps this run actually performed, with "
           "the parameters it used. Citations are given in the form used by the "
           "reference list that follows.", ""]
    for section, entries in by_section.items():
        sentences = []
        for e in entries:
            text = e.sentence()
            params = e.param_text()
            if params:
                text += f" [{params}]"
            sentences.append(text)
        out += [f"**{section}.** " + "; ".join(sentences) + ".", ""]

    caveats = rep.methods.caveats()
    if caveats:
        out += ["**Caveats attached to the methods above.**", ""]
        out += [f"- *{label}*: {text}" for label, text in caveats] + [""]
    return out


def _references_markdown(rep: Report) -> list[str]:
    if not rep.references:
        return []
    out = ["## References", ""]
    for c in rep.references:
        line = f"- {c.formatted}"
        if c.link:
            line = f"- {c.authors} ({c.year}). {c.title}. {c.source}. [{c.link}]({c.link})"
        out.append(line)
    out.append("")
    return out


def _md_table(df: pd.DataFrame, max_cols: int = 10) -> str:
    d = df.copy()
    if d.shape[1] > max_cols:
        d = d.iloc[:, :max_cols]
    d = d.map(_fmt_cell) if hasattr(d, "map") else d.applymap(_fmt_cell)
    header = "| " + " | ".join(str(c) for c in d.columns) + " |"
    rule = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(str(v) for v in r) + " |"
            for r in d.itertuples(index=False)]
    return "\n".join([header, rule] + rows)


def _fmt_cell(v):
    if isinstance(v, float):
        if not np.isfinite(v):
            return ""
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e5):
            return f"{v:.2e}"
        return f"{v:.4g}"
    return v


# --------------------------------------------------------------------------
_CONFIDENCE_COLOUR = {
    "strong": "#0072B2", "moderate": "#009E73",
    "weak": "#E69F00", "none": "#7F7F7F", "unknown": "#7F7F7F",
}


def render_html(rep: Report) -> str:
    accent = _CONFIDENCE_COLOUR.get(rep.confidence, "#0072B2")
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{_html.escape(rep.title)}</title>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        _CSS.replace("__ACCENT__", accent),
        "</head><body><main>",
        f"<h1>{_html.escape(rep.title)}</h1>",
        f"<p class='meta'>Generated {_html.escape(rep.generated)} by SOMTrack "
        f"{_html.escape(str(rep.provenance.get('SOMTrack version', '')))}</p>",
    ]

    if rep.verdict_text:
        parts.append("<section class='verdict'><h2>Conclusion</h2>")
        for para in rep.verdict_text.split("\n\n"):
            parts.append(f"<p>{_html.escape(para)}</p>")
        parts.append("</section>")

    for s in rep.sections:
        parts.append(f"<section><h2>{_html.escape(s.title)}</h2>")
        for para in s.body:
            parts.append(f"<p>{_html.escape(para)}</p>")
        for caption, df in s.tables:
            parts.append(f"<figure><figcaption>{_html.escape(caption)}</figcaption>")
            parts.append(_html_table(df))
            parts.append("</figure>")
        parts.append("</section>")

    if rep.caveats:
        parts.append("<section class='caveats'><h2>Limits of this analysis</h2><ul>")
        parts += [f"<li>{_html.escape(c)}</li>" for c in rep.caveats]
        parts.append("</ul></section>")

    by_section = rep.methods.by_section()
    if by_section:
        parts.append("<section><h2>Methods</h2>")
        parts.append("<p>The text below describes the steps this run actually "
                     "performed, with the parameters it used.</p>")
        for section, entries in by_section.items():
            sentences = []
            for e in entries:
                text = _html.escape(e.sentence())
                params = e.param_text()
                if params:
                    text += f" <span class='params'>[{_html.escape(params)}]</span>"
                sentences.append(text)
            parts.append(f"<p><strong>{_html.escape(section)}.</strong> "
                         + "; ".join(sentences) + ".</p>")
        caveats = rep.methods.caveats()
        if caveats:
            parts.append("<ul class='small'>")
            parts += [f"<li><em>{_html.escape(a)}</em>: {_html.escape(b)}</li>"
                      for a, b in caveats]
            parts.append("</ul>")
        parts.append("</section>")

    if rep.references:
        parts.append("<section><h2>References</h2><ol class='refs'>")
        for c in rep.references:
            link = (f" <a href='{_html.escape(c.link)}'>{_html.escape(c.link)}</a>"
                    if c.link else "")
            parts.append(f"<li>{_html.escape(c.authors)} ({c.year}). "
                         f"{_html.escape(c.title)}. {_html.escape(c.source)}.{link}</li>")
        parts.append("</ol></section>")

    parts.append("<section><h2>Reproducibility</h2><table class='kv'>")
    for k, v in rep.provenance.items():
        parts.append(f"<tr><th>{_html.escape(str(k))}</th>"
                     f"<td><code>{_html.escape(str(v))}</code></td></tr>")
    parts.append("</table></section>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def _html_table(df: pd.DataFrame, max_cols: int = 12) -> str:
    d = df.copy()
    if d.shape[1] > max_cols:
        d = d.iloc[:, :max_cols]
    d = d.map(_fmt_cell) if hasattr(d, "map") else d.applymap(_fmt_cell)
    head = "".join(f"<th>{_html.escape(str(c))}</th>" for c in d.columns)
    body = "".join(
        "<tr>" + "".join(f"<td>{_html.escape(str(v))}</td>" for v in row) + "</tr>"
        for row in d.itertuples(index=False))
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


_CSS = """<style>
:root{--ink:#1a1a1a;--muted:#5b5b5b;--rule:#e2e2e2;--bg:#ffffff;
      --panel:#f6f8fa;--accent:__ACCENT__;}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){
  --ink:#e8e8e8;--muted:#a8a8a8;--rule:#333;--bg:#161616;--panel:#1f2124;}}
:root[data-theme="dark"]{--ink:#e8e8e8;--muted:#a8a8a8;--rule:#333;
  --bg:#161616;--panel:#1f2124;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
     font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,sans-serif;}
main{max-width:52rem;margin:0 auto;padding:2.5rem 16px 5rem;}
h1{font-size:1.8rem;line-height:1.25;margin:0 0 .3rem;}
h2{font-size:1.15rem;margin:2.4rem 0 .7rem;padding-bottom:.35rem;
   border-bottom:1px solid var(--rule);}
p{margin:.7rem 0;}
.meta{color:var(--muted);font-size:.87rem;margin-bottom:2rem;}
.verdict{background:var(--panel);border-left:4px solid var(--accent);
         padding:1rem 1.2rem;border-radius:0 6px 6px 0;margin:1.5rem 0;}
.verdict h2{margin-top:0;border:0;color:var(--accent);}
.caveats li{margin:.4rem 0;}
figure{margin:1.2rem 0;overflow-x:auto;}
figcaption{font-weight:600;font-size:.9rem;margin-bottom:.4rem;}
table{border-collapse:collapse;font-size:.83rem;width:100%;}
th,td{text-align:left;padding:.32rem .6rem;border-bottom:1px solid var(--rule);
      white-space:nowrap;}
th{font-weight:600;color:var(--muted);}
table.kv{width:auto;} table.kv th{padding-right:1.5rem;}
code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.85em;}
.params{color:var(--muted);font-size:.9em;}
.refs li{margin:.45rem 0;font-size:.9rem;}
.refs a{color:var(--accent);word-break:break-all;}
.small{font-size:.88rem;color:var(--muted);}
@media (max-width:600px){main{padding:1.5rem 16px 3rem;}h1{font-size:1.45rem;}}
</style>"""


# ==========================================================================
def write_report(rep: Report, out_dir: str | Path, cfg=None) -> list[Path]:
    """Write the report next to the figures and tables."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    want_md = True if cfg is None else cfg.markdown
    want_html = True if cfg is None else cfg.html

    if want_md:
        p = out / "RESULTS_REPORT.md"
        p.write_text(render_markdown(rep), encoding="utf-8")
        written.append(p)
    if want_html:
        p = out / "RESULTS_REPORT.html"
        p.write_text(render_html(rep), encoding="utf-8")
        written.append(p)
    return written


def write_methods_text(rep: Report, path: str | Path) -> Path:
    """The methods paragraph and reference list on their own, ready to paste."""
    lines = [f"Methods -- {rep.title}",
             "=" * 66, ""]
    for section, entries in rep.methods.by_section().items():
        sentences = []
        for e in entries:
            text = e.sentence()
            params = e.param_text()
            if params:
                text += f" [{params}]"
            sentences.append(text)
        lines += [f"{section}. " + "; ".join(sentences) + ".", ""]

    caveats = rep.methods.caveats()
    if caveats:
        lines += ["Caveats", "-" * 66]
        lines += [f"  {a}: {b}" for a, b in caveats] + [""]

    if rep.references:
        lines += ["References", "-" * 66]
        lines += [f"  {c.formatted}" for c in rep.references] + [""]

    lines += ["Generated by SOMTrack "
              f"{rep.provenance.get('SOMTrack version', '')} on {rep.generated}.",
              "Figures were produced with matplotlib; PDF and SVG output embeds "
              "text as editable text (TrueType, fonttype 42).", ""]

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


__all__ = ["Report", "ReportSection", "build_report", "render_markdown",
           "render_html", "write_report", "write_methods_text"]
