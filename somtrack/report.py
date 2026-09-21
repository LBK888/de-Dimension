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

**Languages.**  The report is written in English and then, by default, again in
Traditional Chinese (``ReportConfig.translation``).  The two are never mixed: the
English report is complete on its own, and the translation follows it as a
second, complete document.  What is translated is the prose -- headings,
paragraphs, the verdict, table and figure captions, caveats and the methods
paragraph.  What is not is everything that is data or a figure: the figures
themselves, table contents and column names, metric and group names, parameter
names and the bibliographic entries, so every number and label can be matched
between the two versions and against the figures.
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
from .i18n import HTML_LANG, Text, normalise, render

# per-language punctuation, used where the report joins pieces itself
_PUNCT = {
    "en": {"stop": ".", "clause": "; ", "colon": ": ", "dash": " -- "},
    "zh_TW": {"stop": "。", "clause": "；", "colon": "：", "dash": "——"},
}


def _p(lang: str) -> dict:
    return _PUNCT.get(lang, _PUNCT["en"])


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
    # English first, then each translation, as complete documents
    languages: list[str] = field(default_factory=lambda: ["en", "zh_TW"])

    def section(self, title: str) -> ReportSection:
        for s in self.sections:
            if s.title == title:
                return s
        s = ReportSection(title)
        self.sections.append(s)
        return s


def _languages(cfg) -> list[str]:
    extra = normalise(getattr(cfg, "translation", "") or "en")
    return ["en"] if extra == "en" else ["en", extra]


# ==========================================================================
# Assembly
# ==========================================================================
def build_report(result, log: MethodsLog | None = None) -> Report:
    """Turn an :class:`somtrack.pipeline.AnalysisResult` into a written report."""
    cfg = result.config
    log = log or getattr(result, "methods_log", None) or MethodsLog()
    prep = result.prep

    title = cfg.report.title or Text("Multivariate analysis of locomotion metrics")
    rep = Report(title=title,
                 generated=datetime.now().strftime("%Y-%m-%d %H:%M"),
                 methods=log, languages=_languages(cfg.report))

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
    s.add(Text("{n} samples described by {m} metrics, in {k} groups: {counts}.",
               n=prep.n_samples, m=prep.n_features, k=prep.n_groups, counts=counts))
    structure = getattr(result, "block_structure", None)
    if structure is not None:
        s.add(structure.note)
    if prep.dropped_features:
        names = (", ".join(prep.dropped_features[:10])
                 + ("..." if len(prep.dropped_features) > 10 else ""))
        s.add(Text("{n} metrics were dropped before analysis ({names}).",
                   n=len(prep.dropped_features), names=names))
    for note in prep.notes:
        s.add(note)


def _separation_section(rep: Report, result) -> None:
    sep = getattr(result, "separation", None)
    if sep is None or not np.isfinite(sep.permanova_p):
        return
    s = rep.section("Do the groups differ?")
    s.add(Text("PERMANOVA on {metric} distances gave pseudo-F = {F:.2f} with "
               "R2 = {r2:.3f} and p = {p:.4f} over {n} permutations. The grouping "
               "therefore accounts for {pct:.1f}% of the multivariate variation.",
               metric=Text(sep.metric), F=sep.permanova_F, r2=sep.permanova_R2,
               p=sep.permanova_p, n=sep.n_permutations,
               pct=sep.permanova_R2 * 100))
    if np.isfinite(sep.permdisp_p):
        s.add(Text("PERMDISP, which asks the separate question of whether the "
                   "groups differ in how variable they are, gave F = {F:.2f}, "
                   "p = {p:.4f}.", F=sep.permdisp_F, p=sep.permdisp_p))
    if np.isfinite(sep.energy_p):
        s.add(Text("The energy k-sample test, which responds to any difference in "
                   "distribution rather than only to a shift in the average, gave "
                   "p = {p:.4f}.", p=sep.energy_p))
    s.add(sep.interpretation)
    s.add_table("Omnibus tests", sep.table())
    s.add_table("Every pair of groups, FDR-corrected", sep.pairwise)


def _classification_section(rep: Report, result) -> None:
    cls = getattr(result, "classification", None)
    if cls is None or not np.isfinite(cls.balanced_accuracy):
        return
    s = rep.section("Can new samples be assigned to a group?")
    lo, hi = cls.ba_ci
    ci = (Text(" (95% CI {lo:.2f} to {hi:.2f})", lo=lo, hi=hi)
          if np.isfinite(lo) else "")
    s.add(Text("Using {model} and {cv}, held-out samples were assigned to their "
               "group with a balanced accuracy of {ba:.3f}{ci}, against a chance "
               "level of {chance:.3f}. Shuffling the labels {n} times gave "
               "p = {p:.4f}. Cohen's kappa was {kappa:.3f}.",
               model=Text(cls.model_label), cv=cls.cv_name,
               ba=cls.balanced_accuracy, ci=ci, chance=cls.chance,
               n=cls.n_permutations, p=cls.permutation_p, kappa=cls.kappa))
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
        s.add(Text("Classifier weights are reported next to Haufe-transformed "
                   "activation patterns. The weights say how the model extracts the "
                   "signal and can be large for a metric that carries none; the "
                   "activation pattern says which metrics actually covary with the "
                   "group difference, and is the column to read."))
        s.add_table("Classifier weights and activation patterns", wt, limit=15)

    if imp is not None and not imp.table.empty:
        s.add(Text("Permutation importance measures how much balanced accuracy is "
                   "lost when one metric is shuffled, averaged over {n} repeats "
                   "within each cross-validation fold.", n=imp.n_repeats))
        s.add_table("Permutation importance, per metric", imp.table, limit=15)
        if not imp.cluster_table.empty:
            s.add(Text("Metrics correlated above |r| = {r:g} were also permuted as "
                       "whole clusters, because shuffling one member of a "
                       "correlated group understates all of them: the model simply "
                       "reads the others.", r=imp.clustered_at))
            s.add_table("Permutation importance, per correlated cluster",
                        imp.cluster_table, limit=12)

    if eff is not None and not eff.empty:
        s.add(Text("Effect sizes are given in the original measurement units with "
                   "bias-corrected bootstrap confidence intervals, so the size of "
                   "each difference can be judged independently of its p value."))
        s.add_table("Effect size per metric, against the reference group",
                    eff, limit=20)


def _projection_section(rep: Report, result) -> None:
    proj = getattr(result, "projections", None) or {}
    if not proj:
        return
    s = rep.section("Projections")
    s.add(Text("Every projection is reported with its neighbourhood preservation, "
               "so a picture can be judged before it is believed. Trustworthiness "
               "penalises neighbours the projection invented; continuity penalises "
               "neighbours it lost; the area under the R_NX curve summarises both "
               "across all neighbourhood sizes and is comparable between methods."))

    from .analysis import quality_table

    s.add_table("Projection quality", quality_table(result.projections).round(4))

    agree = getattr(result, "projection_agreement", None)
    if agree is not None and not agree.empty and len(agree) > 1:
        off = agree.to_numpy()[np.triu_indices(len(agree), 1)]
        s.add(Text("Across the projections that were run, neighbouring samples "
                   "agreed on average {pct:.0f}% of the time. Structure that "
                   "survives several projections is a property of the data; "
                   "structure visible in only one is a property of that algorithm.",
                   pct=off.mean() * 100))
        s.add_table("Shared nearest neighbours between projections",
                    agree.round(3).reset_index().rename(columns={"index": ""}))

    supervised = [p for p in proj.values() if p.supervised]
    if supervised:
        s.add(Text("Supervised projections were run and are shown both in-sample and "
                   "out-of-fold. A supervised projection separates the groups by "
                   "construction and does so on random data too, so its in-sample "
                   "panel is an illustration; the out-of-fold panel and the "
                   "cross-validated score are the evidence."))


def _som_section(rep: Report, result) -> None:
    som = getattr(result, "som", None)
    if som is None:
        return
    s = rep.section("Self-organising map")
    cfg = result.config.som
    lattice = {"hex": "hexagonal", "rect": "rectangular"}.get(cfg.topology,
                                                              cfg.topology)
    s.add(Text("A {w} x {h} {lattice} map was trained for {epochs} epochs with the "
               "{algorithm} algorithm.", w=som.width, h=som.height,
               lattice=Text(lattice), epochs=cfg.epochs, algorithm=cfg.algorithm))
    if result.quality:
        values = ", ".join(f"{k.replace('_', ' ')} = {v:.3f}"
                           for k, v in result.quality.items()
                           if isinstance(v, (int, float)))
        s.add(Text("Map quality: {values}", values=values))
    nc = getattr(result, "node_clusters", None)
    if nc is not None and nc.best_k:
        s.add(Text("The codebook was divided into {k} behavioural clusters by "
                   "{method}, chosen by the average rank of the silhouette, "
                   "Davies-Bouldin and Calinski-Harabasz indices.",
                   k=nc.best_k, method=nc.method))


# ==========================================================================
# Rendering
# ==========================================================================
def _t(obj, lang: str) -> str:
    return render(obj, lang)


def _translation_note(lang: str) -> str:
    return _t("This is a translation of the English report above. Figures, table "
              "contents, metric and group names and the reference list are kept "
              "in English, so every value can be matched between the two "
              "versions.", lang)


def render_markdown(rep: Report) -> str:
    out: list[str] = []
    for i, lang in enumerate(rep.languages):
        if i:
            out += ["", "---", ""]
        out += _markdown_document(rep, lang)
    return "\n".join(out)


def _markdown_document(rep: Report, lang: str) -> list[str]:
    P = _p(lang)
    title = _t(rep.title, lang)
    if lang != "en":
        title = _t("{title} (translation)", lang).format(title=title)
    version = rep.provenance.get("SOMTrack version", "")
    out: list[str] = [f"# {title}", "",
                      "*" + _t("Generated {date} by SOMTrack {version}", lang).format(
                          date=rep.generated, version=version) + "*", ""]
    if lang != "en":
        out += [f"> {_translation_note(lang)}", ""]

    if rep.verdict_text:
        out += [f"## {_t('Conclusion', lang)}", "", _t(rep.verdict_text, lang), ""]

    for s in rep.sections:
        out += [f"## {_t(s.title, lang)}", ""]
        for para in s.body:
            out += [_t(para, lang), ""]
        for caption, df in s.tables:
            out += [f"**{_t(caption, lang)}**", "", _md_table(df), ""]

    if rep.caveats:
        out += [f"## {_t('Limits of this analysis', lang)}", ""]
        out += [f"- {_t(c, lang)}" for c in rep.caveats] + [""]

    if rep.figures:
        out += [f"## {_t('Figures', lang)}", ""]
        for name, caption in rep.figures:
            out.append(f"- **{name}**{P['dash']}{_t(caption, lang)}" if caption
                       else f"- **{name}**")
        out.append("")

    out += _methods_markdown(rep, lang)
    out += _references_markdown(rep, lang)

    out += [f"## {_t('Reproducibility', lang)}", ""]
    for k, v in rep.provenance.items():
        out.append(f"- {_t(k, lang)}{P['colon']}`{v}`")
    out += ["",
            _t("Re-running `somtrack run` with the saved `analysis_config.json` "
               "reproduces every number above.", lang), ""]
    return out


def _methods_sentences(rep: Report, lang: str, escape=lambda x: x,
                       wrap_params=lambda x: f" [{x}]") -> list[tuple[str, str]]:
    """``(section heading, the paragraph)`` for every section of the methods log."""
    P = _p(lang)
    out = []
    for section, entries in rep.methods.by_section().items():
        sentences = []
        for e in entries:
            text = escape(_t(e.sentence(), lang))
            params = e.param_text()
            if params:
                text += wrap_params(escape(params))
            sentences.append(text)
        out.append((_t(section, lang), P["clause"].join(sentences) + P["stop"]))
    return out


def _methods_markdown(rep: Report, lang: str = "en") -> list[str]:
    if not rep.methods.by_section():
        return []
    P = _p(lang)
    out = [f"## {_t('Methods', lang)}", "",
           _t("The text below describes the steps this run actually performed, with "
              "the parameters it used. Citations are given in the form used by the "
              "reference list that follows.", lang), ""]
    for section, paragraph in _methods_sentences(rep, lang):
        out += [f"**{section}{P['stop']}** {paragraph}", ""]

    caveats = rep.methods.caveats()
    if caveats:
        out += [f"**{_t('Caveats attached to the methods above.', lang)}**", ""]
        out += [f"- *{_t(label, lang)}*{P['colon']}{_t(text, lang)}"
                for label, text in caveats] + [""]
    return out


def _references_markdown(rep: Report, lang: str = "en") -> list[str]:
    if not rep.references:
        return []
    if lang != "en":
        # A reference is not translated, and one list per report keeps "every
        # citation appears once" true of the whole file.
        return [f"## {_t('References', lang)}", "",
                _t("The reference list is the one at the end of the English "
                   "report above; bibliographic entries are not translated.", lang),
                ""]
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
    esc = _html.escape
    parts = [
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>",
        f"<title>{esc(_t(rep.title, 'en'))}</title>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        _CSS.replace("__ACCENT__", accent),
        "</head><body><main>",
    ]
    if len(rep.languages) > 1:
        links = " · ".join(
            f"<a href='#lang-{lang}' lang='{HTML_LANG.get(lang, lang)}'>"
            f"{esc(_t('English', lang) if lang == 'en' else _t('Translation', lang))}</a>"
            for lang in rep.languages)
        parts.append(f"<nav class='langs'>{links}</nav>")
    for i, lang in enumerate(rep.languages):
        if i:
            parts.append("<hr class='lang-break'>")
        parts.append(f"<div id='lang-{lang}' lang='{HTML_LANG.get(lang, lang)}'>")
        parts += _html_document(rep, lang)
        parts.append("</div>")
    parts.append("</main></body></html>")
    return "\n".join(parts)


def _html_document(rep: Report, lang: str) -> list[str]:
    esc = _html.escape
    P = _p(lang)
    title = _t(rep.title, lang)
    if lang != "en":
        title = _t("{title} (translation)", lang).format(title=title)
    version = str(rep.provenance.get("SOMTrack version", ""))
    parts = [
        f"<h1>{esc(title)}</h1>",
        "<p class='meta'>" + esc(_t("Generated {date} by SOMTrack {version}", lang)
                                 .format(date=rep.generated, version=version)) + "</p>",
    ]
    if lang != "en":
        parts.append(f"<p class='translation-note'>{esc(_translation_note(lang))}</p>")

    if rep.verdict_text:
        parts.append(f"<section class='verdict'><h2>{esc(_t('Conclusion', lang))}</h2>")
        for para in _t(rep.verdict_text, lang).split("\n\n"):
            parts.append(f"<p>{esc(para)}</p>")
        parts.append("</section>")

    for s in rep.sections:
        parts.append(f"<section><h2>{esc(_t(s.title, lang))}</h2>")
        for para in s.body:
            parts.append(f"<p>{esc(_t(para, lang))}</p>")
        for caption, df in s.tables:
            parts.append(f"<figure><figcaption>{esc(_t(caption, lang))}</figcaption>")
            parts.append(_html_table(df))
            parts.append("</figure>")
        parts.append("</section>")

    if rep.caveats:
        parts.append(f"<section class='caveats'><h2>"
                     f"{esc(_t('Limits of this analysis', lang))}</h2><ul>")
        parts += [f"<li>{esc(_t(c, lang))}</li>" for c in rep.caveats]
        parts.append("</ul></section>")

    if rep.figures:
        parts.append(f"<section><h2>{esc(_t('Figures', lang))}</h2><ul class='figs'>")
        for name, caption in rep.figures:
            cap = f"{P['dash']}{esc(_t(caption, lang))}" if caption else ""
            parts.append(f"<li><strong>{esc(name)}</strong>{cap}</li>")
        parts.append("</ul></section>")

    if rep.methods.by_section():
        parts.append(f"<section><h2>{esc(_t('Methods', lang))}</h2>")
        parts.append("<p>" + esc(_t("The text below describes the steps this run "
                                    "actually performed, with the parameters it "
                                    "used.", lang)) + "</p>")
        for section, paragraph in _methods_sentences(
                rep, lang, escape=esc,
                wrap_params=lambda x: f" <span class='params'>[{x}]</span>"):
            parts.append(f"<p><strong>{esc(section)}{P['stop']}</strong> {paragraph}</p>")
        caveats = rep.methods.caveats()
        if caveats:
            parts.append("<ul class='small'>")
            parts += [f"<li><em>{esc(_t(a, lang))}</em>{P['colon']}{esc(_t(b, lang))}</li>"
                      for a, b in caveats]
            parts.append("</ul>")
        parts.append("</section>")

    if rep.references:
        parts.append(f"<section><h2>{esc(_t('References', lang))}</h2>")
        if lang != "en":
            parts.append("<p><a href='#lang-en-references'>"
                         + esc(_t("The reference list is the one at the end of the "
                                  "English report above; bibliographic entries are "
                                  "not translated.", lang)) + "</a></p>")
        else:
            parts.append("<ol class='refs' id='lang-en-references'>")
            for c in rep.references:
                link = (f" <a href='{esc(c.link)}'>{esc(c.link)}</a>"
                        if c.link else "")
                parts.append(f"<li>{esc(c.authors)} ({c.year}). "
                             f"{esc(c.title)}. {esc(c.source)}.{link}</li>")
            parts.append("</ol>")
        parts.append("</section>")

    parts.append(f"<section><h2>{esc(_t('Reproducibility', lang))}</h2><table class='kv'>")
    for k, v in rep.provenance.items():
        parts.append(f"<tr><th>{esc(_t(str(k), lang))}</th>"
                     f"<td><code>{esc(str(v))}</code></td></tr>")
    parts.append("</table></section>")
    return parts


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
     font:15px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Arial,
     "Microsoft JhengHei","PingFang TC","Noto Sans TC",sans-serif;}
main{max-width:52rem;margin:0 auto;padding:2.5rem 16px 5rem;}
h1{font-size:1.8rem;line-height:1.25;margin:0 0 .3rem;}
h2{font-size:1.15rem;margin:2.4rem 0 .7rem;padding-bottom:.35rem;
   border-bottom:1px solid var(--rule);}
p{margin:.7rem 0;}
.meta{color:var(--muted);font-size:.87rem;margin-bottom:2rem;}
.langs{font-size:.9rem;margin-bottom:1.5rem;color:var(--muted);}
.langs a{color:var(--accent);}
hr.lang-break{border:0;border-top:3px double var(--rule);margin:4rem 0 3rem;}
.translation-note{background:var(--panel);padding:.7rem 1rem;border-radius:6px;
                  color:var(--muted);font-size:.9rem;}
.verdict{background:var(--panel);border-left:4px solid var(--accent);
         padding:1rem 1.2rem;border-radius:0 6px 6px 0;margin:1.5rem 0;}
.verdict h2{margin-top:0;border:0;color:var(--accent);}
.caveats li{margin:.4rem 0;}
.figs li{margin:.3rem 0;}
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
    """The methods paragraph and reference list on their own, ready to paste.

    English first; each translation follows as its own complete block, pointing
    back to the one reference list.
    """
    lines: list[str] = []
    for i, lang in enumerate(rep.languages):
        if i:
            lines += ["", "#" * 66, ""]
        lines += _methods_text_block(rep, lang)

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _methods_text_block(rep: Report, lang: str) -> list[str]:
    P = _p(lang)
    title = _t(rep.title, lang)
    head = (f"Methods -- {title}" if lang == "en"
            else _t("Methods -- {title} (translation)", lang).format(title=title))
    lines = [head, "=" * 66, ""]
    if lang != "en":
        lines += [_translation_note(lang), ""]
    for section, paragraph in _methods_sentences(rep, lang):
        lines += [f"{section}{P['stop']} {paragraph}", ""]

    caveats = rep.methods.caveats()
    if caveats:
        lines += [_t("Caveats", lang), "-" * 66]
        lines += [f"  {_t(a, lang)}{P['colon']}{_t(b, lang)}" for a, b in caveats] + [""]

    if rep.references:
        lines += [_t("References", lang), "-" * 66]
        if lang == "en":
            lines += [f"  {c.formatted}" for c in rep.references] + [""]
        else:
            lines += ["  " + _t("The reference list is the one at the end of the "
                                "English section above; bibliographic entries are "
                                "not translated.", lang), ""]

    version = rep.provenance.get("SOMTrack version", "")
    lines += [_t("Generated by SOMTrack {version} on {date}.", lang).format(
                  version=version, date=rep.generated),
              _t("Figures were produced with matplotlib; PDF and SVG output embeds "
                 "text as editable text (TrueType, fonttype 42).", lang), ""]
    return lines


__all__ = ["Report", "ReportSection", "build_report", "render_markdown",
           "render_html", "write_report", "write_methods_text"]
