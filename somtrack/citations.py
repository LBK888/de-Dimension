"""Bibliography and the per-run methods log.

Two things live here.

:class:`Citation` and :data:`CITATIONS`
    A single, checked bibliography for every algorithm SOMTrack can run.  Each
    entry carries the information a reference list needs, so nothing has to be
    re-typed when a run is written up.

:class:`MethodsLog`
    What a *particular* run actually did.  Analysis code calls
    :meth:`MethodsLog.record` as each step happens, so the methods section
    describes the steps that ran rather than the steps that exist.  A run that
    skipped UMAP does not cite McInnes et al.; a run that used a supervised
    projection cites it *and* carries the caveat that goes with it.

The report writer (:mod:`somtrack.report`) turns the log into

    Visualization: SuperPlots (Lord et al., 2020); estimation plots with
    bootstrap confidence intervals (Ho et al., 2019).

plus a de-duplicated reference list in citation order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Literal

from .i18n import Text

# one author's initials: "T.", "S.J.", "C.J.F." -- one group per author
_INITIALS = re.compile(r"\b(?:[A-Z]\.){1,4}")

Section = Literal[
    "Software",
    "Data preparation",
    "Feature extraction",
    "Clustering",
    "Projection",
    "Statistics",
    "Visualization",
]

SECTION_ORDER: tuple[Section, ...] = (
    "Data preparation",
    "Feature extraction",
    "Clustering",
    "Projection",
    "Statistics",
    "Visualization",
    "Software",
)


# ==========================================================================
# Bibliography
# ==========================================================================
@dataclass(frozen=True)
class Citation:
    key: str
    authors: str
    year: int
    title: str
    source: str
    doi: str = ""
    url: str = ""

    @property
    def short(self) -> str:
        """``Lord et al., 2020`` -- the form used inline in a methods sentence.

        Authors are stored as ``Surname, I.I., Surname, I.I. & Surname, I.I.``,
        so the surnames cannot be counted by splitting on commas.  Counting the
        initial groups instead is unambiguous.
        """
        first = self.authors.split(",")[0].strip()
        if "et al" in self.authors:
            return f"{first} et al., {self.year}"

        n_authors = len(_INITIALS.findall(self.authors))
        if n_authors >= 3:
            return f"{first} et al., {self.year}"
        if n_authors == 2:
            second = self.authors.split("&")[-1].strip().split(",")[0].strip()
            return f"{first} & {second}, {self.year}"
        return f"{first}, {self.year}"

    @property
    def formatted(self) -> str:
        """One full reference-list line."""
        bits = [f"{self.authors} ({self.year}). {self.title}. {self.source}."]
        if self.doi:
            bits.append(f"doi:{self.doi}")
        elif self.url:
            bits.append(self.url)
        return " ".join(bits)

    @property
    def link(self) -> str:
        if self.doi:
            return f"https://doi.org/{self.doi}"
        return self.url


def _c(*args, **kw) -> tuple[str, Citation]:
    c = Citation(*args, **kw)
    return c.key, c


CITATIONS: dict[str, Citation] = dict([
    # ---------------------------------------------------------------- software
    _c("harris2020", "Harris, C.R. et al.", 2020,
       "Array programming with NumPy", "Nature 585:357-362",
       doi="10.1038/s41586-020-2649-2"),
    _c("virtanen2020", "Virtanen, P. et al.", 2020,
       "SciPy 1.0: fundamental algorithms for scientific computing in Python",
       "Nature Methods 17:261-272", doi="10.1038/s41592-019-0686-2"),
    _c("pedregosa2011", "Pedregosa, F. et al.", 2011,
       "Scikit-learn: machine learning in Python",
       "Journal of Machine Learning Research 12:2825-2830"),
    _c("hunter2007", "Hunter, J.D.", 2007,
       "Matplotlib: a 2D graphics environment",
       "Computing in Science & Engineering 9:90-95", doi="10.1109/MCSE.2007.55"),
    _c("mckinney2010", "McKinney, W.", 2010,
       "Data structures for statistical computing in Python",
       "Proceedings of the 9th Python in Science Conference, 56-61",
       doi="10.25080/Majora-92bf1922-00a"),

    # ------------------------------------------------------- data preparation
    _c("savitzky1964", "Savitzky, A. & Golay, M.J.E.", 1964,
       "Smoothing and differentiation of data by simplified least squares procedures",
       "Analytical Chemistry 36:1627-1639", doi="10.1021/ac60214a047"),
    _c("otsu1979", "Otsu, N.", 1979,
       "A threshold selection method from gray-level histograms",
       "IEEE Transactions on Systems, Man, and Cybernetics 9:62-66",
       doi="10.1109/TSMC.1979.4310076"),

    # ----------------------------------------------------- feature extraction
    _c("metzler2000", "Metzler, R. & Klafter, J.", 2000,
       "The random walk's guide to anomalous diffusion: a fractional dynamics approach",
       "Physics Reports 339:1-77", doi="10.1016/S0370-1573(00)00070-3"),
    _c("peng1994", "Peng, C.-K. et al.", 1994,
       "Mosaic organization of DNA nucleotides", "Physical Review E 49:1685-1689",
       doi="10.1103/PhysRevE.49.1685"),
    _c("katz1988", "Katz, M.J.", 1988,
       "Fractals and the analysis of waveforms",
       "Computers in Biology and Medicine 18:145-156",
       doi="10.1016/0010-4825(88)90041-8"),
    _c("goh2008", "Goh, K.-I. & Barabasi, A.-L.", 2008,
       "Burstiness and memory in complex systems",
       "Europhysics Letters 81:48002", doi="10.1209/0295-5075/81/48002"),
    _c("batschelet1981", "Batschelet, E.", 1981,
       "Circular Statistics in Biology", "Academic Press, London"),
    _c("triantafyllou1993", "Triantafyllou, G.S., Triantafyllou, M.S. & Grosenbaugh, M.A.",
       1993, "Optimal thrust development in oscillating foils with application to "
       "fish propulsion", "Journal of Fluids and Structures 7:205-224",
       doi="10.1006/jfls.1993.1012"),

    # ------------------------------------------------------------- clustering
    _c("kohonen1982", "Kohonen, T.", 1982,
       "Self-organized formation of topologically correct feature maps",
       "Biological Cybernetics 43:59-69", doi="10.1007/BF00337288"),
    _c("kohonen1999", "Kohonen, T.", 1999,
       "Comparison of SOM point densities based on different criteria",
       "Neural Computation 11:2081-2095", doi="10.1162/089976699300016098"),
    _c("kohonen2001", "Kohonen, T.", 2001,
       "Self-Organizing Maps, 3rd edition", "Springer, Berlin",
       doi="10.1007/978-3-642-56927-2"),
    _c("melssen2006", "Melssen, W., Wehrens, R. & Buydens, L.", 2006,
       "Supervised Kohonen networks for classification problems",
       "Chemometrics and Intelligent Laboratory Systems 83:99-113",
       doi="10.1016/j.chemolab.2006.02.003"),
    _c("hammer2002", "Hammer, B. & Villmann, T.", 2002,
       "Generalized relevance learning vector quantization",
       "Neural Networks 15:1059-1068", doi="10.1016/S0893-6080(02)00079-5"),
    _c("alahakoon2000", "Alahakoon, D., Halgamuge, S.K. & Srinivasan, B.", 2000,
       "Dynamic self-organizing maps with controlled growth for knowledge discovery",
       "IEEE Transactions on Neural Networks 11:601-614", doi="10.1109/72.846732"),
    _c("vesanto2000", "Vesanto, J. & Alhoniemi, E.", 2000,
       "Clustering of the self-organizing map",
       "IEEE Transactions on Neural Networks 11:586-600", doi="10.1109/72.846731"),
    _c("kiviluoto1996", "Kiviluoto, K.", 1996,
       "Topology preservation in self-organizing maps",
       "Proceedings of ICNN'96, 294-299", doi="10.1109/ICNN.1996.548907"),
    _c("ultsch1990", "Ultsch, A. & Siemon, H.P.", 1990,
       "Kohonen's self organizing feature maps for exploratory data analysis",
       "Proceedings of INNC'90, 305-308"),
    _c("macqueen1967", "MacQueen, J.", 1967,
       "Some methods for classification and analysis of multivariate observations",
       "Proceedings of the 5th Berkeley Symposium 1:281-297"),
    _c("ward1963", "Ward, J.H.", 1963,
       "Hierarchical grouping to optimize an objective function",
       "Journal of the American Statistical Association 58:236-244",
       doi="10.1080/01621459.1963.10500845"),
    _c("dempster1977", "Dempster, A.P., Laird, N.M. & Rubin, D.B.", 1977,
       "Maximum likelihood from incomplete data via the EM algorithm",
       "Journal of the Royal Statistical Society B 39:1-38"),
    _c("rousseeuw1987", "Rousseeuw, P.J.", 1987,
       "Silhouettes: a graphical aid to the interpretation and validation of "
       "cluster analysis", "Journal of Computational and Applied Mathematics 20:53-65",
       doi="10.1016/0377-0427(87)90125-7"),
    _c("davies1979", "Davies, D.L. & Bouldin, D.W.", 1979,
       "A cluster separation measure",
       "IEEE Transactions on Pattern Analysis and Machine Intelligence 1:224-227",
       doi="10.1109/TPAMI.1979.4766909"),
    _c("calinski1974", "Calinski, T. & Harabasz, J.", 1974,
       "A dendrite method for cluster analysis",
       "Communications in Statistics 3:1-27", doi="10.1080/03610927408827101"),

    # ------------------------------------------------------------- projection
    _c("hotelling1933", "Hotelling, H.", 1933,
       "Analysis of a complex of statistical variables into principal components",
       "Journal of Educational Psychology 24:417-441", doi="10.1037/h0071325"),
    _c("maaten2008", "van der Maaten, L. & Hinton, G.", 2008,
       "Visualizing data using t-SNE",
       "Journal of Machine Learning Research 9:2579-2605"),
    _c("mcinnes2018", "McInnes, L., Healy, J. & Melville, J.", 2018,
       "UMAP: Uniform Manifold Approximation and Projection for dimension reduction",
       "arXiv:1802.03426", doi="10.48550/arXiv.1802.03426"),
    _c("narayan2021", "Narayan, A., Berger, B. & Cho, H.", 2021,
       "Assessing single-cell transcriptomic variability through density-preserving "
       "data visualization", "Nature Biotechnology 39:765-774",
       doi="10.1038/s41587-020-00801-7"),
    _c("wang2021", "Wang, Y., Huang, H., Rudin, C. & Shaposhnik, Y.", 2021,
       "Understanding how dimension reduction tools work: an empirical approach to "
       "deciphering t-SNE, UMAP, TriMAP, and PaCMAP for data visualization",
       "Journal of Machine Learning Research 22:1-73"),
    _c("moon2019", "Moon, K.R. et al.", 2019,
       "Visualizing structure and transitions in high-dimensional biological data",
       "Nature Biotechnology 37:1482-1492", doi="10.1038/s41587-019-0336-3"),
    _c("torgerson1952", "Torgerson, W.S.", 1952,
       "Multidimensional scaling: I. Theory and method", "Psychometrika 17:401-419",
       doi="10.1007/BF02288916"),
    _c("gower1966", "Gower, J.C.", 1966,
       "Some distance properties of latent root and vector methods used in "
       "multivariate analysis", "Biometrika 53:325-338", doi="10.1093/biomet/53.3-4.325"),
    _c("kruskal1964", "Kruskal, J.B.", 1964,
       "Multidimensional scaling by optimizing goodness of fit to a nonmetric "
       "hypothesis", "Psychometrika 29:1-27", doi="10.1007/BF02289565"),
    _c("tenenbaum2000", "Tenenbaum, J.B., de Silva, V. & Langford, J.C.", 2000,
       "A global geometric framework for nonlinear dimensionality reduction",
       "Science 290:2319-2323", doi="10.1126/science.290.5500.2319"),
    _c("scholkopf1998", "Scholkopf, B., Smola, A. & Muller, K.-R.", 1998,
       "Nonlinear component analysis as a kernel eigenvalue problem",
       "Neural Computation 10:1299-1319", doi="10.1162/089976698300017467"),
    _c("fisher1936", "Fisher, R.A.", 1936,
       "The use of multiple measurements in taxonomic problems",
       "Annals of Eugenics 7:179-188", doi="10.1111/j.1469-1809.1936.tb02137.x"),
    _c("ledoit2004", "Ledoit, O. & Wolf, M.", 2004,
       "A well-conditioned estimator for large-dimensional covariance matrices",
       "Journal of Multivariate Analysis 88:365-411",
       doi="10.1016/S0047-259X(03)00096-4"),
    _c("barker2003", "Barker, M. & Rayens, W.", 2003,
       "Partial least squares for discrimination",
       "Journal of Chemometrics 17:166-173", doi="10.1002/cem.785"),
    _c("cortes1995", "Cortes, C. & Vapnik, V.", 1995,
       "Support-vector networks", "Machine Learning 20:273-297",
       doi="10.1007/BF00994018"),
    _c("bjorklund2023", "Bjorklund, A., Makela, J. & Puolamaki, K.", 2023,
       "SLISEMAP: supervised dimensionality reduction through local explanations",
       "Machine Learning 112:1-43", doi="10.1007/s10994-022-06261-1"),

    # ------------------------------------------- projection quality / caveats
    _c("venna2006", "Venna, J. & Kaski, S.", 2006,
       "Local multidimensional scaling", "Neural Networks 19:889-899",
       doi="10.1016/j.neunet.2006.05.014"),
    _c("lee2009", "Lee, J.A. & Verleysen, M.", 2009,
       "Quality assessment of dimensionality reduction: rank-based criteria",
       "Neurocomputing 72:1431-1443", doi="10.1016/j.neucom.2008.12.017"),
    _c("lee2013", "Lee, J.A. et al.", 2013,
       "Type 1 and 2 mixtures of Kullback-Leibler divergences as cost functions in "
       "dimensionality reduction based on similarity preservation",
       "Neurocomputing 112:92-108", doi="10.1016/j.neucom.2012.12.036"),
    _c("xia2024", "Xia, L., Lee, C. & Li, J.J.", 2024,
       "Statistical method scDEED for detecting dubious 2D single-cell embeddings "
       "and optimizing t-SNE and UMAP hyperparameters",
       "Nature Communications 15:1753", doi="10.1038/s41467-024-45891-y"),
    _c("gower1975", "Gower, J.C.", 1975,
       "Generalized Procrustes analysis", "Psychometrika 40:33-51",
       doi="10.1007/BF02291478"),
    _c("cardini2019", "Cardini, A., O'Higgins, P. & Rohlf, F.J.", 2019,
       "Seeing distinct groups where there are none: spurious patterns from "
       "between-group PCA", "Evolutionary Biology 46:303-316",
       doi="10.1007/s11692-019-09487-5"),
    _c("cardini2020", "Cardini, A. & Polly, P.D.", 2020,
       "Cross-validated between group PCA scatterplots: a solution to spurious "
       "group separation?", "Evolutionary Biology 47:85-95",
       doi="10.1007/s11692-020-09494-x"),
    _c("westerhuis2008", "Westerhuis, J.A. et al.", 2008,
       "Assessment of PLSDA cross validation", "Metabolomics 4:81-89",
       doi="10.1007/s11306-007-0099-6"),

    # ------------------------------------------------------------- statistics
    _c("anderson2001", "Anderson, M.J.", 2001,
       "A new method for non-parametric multivariate analysis of variance",
       "Austral Ecology 26:32-46", doi="10.1111/j.1442-9993.2001.01070.pp.x"),
    _c("anderson2006", "Anderson, M.J.", 2006,
       "Distance-based tests for homogeneity of multivariate dispersions",
       "Biometrics 62:245-253", doi="10.1111/j.1541-0420.2005.00440.x"),
    _c("anderson2003", "Anderson, M.J. & ter Braak, C.J.F.", 2003,
       "Permutation tests for multi-factorial analysis of variance",
       "Journal of Statistical Computation and Simulation 73:85-113",
       doi="10.1080/00949650215733"),
    _c("szekely2004", "Szekely, G.J. & Rizzo, M.L.", 2004,
       "Testing for equal distributions in high dimension",
       "InterStat 5:1249-1272"),
    _c("szekely2013", "Szekely, G.J. & Rizzo, M.L.", 2013,
       "Energy statistics: a class of statistics based on distances",
       "Journal of Statistical Planning and Inference 143:1249-1272",
       doi="10.1016/j.jspi.2013.03.018"),
    _c("gretton2012", "Gretton, A. et al.", 2012,
       "A kernel two-sample test",
       "Journal of Machine Learning Research 13:723-773"),
    _c("mahalanobis1936", "Mahalanobis, P.C.", 1936,
       "On the generalised distance in statistics",
       "Proceedings of the National Institute of Sciences of India 2:49-55"),
    _c("benjamini1995", "Benjamini, Y. & Hochberg, Y.", 1995,
       "Controlling the false discovery rate: a practical and powerful approach to "
       "multiple testing", "Journal of the Royal Statistical Society B 57:289-300",
       doi="10.1111/j.2517-6161.1995.tb02031.x"),
    _c("kruskal1952", "Kruskal, W.H. & Wallis, W.A.", 1952,
       "Use of ranks in one-criterion variance analysis",
       "Journal of the American Statistical Association 47:583-621",
       doi="10.1080/01621459.1952.10483441"),
    _c("cohen1988", "Cohen, J.", 1988,
       "Statistical Power Analysis for the Behavioral Sciences, 2nd edition",
       "Lawrence Erlbaum, Hillsdale NJ"),
    _c("cohen1960", "Cohen, J.", 1960,
       "A coefficient of agreement for nominal scales",
       "Educational and Psychological Measurement 20:37-46",
       doi="10.1177/001316446002000104"),
    _c("hedges1981", "Hedges, L.V.", 1981,
       "Distribution theory for Glass's estimator of effect size and related "
       "estimators", "Journal of Educational Statistics 6:107-128",
       doi="10.3102/10769986006002107"),
    _c("efron1987", "Efron, B.", 1987,
       "Better bootstrap confidence intervals",
       "Journal of the American Statistical Association 82:171-185",
       doi="10.1080/01621459.1987.10478410"),
    _c("breiman2001", "Breiman, L.", 2001,
       "Random forests", "Machine Learning 45:5-32", doi="10.1023/A:1010933404324"),
    _c("altmann2010", "Altmann, A. et al.", 2010,
       "Permutation importance: a corrected feature importance measure",
       "Bioinformatics 26:1340-1347", doi="10.1093/bioinformatics/btq134"),
    _c("ojala2010", "Ojala, M. & Garriga, G.C.", 2010,
       "Permutation tests for studying classifier performance",
       "Journal of Machine Learning Research 11:1833-1863"),
    _c("noirhomme2014", "Noirhomme, Q. et al.", 2014,
       "Biased binomial assessment of cross-validated estimation of classification "
       "accuracies illustrated in diagnosis predictions",
       "NeuroImage: Clinical 4:687-694", doi="10.1016/j.nicl.2014.04.004"),
    _c("brodersen2010", "Brodersen, K.H. et al.", 2010,
       "The balanced accuracy and its posterior distribution",
       "Proceedings of ICPR 2010, 3121-3124", doi="10.1109/ICPR.2010.764"),
    _c("haufe2014", "Haufe, S. et al.", 2014,
       "On the interpretation of weight vectors of linear models in multivariate "
       "neuroimaging", "NeuroImage 87:96-110",
       doi="10.1016/j.neuroimage.2013.10.067"),
    _c("hurlbert1984", "Hurlbert, S.H.", 1984,
       "Pseudoreplication and the design of ecological field experiments",
       "Ecological Monographs 54:187-211", doi="10.2307/1942661"),
    _c("lazic2018", "Lazic, S.E., Clarke-Williams, C.J. & Munafo, M.R.", 2018,
       "What exactly is 'N' in cell culture and animal experiments?",
       "PLoS Biology 16:e2005282", doi="10.1371/journal.pbio.2005282"),

    # ---------------------------------------------------------- visualization
    _c("lord2020", "Lord, S.J., Velle, K.B., Mullins, R.D. & Fritz-Laylin, L.K.", 2020,
       "SuperPlots: communicating reproducibility and variability in cell biology",
       "Journal of Cell Biology 219:e202001064", doi="10.1083/jcb.202001064"),
    _c("ho2019", "Ho, J., Tumkaya, T., Aryal, S., Choi, H. & Claridge-Chang, A.", 2019,
       "Moving beyond P values: data analysis with estimation graphics",
       "Nature Methods 16:565-566", doi="10.1038/s41592-019-0470-3"),
    _c("allen2021", "Allen, M. et al.", 2021,
       "Raincloud plots: a multi-platform tool for robust data visualization",
       "Wellcome Open Research 4:63", doi="10.12688/wellcomeopenres.15191.2"),
    _c("crameri2020", "Crameri, F., Shephard, G.E. & Heron, P.J.", 2020,
       "The misuse of colour in science communication",
       "Nature Communications 11:5444", doi="10.1038/s41467-020-19160-7"),
    _c("crameri2024", "Crameri, F.", 2024,
       "Choosing suitable color palettes for accessible and accurate science figures",
       "Current Protocols 4:e1126", doi="10.1002/cpz1.1126"),
    _c("wong2011", "Wong, B.", 2011,
       "Points of view: color blindness", "Nature Methods 8:441",
       doi="10.1038/nmeth.1618"),
    _c("okabe2008", "Okabe, M. & Ito, K.", 2008,
       "Color universal design (CUD): how to make figures and presentations that "
       "are friendly to colorblind people",
       "Jfly, University of Tokyo", url="https://jfly.uni-koeln.de/color/"),
    _c("glasbey2007", "Glasbey, C., van der Heijden, G., Toh, V.F.K. & Gray, A.", 2007,
       "Colour displays for categorical images",
       "Color Research and Application 32:304-309", doi="10.1002/col.20327"),
])


def cite(*keys: str) -> str:
    """``cite("lord2020", "ho2019")`` -> ``"Lord et al., 2020; Ho et al., 2019"``."""
    out = []
    for k in keys:
        c = CITATIONS.get(k)
        if c is not None:
            out.append(c.short)
    return "; ".join(out)


def reference_list(keys: Iterable[str]) -> list[Citation]:
    """De-duplicated citations, sorted by first author then year."""
    seen: dict[str, Citation] = {}
    for k in keys:
        c = CITATIONS.get(k)
        if c is not None:
            seen[c.key] = c
    return sorted(seen.values(), key=lambda c: (c.authors.lower(), c.year))


# ==========================================================================
# Per-run methods log
# ==========================================================================
@dataclass
class MethodsEntry:
    """One thing that actually happened during a run."""

    section: Section
    label: str                              # "SuperPlots"
    detail: str = ""                        # one clause describing what was done
    citations: tuple[str, ...] = ()
    params: dict = field(default_factory=dict)
    caveat: str = ""                        # printed as a warning where relevant

    def sentence(self) -> str:
        """``SuperPlots, coloured by replicate (Lord et al., 2020)``

        Returned as a :class:`~somtrack.i18n.Text`: the English sentence, which
        the translated report can render again in another language.  Citations
        are inserted verbatim -- a reference is not translated.
        """
        text = _translatable(self.label)
        if self.detail:
            text = Text("{label}, {detail}", label=text,
                        detail=_translatable(self.detail))
        refs = cite(*self.citations)
        if refs:
            text = Text("{text} ({refs})", text=text, refs=refs)
        return text

    def param_text(self) -> str:
        if not self.params:
            return ""
        bits = []
        for k, v in self.params.items():
            if isinstance(v, float):
                bits.append(f"{k} = {v:g}")
            else:
                bits.append(f"{k} = {v}")
        return ", ".join(bits)


def _translatable(value) -> Text:
    """A label or clause from a registry, made renderable in another language."""
    return value if isinstance(value, Text) else Text(str(value))


@dataclass
class MethodsLog:
    """Accumulates the methods used by one run, in the order they ran."""

    entries: list[MethodsEntry] = field(default_factory=list)

    def record(self, section: Section, label: str, detail: str = "",
               citations: Iterable[str] = (), caveat: str = "",
               **params) -> MethodsEntry:
        keys = tuple(citations)
        unknown = [k for k in keys if k not in CITATIONS]
        if unknown:                      # a typo in a citation key is a silent
            raise KeyError(              # bibliography hole; fail loudly instead
                f"Unknown citation key(s) {unknown} recorded for '{label}'. "
                f"Add them to somtrack.citations.CITATIONS."
            )
        entry = MethodsEntry(section=section, label=label, detail=detail,
                             citations=keys, params=dict(params), caveat=caveat)
        self.entries.append(entry)
        return entry

    # ------------------------------------------------------------------
    def by_section(self) -> dict[str, list[MethodsEntry]]:
        out: dict[str, list[MethodsEntry]] = {}
        for e in self.entries:
            out.setdefault(e.section, []).append(e)
        return {s: out[s] for s in SECTION_ORDER if s in out}

    def citation_keys(self) -> list[str]:
        keys: list[str] = []
        for e in self.entries:
            keys.extend(e.citations)
        return keys

    def references(self) -> list[Citation]:
        return reference_list(self.citation_keys())

    def caveats(self) -> list[tuple[str, str]]:
        return [(e.label, e.caveat) for e in self.entries if e.caveat]

    def extend(self, other: "MethodsLog") -> None:
        self.entries.extend(other.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def __bool__(self) -> bool:
        return bool(self.entries)


BASE_SOFTWARE = ("harris2020", "virtanen2020", "pedregosa2011",
                 "hunter2007", "mckinney2010")


__all__ = [
    "Citation", "CITATIONS", "cite", "reference_list",
    "MethodsEntry", "MethodsLog", "Section", "SECTION_ORDER", "BASE_SOFTWARE",
]
