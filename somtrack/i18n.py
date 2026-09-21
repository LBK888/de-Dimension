"""Interface language, and sentences that carry their own translation.

Two kinds of text need translating, and they need different machinery.

*Interface text* -- a button, a hint, a tooltip -- is settled when the widget is
built, in whatever language the app was started in::

    QPushButton(tr("Run analysis"))
    tr("{n} metric(s) selected").format(n=n)

*Generated text* -- the verdict, the methods paragraph, a note about dropped
samples -- is written while the analysis runs, long before anyone decides which
language to read it in, and the report needs it in two languages at once.
:class:`Text` handles that.  It **is** the English sentence (a ``str``), so
figures, CSV files, the command line and every existing comparison see exactly
what they saw before; it also remembers its template and arguments, so the
report can render the same sentence again in Chinese::

    note = Text("Dropped {n} sample(s) containing NaN.", n=3)
    note                      # 'Dropped 3 sample(s) containing NaN.'
    note.render("zh_TW")      # '已移除 3 個含有缺失值（NaN）的樣本。'

Arguments that are themselves :class:`Text` are rendered in the same language;
plain strings (group names, metric names, file names) are inserted verbatim,
because they are data rather than prose.

The English source string is the message id, as in gettext.  Translations live
in :mod:`somtrack.locales`, one module per language.  A string with no
translation falls back to English and is recorded, so the test suite can list
exactly what is missing instead of it being found by a user.
"""

from __future__ import annotations

import importlib
import os
from string import Formatter
from typing import Any, Iterable

DEFAULT = "en"

#: language code -> name shown in the language menu (in its own language)
LANGUAGES: dict[str, str] = {
    "en": "English",
    "zh_TW": "繁體中文（台灣）",
}

#: language code -> value for an HTML ``lang`` attribute
HTML_LANG: dict[str, str] = {"en": "en", "zh_TW": "zh-Hant-TW"}

_state: dict[str, str] = {"lang": DEFAULT}
_catalogs: dict[str, tuple[dict, dict]] = {}
_missing: dict[str, set[str]] = {}


# ==========================================================================
# The current language
# ==========================================================================
def normalise(code: str | None) -> str:
    """``"zh-TW"``, ``"zh_Hant"``, ``"zh_HK"`` -> ``"zh_TW"``; anything unknown -> ``"en"``."""
    if not code:
        return DEFAULT
    c = str(code).strip().replace("-", "_")
    low = c.lower()
    if low.startswith("zh"):
        # Simplified Chinese is not offered; Traditional is the closer fit for
        # Hong Kong and Macau, and a Simplified-Chinese reader still does better
        # with it than with English.
        return "zh_TW"
    if c in LANGUAGES:
        return c
    return DEFAULT


def language() -> str:
    return _state["lang"]


def set_language(code: str | None) -> str:
    """Switch the process-wide language; returns the code actually used."""
    _state["lang"] = normalise(code)
    return _state["lang"]


def language_from_environment() -> str | None:
    """``SOMTRACK_LANG`` if it is set, so a lab can pin the language per machine."""
    value = os.environ.get("SOMTRACK_LANG", "").strip()
    return normalise(value) if value else None


# ==========================================================================
# Catalogue lookup
# ==========================================================================
def _catalog(lang: str) -> tuple[dict, dict]:
    if lang not in _catalogs:
        try:
            mod = importlib.import_module(f"{__package__}.locales.{lang}")
            _catalogs[lang] = (dict(getattr(mod, "MESSAGES", {})),
                               {k: dict(v) for k, v in
                                getattr(mod, "CONTEXTS", {}).items()})
        except ImportError:
            _catalogs[lang] = ({}, {})
    return _catalogs[lang]


def lookup(text: str, lang: str, context: str = "") -> str:
    """The translation of one message id, or the id itself if there is none."""
    if lang == DEFAULT or not text:
        return text
    messages, contexts = _catalog(lang)
    if context:
        hit = contexts.get(context, {}).get(text)
        if hit is not None:
            return hit
    hit = messages.get(text)
    if hit is not None:
        return hit
    # Text that is already rendered (a log line passed through render() twice)
    # is not a missing translation, so only English-looking text is recorded.
    if any(ch.isalpha() for ch in text) and not _has_cjk(text):
        _missing.setdefault(lang, set()).add(f"{context}|{text}" if context else text)
    return text


def _has_cjk(text: str) -> bool:
    return any("　" <= ch <= "鿿" or "＀" <= ch <= "￯" for ch in text)


def tr(text: str, context: str = "", lang: str | None = None) -> str:
    """Interface text in the current (or the given) language.

    ``context`` separates short words that translate differently in different
    places -- ``"sample"`` the unit of analysis versus ``"sample"`` the SOM
    initialisation.
    """
    lang = normalise(lang) if lang else language()
    if isinstance(text, Text):
        return text.render(lang)
    return lookup(str(text), lang, context)


def render(obj: Any, lang: str | None = None) -> str:
    """Anything textual, in the requested language.

    A :class:`Text` re-renders itself; a plain string is looked up in the
    catalogue, which covers the fixed sentences held in registries (method
    summaries, model names, caveats); anything else is ``str()``-ed.
    """
    lang = normalise(lang) if lang else language()
    if isinstance(obj, Text):
        return obj.render(lang)
    if isinstance(obj, str):
        return lookup(obj, lang)
    return "" if obj is None else str(obj)


def missing(lang: str = "zh_TW") -> set[str]:
    """Message ids looked up in ``lang`` this session that had no translation."""
    return set(_missing.get(lang, set()))


def clear_missing() -> None:
    _missing.clear()


def placeholders(template: str) -> set[str]:
    """The ``{field}`` names in a format string -- used to check translations."""
    out: set[str] = set()
    try:
        for _, field, _, _ in Formatter().parse(template):
            if field is not None:
                out.add(field.split(".")[0].split("[")[0])
    except ValueError:
        pass
    return out


# ==========================================================================
# Generated text
# ==========================================================================
def _arg(value: Any, lang: str) -> Any:
    return value.render(lang) if isinstance(value, Text) else value


def _fill(template: str, args: tuple, kwargs: dict, lang: str) -> str:
    if not args and not kwargs:
        return template
    return template.format(*(_arg(a, lang) for a in args),
                           **{k: _arg(v, lang) for k, v in kwargs.items()})


class Text(str):
    """An English sentence that can render itself again in another language.

    ``Text(template, *args, **kwargs)`` formats ``template`` with the arguments
    and *is* the resulting English string.  :meth:`render` looks the template up
    in a language's catalogue and formats the translation with the same
    arguments.  ``Text("euclidean")`` with no arguments is simply a
    translatable word.
    """

    template: str
    args: tuple
    kwargs: dict

    def __new__(cls, template: str, /, *args: Any, **kwargs: Any) -> "Text":
        template = str.__str__(template) if isinstance(template, str) else str(template)
        self = super().__new__(cls, _fill(template, args, kwargs, DEFAULT))
        self.template = template
        self.args = args
        self.kwargs = kwargs
        return self

    def __getnewargs_ex__(self):                 # pickle / deepcopy
        return (self.template, *self.args), dict(self.kwargs)

    def render(self, lang: str | None = None) -> str:
        lang = normalise(lang) if lang else language()
        if lang == DEFAULT:
            return str.__str__(self)
        return _fill(lookup(self.template, lang), self.args, self.kwargs, lang)

    @property
    def english(self) -> str:
        return str.__str__(self)

    def __repr__(self) -> str:
        return f"Text({str.__repr__(self)})"


class Joined(Text):
    """Several pieces of text with a separator that may depend on the language.

    Chinese runs sentences together without a space and lists items with
    ``、`` rather than a comma, so ``Joined(caveats, " ", {"zh_TW": ""})``.
    """

    parts: list
    sep: str
    seps: dict

    def __new__(cls, parts: Iterable[Any], sep: str = " ",
                seps: dict[str, str] | None = None) -> "Joined":
        parts = list(parts)
        english = sep.join(p.render(DEFAULT) if isinstance(p, Text) else str(p)
                           for p in parts)
        self = str.__new__(cls, english)
        self.template = english
        self.args = ()
        self.kwargs = {}
        self.parts = parts
        self.sep = sep
        self.seps = dict(seps or {})
        return self

    def __getnewargs_ex__(self):
        return (self.parts, self.sep, self.seps), {}

    def render(self, lang: str | None = None) -> str:
        lang = normalise(lang) if lang else language()
        if lang == DEFAULT:
            return str.__str__(self)
        sep = self.seps.get(lang, self.sep)
        return sep.join(p.render(lang) if isinstance(p, Text) else str(p)
                        for p in self.parts)


class Clause(Text):
    """A sentence with its closing full stop removed, in whichever language.

    Used where a complete sentence is reused as a clause inside a longer one,
    e.g. a design note that becomes part of a methods sentence.
    """

    inner: Any

    def __new__(cls, inner: Any) -> "Clause":
        english = (inner.render(DEFAULT) if isinstance(inner, Text)
                   else str(inner)).rstrip().rstrip(".")
        self = str.__new__(cls, english)
        self.template = english
        self.args = ()
        self.kwargs = {}
        self.inner = inner
        return self

    def __getnewargs_ex__(self):
        return (self.inner,), {}

    def render(self, lang: str | None = None) -> str:
        lang = normalise(lang) if lang else language()
        if lang == DEFAULT:
            return str.__str__(self)
        full = render(self.inner, lang)
        return full.rstrip().rstrip(".。")


#: separators for the common kinds of list, per language
LIST_SEP = {"zh_TW": "、"}           # ", " between names
CLAUSE_SEP = {"zh_TW": "；"}         # "; " between clauses
SENTENCE_SEP = {"zh_TW": ""}         # " " between sentences


def join_list(items: Iterable[Any]) -> Joined:
    """``a, b, c`` / ``a、b、c``."""
    return Joined(items, ", ", LIST_SEP)


def join_clauses(items: Iterable[Any]) -> Joined:
    """``a; b; c`` / ``a；b；c``."""
    return Joined(items, "; ", CLAUSE_SEP)


def join_sentences(items: Iterable[Any]) -> Joined:
    """Sentences run together: a space in English, nothing in Chinese."""
    return Joined(items, " ", SENTENCE_SEP)


__all__ = [
    "DEFAULT", "LANGUAGES", "HTML_LANG", "normalise", "language", "set_language",
    "language_from_environment", "lookup", "tr", "render", "missing",
    "clear_missing", "placeholders", "Text", "Joined", "Clause", "join_list",
    "join_clauses", "join_sentences", "LIST_SEP", "CLAUSE_SEP", "SENTENCE_SEP",
]
