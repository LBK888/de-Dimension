"""Translation catalogues, one module per language.

Each module defines

``MESSAGES``
    English source string -> translation.  The English string is the message
    id, exactly as it appears in the code (``tr("...")`` / ``Text("...")``) or
    in a registry (method summaries, metric descriptions, recipes).
``CONTEXTS``
    context -> {English -> translation}, for short words that translate
    differently in different places -- the option lists of a combo box, where
    the value itself is also a configuration key.

A translation must keep every ``{placeholder}`` of its source; the test suite
checks that, and lists any message the program asked for that has no
translation.
"""
