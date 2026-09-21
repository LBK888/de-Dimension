"""Projection methods.

Importing any of these modules registers its methods; the registry imports them
all on first use, so nothing here needs to be listed anywhere else.
"""

from . import linear, manifold, supervised  # noqa: F401

__all__ = ["linear", "manifold", "supervised"]
