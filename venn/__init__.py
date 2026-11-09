"""Venn: typed decisions over embeddings.

Public entry points:

- :class:`OutcomeService`: decisions from an application-owned store of past cases
  and their outcomes; runs in-process with no trained head, and can compile a store
  into a small head that answers the same question without a lookup.
- :class:`DecisionService` and :func:`create_app`: the Choice/Score/Noul decision
  engine and its HTTP application, backed by the ``venn-1`` head.
- :class:`VectorTag`, :data:`TEXT_TAG`, :data:`VENN_1`: the profile and tag types
  that every vector must carry.

The engine lives in :mod:`venn.server`, :mod:`venn.heads`, :mod:`venn.index`, :mod:`venn.reader`
and :mod:`venn.runtime`.
"""

from importlib.metadata import PackageNotFoundError, version

from venn.server import DecisionService, create_app
from venn.outcomes import OutcomeService
from venn.profiles import IMAGE_TAGS, TEXT_TAG, VENN_1, VectorTag

try:
    __version__ = version("venn-vdm")
except PackageNotFoundError:
    __version__ = "0+source"
__all__ = ["DecisionService", "OutcomeService", "create_app", "VectorTag", "TEXT_TAG", "IMAGE_TAGS", "VENN_1"]
