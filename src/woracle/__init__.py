"""Woracle — the world-model oracle. Demos in, oracle out.

Lazy top-level: ``import woracle`` stays milliseconds and torch-free (the
kernel rule, ARCH §1). Public names resolve on first attribute access via
PEP 562; the TYPE_CHECKING mirror keeps IDEs and type checkers accurate.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from woracle._version import __version__

if TYPE_CHECKING:
    from woracle.api import compile as compile
    from woracle.api import grade as grade
    from woracle.api import ground as ground
    from woracle.api import load_spec as load_spec
    from woracle.api import report as report
    from woracle.contracts import GradeCard as GradeCard
    from woracle.contracts import Leaderboard as Leaderboard
    from woracle.contracts import TaskSpec as TaskSpec

_LAZY = {
    "compile": ("woracle.api", "compile"),
    "grade": ("woracle.api", "grade"),
    "ground": ("woracle.api", "ground"),
    "load_spec": ("woracle.api", "load_spec"),
    "report": ("woracle.api", "report"),
    "GradeCard": ("woracle.contracts", "GradeCard"),
    "Leaderboard": ("woracle.contracts", "Leaderboard"),
    "TaskSpec": ("woracle.contracts", "TaskSpec"),
}

__all__ = ["__version__", *sorted(_LAZY)]


def __getattr__(name: str) -> object:
    try:
        module_name, attr = _LAZY[name]
    except KeyError:
        raise AttributeError(f"module 'woracle' has no attribute {name!r}") from None
    import importlib

    value = getattr(importlib.import_module(module_name), attr)
    globals()[name] = value  # cache for subsequent access
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY))
