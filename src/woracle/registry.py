"""Component registry: flat, per-kind, decorator-based + entry-point discovery.

Deliberately boring (ARCH decision 3): a dict per kind, a decorator, and a
defensive entry-points loader. No scopes, no parents, no config-string
instantiation — the mmengine issue tracker is the cautionary tale.

First-party components register at import time via ``@register(kind, name)``.
Third-party packages expose an entry point in group ``woracle.plugins``::

    [project.entry-points."woracle.plugins"]
    my_pkg = "my_pkg.woracle_plugin:register"

whose target is a zero-arg callable that performs its own ``register`` calls.
A broken plugin is reported, never fatal (pytest's `-p no:name` lesson).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from woracle.errors import PluginError

if TYPE_CHECKING:
    from collections.abc import Callable

KINDS = ("grounder", "gate_signal", "channel", "judge")

_REGISTRY: dict[str, dict[str, Any]] = {k: {} for k in KINDS}

ENTRY_POINT_GROUP = "woracle.plugins"
DISABLE_ENV = "WORACLE_DISABLE_PLUGINS"  # "1" => skip entry-point autoload


def register(kind: str, name: str) -> Callable[[type], type]:
    """Class decorator: ``@register("channel", "blob.progress")``."""
    if kind not in KINDS:
        raise PluginError(f"unknown component kind '{kind}' (kinds: {KINDS})")

    def deco(cls: type) -> type:
        if name in _REGISTRY[kind]:
            existing = _REGISTRY[kind][name]
            if existing is cls:  # idempotent re-import
                return cls
            raise PluginError(
                f"{kind} '{name}' already registered by {existing.__module__}.{existing.__qualname__}"
            )
        _REGISTRY[kind][name] = cls
        return cls

    return deco


def get(kind: str, name: str) -> Any:
    if kind not in KINDS:
        raise PluginError(f"unknown component kind '{kind}'")
    try:
        return _REGISTRY[kind][name]
    except KeyError:
        available = ", ".join(sorted(_REGISTRY[kind])) or "<none>"
        raise PluginError(
            f"no {kind} named '{name}' is registered (available: {available}). "
            f"Did you forget to install/import the providing package?"
        ) from None


def available(kind: str) -> list[str]:
    if kind not in KINDS:
        raise PluginError(f"unknown component kind '{kind}'")
    return sorted(_REGISTRY[kind])


def clear(kind: str | None = None) -> None:
    """Testing helper — wipe registrations."""
    for k in KINDS if kind is None else (kind,):
        _REGISTRY[k].clear()


@dataclass
class PluginLoadReport:
    loaded: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)  # ep name -> error
    skipped: bool = False


def load_entry_points(group: str = ENTRY_POINT_GROUP) -> PluginLoadReport:
    """Discover and invoke third-party plugin registrars. Defensive by design."""
    report = PluginLoadReport()
    if os.environ.get(DISABLE_ENV) == "1":
        report.skipped = True
        return report
    from importlib.metadata import entry_points

    for ep in entry_points(group=group):
        try:
            registrar = ep.load()
            registrar()
            report.loaded.append(ep.name)
        except Exception as e:
            report.failed[ep.name] = f"{type(e).__name__}: {e}"
    return report
