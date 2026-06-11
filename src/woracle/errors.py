"""Woracle error taxonomy.

The load-bearing distinction (see ARCH §6):

* ``InfraError`` — the machinery broke (timeout, OOM, download failure, worker
  death). Retryable; never recorded as evidence about the rollout.
* **Evidence failure is NOT an exception.** "The tracker lost the object" or
  "the role never appeared" is *data* — it is recorded in
  :class:`woracle.contracts.gate.GateSignalValue` with
  ``status="evidence_missing"`` and flows into abstain accounting.

Raising an exception for missing evidence would silently turn honesty into a
crash; recording an exception's traceback as evidence would silently turn a bug
into a verdict. Keep the two lanes separate.
"""

from __future__ import annotations


class WoracleError(Exception):
    """Base class for all woracle errors."""


class InfraError(WoracleError):
    """The machinery failed (retryable). Not evidence about the rollout."""


class SpecError(WoracleError):
    """A task spec is invalid, unreadable, or failed to compile."""


class PluginError(WoracleError):
    """A plugin could not be registered, loaded, or violated its contract."""


class StoreError(WoracleError):
    """The artifact store is corrupt, unwritable, or a key is malformed."""


class MissingDependencyError(WoracleError):
    """A heavy optional dependency is required for this call.

    Raised at *call time* (never import time) and names the exact extra,
    transformers-style.
    """

    def __init__(self, feature: str, extra: str) -> None:
        super().__init__(
            f"{feature} requires the '{extra}' extra. Install with: pip install 'woracle[{extra}]'"
        )
        self.feature = feature
        self.extra = extra
