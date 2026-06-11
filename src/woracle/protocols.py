"""Plugin contracts (typing.Protocol — structural, at the boundary).

Third-party plugins do NOT need to import a woracle base class to comply:
anything with these attributes/methods is a valid plugin (lm-eval's 3-method
``LM`` philosophy). Conformance is enforced by the published check suite in
``woracle.testing.conformance``, not by ``isinstance`` at runtime.

Contract obligations every plugin shares:
* deterministic given identical inputs + params (cache correctness);
* evidence problems -> recorded in the returned contract object
  (``bound=False`` / ``status="evidence_missing"``), never raised;
* infra problems -> raise ``InfraError``;
* ``name`` is stable and unique within its kind; ``version`` participates in
  cache keys — bump it on ANY behavior change.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from woracle.contracts import (
        ChannelCaps,
        ChannelScore,
        GateSignalValue,
        GroundedRollout,
        RolloutRef,
        TaskSpec,
    )


class Grounder(Protocol):
    """Stage-2: bind spec roles to a rollout's pixels; write sidecar artifacts."""

    name: str
    version: str

    def ground(self, rollout: RolloutRef, spec: TaskSpec, out_dir: str) -> GroundedRollout: ...


class GateSignal(Protocol):
    """Stage-3 ingredient: one structural health measurement (higher=healthier)."""

    name: str
    version: str

    def measure(self, grounded: GroundedRollout) -> GateSignalValue: ...


class Channel(Protocol):
    """Stage-4: score one dimension of a grounded rollout against the spec."""

    name: str
    version: str
    caps: ChannelCaps

    def score(self, grounded: GroundedRollout, spec: TaskSpec) -> ChannelScore: ...
