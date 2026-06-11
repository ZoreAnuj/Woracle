"""Binding — the output of Stage-2 grounding: roles bound to THIS scene.

A ``RoleBinding`` records *whether and how well* a spec role was located in the
rollout's pixels, plus refs to the per-frame artifacts (tracks, masks) that the
pure downstream stages (gate/channels) consume.

Honesty contract: a role that could not be bound is **not** an exception — it
is a ``RoleBinding`` with ``bound=False`` and a reason, which the gate turns
into abstention if the role was required.
"""

from __future__ import annotations

from pydantic import Field

from woracle.contracts.base import ArtifactRef, VersionedModel, WoracleModel
from woracle.contracts.rollout import RolloutRef


class RoleBinding(WoracleModel):
    role: str
    bound: bool
    # Copied from the spec role at grounding time so pure downstream stages
    # (gate, signals) can be required-aware without re-reading the spec.
    required: bool = True
    # [0,1] — detection/persistence RATE only. The binding study measured
    # detector confidence as content-blind on generated video (absent-object
    # inversion): NEVER read quality as phrase fidelity or correctness.
    quality: float = 0.0
    reason: str = ""  # why not bound / quality notes
    # Sidecar payloads, relative to the bundle dir:
    #   tracks: (T, 2) float32 centroid track (x, y), NaN where unobserved
    #   masks:  (T, H, W) bool or uint8 occupancy
    tracks: ArtifactRef | None = None
    masks: ArtifactRef | None = None
    # Per-frame visibility/confidence (T,) float32 — abstention raw signal.
    visibility: ArtifactRef | None = None


class GroundedRollout(VersionedModel):
    """Everything Stage-2 produced for one rollout. Pure stages read only this."""

    rollout: RolloutRef
    spec_name: str
    spec_hash: str
    bindings: list[RoleBinding]
    grounder: str = ""  # component name@version
    bundle_dir: str = ""  # where sidecars live (set on save/load)
    extras: dict[str, str] = Field(default_factory=dict)

    def binding(self, role: str) -> RoleBinding:
        for b in self.bindings:
            if b.role == role:
                return b
        from woracle.errors import PluginError

        raise PluginError(f"grounder produced no binding for role '{role}'")
