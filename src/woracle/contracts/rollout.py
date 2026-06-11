"""Rollout and demo references — how episodes enter the pipeline.

Frames are NEVER embedded in JSON: a rollout points at a sidecar payload
(P0: ``frames.npz``; P1 adds mp4 via the [io] decode path).
"""

from __future__ import annotations

from pydantic import Field

from woracle.contracts.base import ArtifactRef, VersionedModel


class RolloutRef(VersionedModel):
    """A single episode (generated rollout or demonstration)."""

    id: str
    frames: ArtifactRef  # (T, H, W, 3) uint8 payload
    fps: float = 10.0
    n_frames: int = 0
    policy: str = ""  # which policy produced it ("" for demos)
    source: str = ""  # e.g. "wm:cosmos3-nano", "real", "blobworld"
    actions: ArtifactRef | None = None  # optional (T, A) action stream
    meta: dict[str, str] = Field(default_factory=dict)


class DemoSet(VersionedModel):
    """A small set of demonstrations for one task (Stage-1 input)."""

    task: str
    demos: list[RolloutRef]
    labels: dict[str, str] = Field(default_factory=dict)  # rollout id -> "success"|"fail"
