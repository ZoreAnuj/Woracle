"""Scoring contracts — channel scores, the GradeCard snapshot, leaderboards.

GradeCard follows the evidently snapshot model (ARCH decision 5): it is a
self-contained JSON document holding every computed value, the gate report,
and full provenance. Reports/HTML render FROM the snapshot with zero
recomputation, and ``compare()`` over snapshots builds leaderboards.

Verdict isolation (ARCH decision 6 + proposal §4): only channels flagged
``verdict_eligible`` may influence ``success``; ranking-only channels
(trajectory, quality) are structurally walled off from the verdict.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from woracle.contracts.base import Provenance, VersionedModel, WoracleModel
from woracle.contracts.gate import GateReport

ChannelStatus = Literal["ok", "evidence_missing", "skipped"]
SuccessVerdict = Literal["pass", "fail", "abstain"]


class ChannelCaps(WoracleModel):
    """Capabilities a channel declares; drives conformance checks + planning."""

    reference_free: bool = True
    needs_actions: bool = False
    needs_masks: bool = False
    needs_tracks: bool = True
    verdict_eligible: bool = False  # may this channel feed the success verdict?
    value_range: tuple[float, float] = (0.0, 1.0)


class ChannelScore(WoracleModel):
    channel: str
    status: ChannelStatus = "ok"
    value: float | None = None  # None unless status == "ok"
    confidence: float | None = None  # channel's own confidence in its value
    reason: str = ""
    # Small, plottable extras only (e.g. per-frame curve). Big arrays -> sidecars.
    series: dict[str, list[float]] = Field(default_factory=dict)
    details: dict[str, float] = Field(default_factory=dict)


class SuccessReport(WoracleModel):
    verdict: SuccessVerdict
    probability: float | None = None  # calibrated P(success) when available (P5)
    satisfied: list[str] = Field(default_factory=list)  # predicate descriptions
    violated: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)


class GradeCard(VersionedModel):
    """The per-rollout snapshot. Self-contained; renderable without recompute."""

    rollout_id: str
    policy: str = ""
    spec_name: str
    spec_version: int
    spec_hash: str
    gate: GateReport
    channels: list[ChannelScore] = Field(default_factory=list)
    success: SuccessReport
    provenance: Provenance = Field(default_factory=Provenance)

    def channel(self, name: str) -> ChannelScore | None:
        for c in self.channels:
            if c.channel == name:
                return c
        return None


class PolicySummary(WoracleModel):
    policy: str
    n_rollouts: int
    n_gradeable: int
    n_abstained: int
    pass_rate_on_graded: float | None = None  # NOT a calibrated success rate (P5)
    mean_channel_values: dict[str, float] = Field(default_factory=dict)


class Leaderboard(VersionedModel):
    """Comparison across policies, assembled from GradeCards (snapshots only).

    Honesty invariants surfaced at this level:
    * abstention rate is reported PER POLICY (informative missingness);
    * pass rates are labeled as on-graded-rollouts-only until P5 calibration.
    """

    spec_name: str
    spec_hash: str
    policies: list[PolicySummary] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)
