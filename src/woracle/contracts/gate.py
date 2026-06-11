"""Gate contracts — validity signals and the three-way gradeability verdict.

deepchecks-style separation (ARCH decision 6): signals *measure*, the gate
*composes* measurements into a verdict, thresholds live in ``GatePolicy`` —
never inside signal code.

Honesty semantics:
* ``status="ok"``               — the signal measured something.
* ``status="evidence_missing"`` — the signal could not measure (role unbound,
  payload absent). This is DATA, not an error; it feeds abstention.
Infra failures raise ``InfraError`` and never appear as signal values.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from woracle.contracts.base import VersionedModel, WoracleModel

SignalStatus = Literal["ok", "evidence_missing"]
GateVerdict = Literal["gradeable", "degraded", "ungradeable"]


class GateSignalValue(WoracleModel):
    name: str
    status: SignalStatus = "ok"
    value: float | None = None  # None iff evidence_missing
    # Direction convention: higher = healthier. Signals must normalize to this.
    reason: str = ""
    details: dict[str, float] = Field(default_factory=dict)


class GateThreshold(WoracleModel):
    signal: str
    ungradeable_below: float | None = None
    degraded_below: float | None = None


class GatePolicy(WoracleModel):
    """Composition rules. P0: static thresholds; P2 adds conformal calibration."""

    thresholds: list[GateThreshold] = Field(default_factory=list)
    # evidence_missing on a signal listed here => ungradeable
    required_signals: list[str] = Field(default_factory=list)
    # unbound required role => ungradeable (set False only for debugging)
    require_role_bindings: bool = True

    def threshold(self, signal: str) -> GateThreshold | None:
        for t in self.thresholds:
            if t.signal == signal:
                return t
        return None


class GateReport(VersionedModel):
    verdict: GateVerdict
    signals: list[GateSignalValue] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    policy_digest: str = ""

    @property
    def gradeable(self) -> bool:
        return self.verdict != "ungradeable"
