"""Ordered phase coverage — did the task's phases happen, in order?

ORCA's core insight, reimplemented over our predicate engine (the published
ORCA operates on embedding-frame matching; ours operates on spec phases — the
relation to ORCA is the ORDERED-coverage idea, and naive DTW/OT's known
failure of rewarding order-violating trajectories is exactly what the
ordering term prevents). Rank-only: phase coverage is progress evidence,
never the verdict.

score = (phases satisfied in non-decreasing first-satisfaction order) / n_phases
"""

from __future__ import annotations

import numpy as np

from woracle.channels.predicates import eval_conjunction
from woracle.contracts import ChannelCaps, ChannelScore, GroundedRollout, TaskSpec
from woracle.registry import register


@register("channel", "phase.ordered_coverage")
class OrderedPhaseCoverageChannel:
    name = "phase.ordered_coverage"
    version = "0.1.0"
    caps = ChannelCaps(reference_free=True, needs_tracks=True, verdict_eligible=False)

    def __init__(self, window: int = 5, stride: int = 2) -> None:
        self.window = int(window)
        self.stride = int(stride)

    def score(self, grounded: GroundedRollout, spec: TaskSpec) -> ChannelScore:
        if not spec.phases:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="spec declares no phases",
            )
        from woracle.channels.verdict import role_data

        roles = role_data(grounded)
        lengths = [len(r.track) for r in roles.values() if r.track is not None]
        if not lengths:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="no role tracks available",
            )
        T = max(lengths)
        phases = sorted(spec.phases, key=lambda p: p.order)

        # SEQUENTIAL ordered coverage (the ORCA insight): phase i is searched
        # only from its predecessor's satisfaction time onward — "stationary
        # at t=0" must never satisfy a terminal settle phase.
        first_sat: list[int | None] = []
        any_window_evaluable = False
        cursor = 0
        for phase in phases:
            t_sat: int | None = None
            if phase.active:
                for t0 in range(cursor, max(cursor + 1, T - self.window), self.stride):
                    window = slice(t0, min(T, t0 + self.window))
                    results = eval_conjunction(phase.active, roles, window)
                    if any(r.status == "evidence_missing" for r in results):
                        continue
                    any_window_evaluable = True
                    if all(r.holds for r in results):
                        t_sat = t0
                        break
            first_sat.append(t_sat)
            if t_sat is not None:
                cursor = t_sat  # later phases must satisfy at-or-after this

        if not any_window_evaluable:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="no phase predicate was evaluable in any window",
            )

        credited = sum(1 for t in first_sat if t is not None)
        value = credited / len(phases)
        details = {
            f"t_first:{p.name}": float(t) if t is not None else -1.0
            for p, t in zip(phases, first_sat, strict=True)
        }
        unmet = [p.name for p, t in zip(phases, first_sat, strict=True) if t is None]
        return ChannelScore(
            channel=self.name,
            value=float(value),
            confidence=float(np.mean([t is not None for t in first_sat])),
            reason=f"phases never satisfied in order: {', '.join(unmet)}" if unmet else "",
            details=details,
        )
