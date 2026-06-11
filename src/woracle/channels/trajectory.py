"""Trajectory channel: DTW distance of a role's track against demo references.

Rank-only by design (verdict isolation): "moves like the demos" is ranking
evidence, never a success verdict. Plain O(nm) dynamic-time-warping on
normalized 2-D tracks — ~40 lines we own outright (no license traps), exact,
and fast enough for tracks of a few thousand points.
"""

from __future__ import annotations

import numpy as np

from woracle.contracts import ChannelCaps, ChannelScore, GroundedRollout, TaskSpec
from woracle.errors import SpecError
from woracle.registry import register


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Classic DTW between (n,2) and (m,2) sequences; mean per-step cost along
    the optimal path (length-normalized, symmetric in a/b)."""
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[1]:
        raise SpecError("dtw_distance expects (n,d) and (m,d) sequences")
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        raise SpecError("dtw_distance: empty sequence")
    cost = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    acc = np.full((n + 1, m + 1), np.inf)
    acc[0, 0] = 0.0
    for i in range(1, n + 1):
        acc[i, 1:] = cost[i - 1]
        for j in range(1, m + 1):
            acc[i, j] += min(acc[i - 1, j], acc[i, j - 1], acc[i - 1, j - 1])
    # path length is bounded by n+m; normalize by that bound for comparability
    return float(acc[n, m] / (n + m))


def _normalize(track: np.ndarray) -> np.ndarray:
    """Drop NaNs, translate to start-at-origin, scale by trajectory extent —
    makes the comparison camera-shift tolerant (viewpoint is the #1 transfer
    killer; full invariance is future work, stated plainly)."""
    pts = track[np.isfinite(track[:, 0])]
    if len(pts) < 2:
        raise SpecError("not enough observed points to normalize")
    pts = pts - pts[0]
    extent = float(np.abs(pts).max())
    return pts / max(extent, 1e-6)


@register("channel", "trajectory.dtw_demo")
class TrajectoryDTWChannel:
    """Min normalized DTW distance to ANY demo reference track of the role."""

    name = "trajectory.dtw_demo"
    version = "0.1.0"
    caps = ChannelCaps(
        reference_free=False,
        needs_tracks=True,
        verdict_eligible=False,
        value_range=(0.0, 1.0),
    )

    def __init__(
        self,
        demo_tracks: list[np.ndarray] | None = None,
        role: str = "carried_object",
        scale: float = 0.25,
    ) -> None:
        self.demo_tracks = demo_tracks or []
        self.role = role
        self.scale = float(scale)

    def score(self, grounded: GroundedRollout, spec: TaskSpec) -> ChannelScore:
        if not self.demo_tracks:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="no demo reference tracks configured (reference-based channel)",
            )
        from woracle.testing.plugins import role_data

        rd = role_data(grounded).get(self.role)
        if rd is None or rd.track is None:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason=f"no track for role '{self.role}'",
            )
        try:
            q = _normalize(rd.track)
            dists = [dtw_distance(q, _normalize(ref)) for ref in self.demo_tracks]
        except SpecError as e:
            return ChannelScore(channel=self.name, status="evidence_missing", reason=str(e))
        d = float(min(dists))
        value = float(np.exp(-d / self.scale))
        return ChannelScore(
            channel=self.name,
            value=value,
            confidence=float(np.isfinite(rd.track[:, 0]).mean()),
            details={"min_dtw": round(d, 4), "n_refs": float(len(self.demo_tracks))},
        )
