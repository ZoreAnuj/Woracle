"""The verdict channel + grounded-artifact loaders (production layer).

Moved out of woracle.testing (critic finding: the ONLY verdict-eligible
channel and the canonical artifact loader were living in the testing package
— a layering inversion). ``woracle.testing.plugins`` re-exports both for
backward compatibility.
"""

from __future__ import annotations

import os

import numpy as np

from woracle.channels.predicates import RoleData, eval_conjunction
from woracle.contracts import (
    ArtifactRef,
    ChannelCaps,
    ChannelScore,
    GroundedRollout,
    TaskSpec,
)
from woracle.registry import register


def _load_npz(grounded: GroundedRollout, ref: ArtifactRef | None, key: str) -> np.ndarray | None:
    if ref is None:
        return None
    path = ref.resolve(grounded.bundle_dir)
    if not os.path.isfile(path):
        return None
    with np.load(path) as z:
        return z[key]


def role_data(grounded: GroundedRollout) -> dict[str, RoleData]:
    """Materialize contract-level role data from a grounded bundle's sidecars."""
    out: dict[str, RoleData] = {}
    for b in grounded.bindings:
        out[b.role] = RoleData(
            track=_load_npz(grounded, b.tracks, "track"),
            mask=_load_npz(grounded, b.masks, "mask"),
            visibility=_load_npz(grounded, b.visibility, "vis"),
        )
    return out


@register("channel", "success.predicates")
class PredicateSuccessChannel:
    """Verdict channel: the spec's success conjunction over the final window."""

    name = "success.predicates"
    version = "0.1.0"
    caps = ChannelCaps(
        reference_free=True, needs_tracks=True, needs_masks=True, verdict_eligible=True
    )

    def score(self, grounded: GroundedRollout, spec: TaskSpec) -> ChannelScore:
        roles = role_data(grounded)
        # Window length comes from the ARTIFACTS, not trusted metadata (I3).
        lengths = [len(r.track) for r in roles.values() if r.track is not None]
        T = max(lengths) if lengths else 0
        if T == 0:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="no role tracks available to evaluate predicates on",
            )
        k = max(1, spec.success_sustain_frames)
        window = slice(max(0, T - k), T)
        results = eval_conjunction(spec.success, roles, window)
        missing = [r for r in results if r.status == "evidence_missing"]
        if missing:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="; ".join(dict.fromkeys(r.reason for r in missing)),
            )
        all_hold = all(r.holds for r in results)
        margins = np.array([r.margin for r in results], dtype=np.float64)
        # Confidence: squashed worst margin (how comfortably the verdict holds).
        conf = float(1.0 / (1.0 + np.exp(-min(margins) / 3.0)))
        details = {f"pred:{r.predicate.describe()}": (1.0 if r.holds else 0.0) for r in results}
        details.update({f"margin:{r.predicate.describe()}": round(r.margin, 3) for r in results})
        return ChannelScore(
            channel=self.name,
            value=1.0 if all_hold else 0.0,
            confidence=conf,
            reason="; ".join(r.reason for r in results if r.reason),
            details=details,
        )
