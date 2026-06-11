"""Blobworld reference components — real (if simple) implementations.

These are not mocks: the grounder genuinely segments by color, the signals
genuinely measure, the channels genuinely score. They make the P0 end-to-end
test a *semantic* test (success ranks above failure; vanish abstains), and
they double as executable documentation of each Protocol.
"""

from __future__ import annotations

import os

import numpy as np

from woracle.contracts import (
    ArtifactRef,
    ChannelCaps,
    ChannelScore,
    GateSignalValue,
    GroundedRollout,
    RoleBinding,
    RolloutRef,
    TaskSpec,
    digest_file,
)
from woracle.io import load_frames
from woracle.registry import register

_COLOR_KEYWORDS = {
    "red": ([150, 0, 0], [255, 90, 90]),
    "green": ([0, 120, 0], [90, 230, 110]),
    "blue": ([0, 0, 150], [90, 110, 255]),
}


def _color_for_role(candidates: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    for cand in candidates:
        for kw, (lo, hi) in _COLOR_KEYWORDS.items():
            if kw in cand.lower():
                return np.array(lo), np.array(hi)
    return None


# role_data / PredicateSuccessChannel moved to woracle.channels.verdict
# (production layer); re-exported here for backward compatibility.
from woracle.channels.verdict import (  # noqa: E402, F401
    PredicateSuccessChannel,
    role_data,
)


@register("grounder", "blob.color")
class BlobColorGrounder:
    """Binds roles by color keywords found in role candidates."""

    name = "blob.color"
    version = "0.1.0"

    @property
    def params(self) -> dict:
        return {}

    def ground(self, rollout: RolloutRef, spec: TaskSpec, out_dir: str) -> GroundedRollout:
        frames = load_frames(rollout)  # (T, H, W, 3) uint8
        T = frames.shape[0]
        bindings: list[RoleBinding] = []
        for role in spec.roles:
            rng_pair = _color_for_role(role.candidates)
            if rng_pair is None:
                bindings.append(
                    RoleBinding(
                        role=role.name,
                        bound=False,
                        required=role.required,
                        reason="no color keyword in candidates",
                    )
                )
                continue
            lo, hi = rng_pair
            mask = np.all((frames >= lo) & (frames <= hi), axis=-1)  # (T, H, W) bool
            area = mask.reshape(T, -1).sum(axis=1).astype(np.float32)
            track = np.full((T, 2), np.nan, np.float32)
            for t in range(T):
                ys, xs = np.nonzero(mask[t])
                if len(xs) > 0:
                    track[t] = (xs.mean(), ys.mean())
            max_area = float(area.max())
            vis = (area / max_area).astype(np.float32) if max_area > 0 else area
            seen_early = bool((area[: max(3, T // 20)] > 0).any())
            if not seen_early:
                bindings.append(
                    RoleBinding(
                        role=role.name,
                        bound=False,
                        required=role.required,
                        reason="role never visible near rollout start",
                    )
                )
                continue
            tpath, mpath, vpath = (
                os.path.join(out_dir, f"{role.name}.track.npz"),
                os.path.join(out_dir, f"{role.name}.mask.npz"),
                os.path.join(out_dir, f"{role.name}.vis.npz"),
            )
            np.savez_compressed(tpath, track=track)
            np.savez_compressed(mpath, mask=mask.astype(np.uint8))
            np.savez_compressed(vpath, vis=vis)
            bindings.append(
                RoleBinding(
                    role=role.name,
                    bound=True,
                    required=role.required,
                    quality=float(np.mean(area > 0)),
                    tracks=ArtifactRef(
                        path=os.path.basename(tpath), sha256=digest_file(tpath), kind="track.npz"
                    ),
                    masks=ArtifactRef(
                        path=os.path.basename(mpath), sha256=digest_file(mpath), kind="mask.npz"
                    ),
                    visibility=ArtifactRef(
                        path=os.path.basename(vpath), sha256=digest_file(vpath), kind="vis.npz"
                    ),
                )
            )
        grounded = GroundedRollout(
            rollout=rollout,
            spec_name=spec.name,
            spec_hash=spec.content_hash(),
            bindings=bindings,
            grounder=f"{self.name}@{self.version}",
            bundle_dir=out_dir,
        )
        with open(os.path.join(out_dir, "grounded.json"), "w", encoding="utf-8") as f:
            f.write(grounded.to_json())
        return grounded


@register("gate_signal", "binding_health")
class BindingHealthSignal:
    """min binding quality over REQUIRED roles; evidence_missing if a required
    role is unbound. Optional roles never gate, only annotate."""

    name = "binding_health"
    version = "0.1.0"

    def measure(self, grounded: GroundedRollout) -> GateSignalValue:
        required = [b for b in grounded.bindings if b.required]
        optional_unbound = sorted(
            b.role for b in grounded.bindings if not b.required and not b.bound
        )
        missing = sorted(b.role for b in required if not b.bound)
        if missing:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason=f"unbound required roles: {', '.join(missing)}",
            )
        if not required:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="spec declares no required roles",
            )
        value = float(min(b.quality for b in required))
        details = {f"optional_unbound:{r}": 0.0 for r in optional_unbound}
        return GateSignalValue(name=self.name, value=value, details=details)


@register("gate_signal", "permanence")
class PermanenceSignal:
    """Object permanence: did any bound role's visible area collapse?

    value = min over bound roles of (late visibility / early visibility).
    A role that vanishes (WM deleted it) drives this to ~0.
    """

    name = "permanence"
    version = "0.1.0"

    def measure(self, grounded: GroundedRollout) -> GateSignalValue:
        roles = role_data(grounded)
        worst, worst_role = 1.0, ""
        any_data = False
        details: dict[str, float] = {}
        for b in grounded.bindings:
            if not b.bound:
                continue
            vis = roles[b.role].visibility
            if vis is None or len(vis) < 4:
                continue
            any_data = True
            k = max(2, len(vis) // 10)
            early = float(np.mean(vis[:k]))
            tail = float(np.min(_running_mean(vis, k)))
            ratio = 1.0 if early <= 0 else max(0.0, min(1.0, tail / early))
            details[b.role] = round(ratio, 4)
            if ratio < worst:
                worst, worst_role = ratio, b.role
        if not any_data:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="no visibility series for any bound role",
            )
        reason = (
            f"'{worst_role}' visibility collapsed to {worst:.2f}x early level"
            if worst < 0.5
            else ""
        )
        return GateSignalValue(name=self.name, value=worst, reason=reason, details=details)


def _running_mean(x: np.ndarray, k: int) -> np.ndarray:
    c = np.convolve(x, np.ones(k) / k, mode="valid")
    return c if len(c) > 0 else x


@register("gate_signal", "motion_sanity")
class MotionSanitySignal:
    """Frozen-rollout detector: fraction of frames with any scene motion."""

    name = "motion_sanity"
    version = "0.1.0"

    def measure(self, grounded: GroundedRollout) -> GateSignalValue:
        roles = role_data(grounded)
        tracks = [r.track for r in roles.values() if r.track is not None]
        if not tracks:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="no tracks available",
            )
        moving = np.zeros(0, bool)
        for tr in tracks:
            valid = np.asarray(np.isfinite(tr).all(axis=1))
            pair_ok = valid[1:] & valid[:-1]
            d = np.linalg.norm(np.diff(tr, axis=0), axis=1)
            step = pair_ok & (d > 0.2)  # NaN-adjacent pairs never count as motion
            moving = step if moving.size == 0 else (moving | step)
        frac = float(moving.mean()) if moving.size else 0.0
        return GateSignalValue(
            name=self.name,
            value=min(1.0, frac * 2.0),
            details={"frac_frames_moving": round(frac, 4)},
        )


@register("channel", "progress.goal_distance")
class GoalDistanceProgress:
    """Rank-only progress: normalized approach of carried object to receptacle.

    NOT verdict-eligible by design — progress channels rank, predicates decide
    (proposal §4: trajectory/flow are walled off from the success verdict).
    """

    name = "progress.goal_distance"
    version = "0.1.0"
    caps = ChannelCaps(reference_free=True, needs_tracks=True, verdict_eligible=False)

    def score(self, grounded: GroundedRollout, spec: TaskSpec) -> ChannelScore:
        roles = role_data(grounded)
        sub, obj = roles.get("carried_object"), roles.get("receptacle")
        if sub is None or sub.track is None or obj is None or obj.track is None:
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="carried_object/receptacle tracks unavailable",
            )
        s = sub.track
        valid = np.asarray(np.isfinite(s).all(axis=1))
        if not valid.any() or not np.isfinite(obj.track).any():
            return ChannelScore(
                channel=self.name,
                status="evidence_missing",
                reason="no observed positions for carried_object/receptacle",
            )
        o = np.nanmean(obj.track, axis=0)
        d = np.linalg.norm(s - o, axis=1)  # NaN where unobserved
        first = int(np.argmax(valid))
        last = int(len(valid) - 1 - np.argmax(valid[::-1]))
        d0 = max(float(d[first]), 1e-6)
        curve = np.clip(1.0 - d / d0, 0.0, 1.0)
        series = [round(float(v), 4) if np.isfinite(v) else None for v in curve]
        return ChannelScore(
            channel=self.name,
            value=float(curve[last]),
            confidence=float(valid.mean()),
            series={"progress": [v for v in series if v is not None]},
            details={"final_distance_px": round(float(d[last]), 2)},
        )


def register_all() -> None:
    """Entry-point style registrar (also exercises the plugin path in tests)."""
    # Importing this module performs registration via decorators.
