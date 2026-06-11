"""P2 structural gate signals — model-free, artifact-driven, honest.

Each signal measures one failure family the binding study / research round
identified. None of them import model libraries (kernel rule): they consume
frames (decode cache) and grounded artifacts (tracks/masks/visibility).

Direction convention: higher = healthier (GateSignalValue contract).
Scale parameters below are MEASUREMENT scales (how a raw quantity maps onto
[0,1]); verdict thresholds live in GatePolicy, never here (ARCH decision 6).
"""

from __future__ import annotations

import numpy as np

from woracle.contracts import GateSignalValue, GroundedRollout
from woracle.registry import register


def _frames(grounded: GroundedRollout) -> np.ndarray | None:
    from woracle.io import load_frames

    try:
        return load_frames(grounded.rollout)
    except Exception:
        return None


def _gray(frames: np.ndarray) -> np.ndarray:
    return frames.mean(axis=-1).astype(np.float32)


def _role_arrays(grounded: GroundedRollout):
    from woracle.testing.plugins import role_data

    return role_data(grounded)


@register("gate_signal", "background_drift")
class BackgroundDriftSignal:
    """WM drift measured where NOTHING should change: the background.

    Task motion confounds whole-frame drift metrics, so we mask out every
    bound role's region (dilated) and compare the remaining background of each
    sampled frame against the rollout's REAL anchor (frame 0). Generated-video
    drift (texture melt, color wander, structural morph) shows up here;
    legitimate task motion does not.

    value = exp(-mean_late_background_deviation / scale)  in (0, 1].
    """

    name = "background_drift"
    version = "0.1.0"

    def __init__(self, scale: float = 12.0, dilate_px: int = 6, samples: int = 24) -> None:
        self.scale = float(scale)
        self.dilate = int(dilate_px)
        self.samples = int(samples)

    def measure(self, grounded: GroundedRollout) -> GateSignalValue:
        frames = _frames(grounded)
        if frames is None or len(frames) < 4:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="frames unavailable for background analysis",
            )
        T, H, W = frames.shape[:3]
        roles = _role_arrays(grounded)
        # Foreground = union of all role masks across time (+ dilation margin).
        fg = np.zeros((H, W), bool)
        for rd in roles.values():
            if rd.mask is not None and rd.mask.size:
                occ = rd.mask.any(axis=0) if rd.mask.ndim == 3 else rd.mask.astype(bool)
                if occ.shape == (H, W):
                    fg |= occ.astype(bool)
        if self.dilate > 0 and fg.any():
            k = self.dilate
            pad = np.zeros_like(fg)
            ys, xs = np.nonzero(fg)
            for dy in (-k, 0, k):
                for dx in (-k, 0, k):
                    yy = np.clip(ys + dy, 0, H - 1)
                    xx = np.clip(xs + dx, 0, W - 1)
                    pad[yy, xx] = True
            fg = fg | pad
        bg = ~fg
        if bg.mean() < 0.2:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason=f"only {bg.mean():.0%} of the frame is background — cannot isolate drift",
            )
        gray = _gray(frames)
        anchor = gray[0]
        idxs = np.unique(np.linspace(1, T - 1, min(self.samples, T - 1)).astype(int))
        devs = np.array([np.abs(gray[t][bg] - anchor[bg]).mean() for t in idxs])
        late = devs[len(devs) // 2 :]  # late half: where AR drift accumulates
        mean_late = float(late.mean())
        value = float(np.exp(-mean_late / self.scale))
        return GateSignalValue(
            name=self.name,
            value=value,
            reason=""
            if value > 0.5
            else f"background deviates {mean_late:.1f} gray-levels from real anchor",
            details={
                "mean_late_dev": round(mean_late, 3),
                "max_dev": round(float(devs.max()), 3),
                "bg_fraction": round(float(bg.mean()), 3),
            },
        )


@register("gate_signal", "appearance_consistency")
class AppearanceConsistencySignal:
    """Does each tracked role still LOOK like what was tracked at the start?

    Catches relatch (track jumps to a different-looking object) and morph
    (WM mutates the object) — the failure classes detector confidence cannot
    catch (measured inversion). v1 is model-free: normalized patch correlation
    + color-histogram similarity between each sample crop and the frame-0
    crop. An embedding upgrade (DINOv2) can land later as a separate signal.

    value = min over tracked roles of median per-sample similarity.
    """

    name = "appearance_consistency"
    version = "0.1.0"

    def __init__(self, crop: int = 24, samples: int = 16) -> None:
        self.crop = int(crop)
        self.samples = int(samples)

    @staticmethod
    def _patch(gray: np.ndarray, frames_hsl: np.ndarray, xy: np.ndarray, c: int):
        H, W = gray.shape
        x, y = int(round(xy[0])), int(round(xy[1]))
        x0, x1 = max(0, x - c), min(W, x + c)
        y0, y1 = max(0, y - c), min(H, y + c)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return None
        return gray[y0:y1, x0:x1], frames_hsl[y0:y1, x0:x1]

    @staticmethod
    def _similarity(p0, pt) -> float:
        g0, c0 = p0
        gt, ct = pt
        h = min(g0.shape[0], gt.shape[0])
        w = min(g0.shape[1], gt.shape[1])
        g0, gt = g0[:h, :w], gt[:h, :w]
        a, b = g0 - g0.mean(), gt - gt.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        ncc = float((a * b).sum() / denom) if denom > 1e-6 else 0.0
        h0 = np.histogram(c0.reshape(-1, 3) @ [1, 2, 4], bins=24, range=(0, 7 * 255))[0]
        h1 = np.histogram(ct[:h, :w].reshape(-1, 3) @ [1, 2, 4], bins=24, range=(0, 7 * 255))[0]
        h0 = h0 / max(h0.sum(), 1)
        h1 = h1 / max(h1.sum(), 1)
        hist_sim = float(np.minimum(h0, h1).sum())
        return 0.5 * (max(ncc, 0.0) + hist_sim)

    def measure(self, grounded: GroundedRollout) -> GateSignalValue:
        frames = _frames(grounded)
        if frames is None:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="frames unavailable",
            )
        gray = _gray(frames)
        roles = _role_arrays(grounded)
        per_role: dict[str, float] = {}
        for b in grounded.bindings:
            rd = roles.get(b.role)
            if not b.bound or rd is None or rd.track is None:
                continue
            track = rd.track
            obs = np.isfinite(track[:, 0])
            if obs.sum() < 3:
                continue
            t0 = int(np.argmax(obs))
            p0 = self._patch(gray[t0], frames[t0], track[t0], self.crop)
            if p0 is None:
                continue
            ts = np.linspace(t0, len(track) - 1, self.samples).astype(int)
            sims = []
            for t in ts:
                if not obs[t]:
                    continue
                pt = self._patch(gray[t], frames[t], track[t], self.crop)
                if pt is not None:
                    sims.append(self._similarity(p0, pt))
            if sims:
                per_role[b.role] = float(np.median(sims))
        if not per_role:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="no tracked role had enough observed samples to compare appearance",
            )
        worst_role = min(per_role, key=per_role.get)  # type: ignore[arg-type]
        value = per_role[worst_role]
        return GateSignalValue(
            name=self.name,
            value=value,
            reason=""
            if value > 0.5
            else f"'{worst_role}' no longer resembles its initial appearance",
            details={k: round(v, 4) for k, v in per_role.items()},
        )


@register("gate_signal", "action_video_consistency")
class ActionVideoConsistencySignal:
    """Did the video MOVE the way the actions COMMANDED?

    Catches the WM "self-correcting" a bad policy into a plausible video
    (dWorldEval finding): commanded effector displacement directions are
    compared with the observed effector-track displacements, scale-free
    (median cosine similarity over chunks), mapped to [0,1].

    Requires an action stream and a bound effector-like role; returns
    evidence_missing otherwise (recorded, never guessed).
    """

    name = "action_video_consistency"
    version = "0.1.0"

    def __init__(
        self,
        xy_dims: tuple[int, int] = (0, 1),
        chunk: int = 8,
        min_cmd_norm: float = 1e-3,
        effector_roles: tuple[str, ...] = ("gripper",),
    ) -> None:
        self.xy_dims = tuple(xy_dims)
        self.chunk = int(chunk)
        self.min_cmd_norm = float(min_cmd_norm)
        self.effector_roles = tuple(effector_roles)

    def measure(self, grounded: GroundedRollout) -> GateSignalValue:
        ref = grounded.rollout
        if ref.actions is None:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="rollout carries no action stream",
            )
        import os

        base = ref.meta.get("_dir", "")
        apath = ref.actions.resolve(base) if base else ref.actions.path
        if not os.path.isfile(apath):
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason=f"actions payload missing: {apath}",
            )
        with np.load(apath) as z:
            actions = z["actions"]
        roles = _role_arrays(grounded)
        track = None
        for rname in self.effector_roles:
            rd = roles.get(rname)
            if rd is not None and rd.track is not None and np.isfinite(rd.track[:, 0]).sum() > 4:
                track = rd.track
                break
        if track is None:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason=f"no usable effector track among roles {self.effector_roles}",
            )
        n = min(len(actions), len(track))
        cmd = actions[:n, list(self.xy_dims)].astype(np.float64)
        obs = track[:n].astype(np.float64)
        coss = []
        for c0 in range(0, n - self.chunk, self.chunk):
            c1 = c0 + self.chunk
            cvec = cmd[c0:c1].sum(axis=0)
            if not np.isfinite(obs[c0]).all() or not np.isfinite(obs[c1 - 1]).all():
                continue
            ovec = obs[c1 - 1] - obs[c0]
            if np.linalg.norm(cvec) < self.min_cmd_norm or np.linalg.norm(ovec) < 1e-6:
                continue
            coss.append(float(np.dot(cvec, ovec) / (np.linalg.norm(cvec) * np.linalg.norm(ovec))))
        if len(coss) < 3:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="too few commanded-motion chunks with observed displacement",
            )
        med = float(np.median(coss))
        value = float((med + 1.0) / 2.0)
        return GateSignalValue(
            name=self.name,
            value=value,
            reason="" if value > 0.5 else "video motion contradicts commanded actions",
            details={"median_cos": round(med, 4), "n_chunks": float(len(coss))},
        )


@register("gate_signal", "track_continuity")
class TrackContinuitySignal:
    """Physical objects move continuously; tracks that jump are broken evidence.

    Catches teleports/relatches: value = exp(-p99_step / scale), where steps
    are per-frame displacements of each bound MOVING role's observed track
    (static roles excluded — their jumps are already appearance/relatch
    territory). Worst role wins (min).
    """

    name = "track_continuity"
    version = "0.1.0"

    def __init__(self, scale_frac: float = 0.06) -> None:
        self.scale_frac = float(scale_frac)  # of image diagonal, per frame

    def measure(self, grounded: GroundedRollout) -> GateSignalValue:
        frames_shape = None
        roles = _role_arrays(grounded)
        per_role: dict[str, float] = {}
        details: dict[str, float] = {}
        for b in grounded.bindings:
            rd = roles.get(b.role)
            if not b.bound or rd is None or rd.track is None:
                continue
            tr = rd.track
            if frames_shape is None and rd.mask is not None and rd.mask.ndim == 3:
                frames_shape = rd.mask.shape[1:]
            obs = np.isfinite(tr[:, 0])
            if obs.sum() < 4:
                continue
            pts = tr[obs]
            steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            if len(steps) == 0:
                continue
            diag = float(np.hypot(*frames_shape)) if frames_shape else 200.0
            p99 = float(np.percentile(steps, 99))
            per_role[b.role] = float(np.exp(-p99 / (self.scale_frac * diag)))
            details[f"p99_step:{b.role}"] = round(p99, 3)
        if not per_role:
            return GateSignalValue(
                name=self.name,
                status="evidence_missing",
                value=None,
                reason="no observed tracks to assess continuity",
            )
        worst = min(per_role, key=per_role.get)  # type: ignore[arg-type]
        value = per_role[worst]
        return GateSignalValue(
            name=self.name,
            value=value,
            reason=""
            if value > 0.5
            else f"'{worst}' track jumps discontinuously (teleport/relatch)",
            details={**details, **{f"score:{k}": round(v, 4) for k, v in per_role.items()}},
        )
