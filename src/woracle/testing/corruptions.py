"""Generative-failure corruption suite for blobworld (P2 gate benchmark).

Applies WM-style failure modes to clean blobworld episodes, with labels —
the ground-truth benchmark for gate calibration and the AUROC exit gate.
Each corruption mimics a failure family documented in the research round:

* ``vanish_late``  — object deleted in the decisive window (C2 class)
* ``freeze``       — rollout collapses to a frozen frame (motion death)
* ``noise_ramp``   — texture melt: noise amplitude grows over time
* ``teleport``     — object discontinuously relocates (identity/relatch class)
* ``bg_morph``     — background drifts away from the real anchor frame
"""

from __future__ import annotations

import numpy as np

from woracle.testing.blobworld import COLORS, make_episode

CORRUPTIONS = ("vanish_late", "freeze", "noise_ramp", "teleport", "bg_morph")


def _red_mask(frame: np.ndarray) -> np.ndarray:
    lo, hi = np.array([150, 0, 0]), np.array([255, 90, 90])
    return np.all((frame >= lo) & (frame <= hi), axis=-1)


def corrupt(frames: np.ndarray, kind: str, seed: int = 0) -> np.ndarray:
    """Return a corrupted COPY of (T, H, W, 3) uint8 frames."""
    rng = np.random.default_rng(seed)
    out = frames.copy()
    T = len(out)
    if kind == "vanish_late":
        start = int(T * 0.75)
        for t in range(start, T):
            m = _red_mask(out[t])
            out[t][m] = (200, 200, 200)
    elif kind == "freeze":
        start = int(T * 0.4)
        out[start:] = out[start]
    elif kind == "noise_ramp":
        for t in range(T):
            amp = 60.0 * (t / max(T - 1, 1)) ** 2
            noise = rng.normal(0, amp, out[t].shape)
            out[t] = np.clip(out[t].astype(np.float32) + noise, 0, 255).astype(np.uint8)
    elif kind == "teleport":
        # every ~12 frames, cut the red square out and paste it elsewhere
        for t in range(int(T * 0.3), T):
            if t % 12 in (0, 1, 2):
                m = _red_mask(out[t])
                if m.any():
                    ys, xs = np.nonzero(m)
                    out[t][m] = (200, 200, 200)
                    dy = int(rng.integers(-30, 30))
                    dx = int(rng.integers(-50, 50))
                    yy = np.clip(ys + dy, 0, out.shape[1] - 1)
                    xx = np.clip(xs + dx, 0, out.shape[2] - 1)
                    out[t][yy, xx] = COLORS["carried_object"]
    elif kind == "bg_morph":
        drift = rng.normal(0, 1.0, (out.shape[1], out.shape[2], 3))
        for t in range(T):
            k = 50.0 * (t / max(T - 1, 1))
            morphed = out[t].astype(np.float32) + k * drift
            keep = _red_mask(out[t])  # keep the task object intact
            out[t] = np.clip(morphed, 0, 255).astype(np.uint8)
            out[t][keep] = frames[t][keep]
    else:
        raise ValueError(f"unknown corruption '{kind}' (kinds: {CORRUPTIONS})")
    return out


def gate_benchmark(seeds: range = range(3), n_frames: int = 60):
    """Yield (name, frames, is_broken) — clean episodes + every corruption."""
    for seed in seeds:
        for clean_kind in ("success", "fail_miss"):
            frames, _ = make_episode(clean_kind, seed=seed, n_frames=n_frames)
            yield f"{clean_kind}_s{seed}", frames, False
        base, _ = make_episode("success", seed=seed, n_frames=n_frames)
        for kind in CORRUPTIONS:
            yield f"{kind}_s{seed}", corrupt(base, kind, seed=seed), True
