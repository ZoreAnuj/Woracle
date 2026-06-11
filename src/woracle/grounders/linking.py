"""Pure track-linking math for detection-based grounding (no models here).

Tested directly on synthetic detections — the model-facing wrapper in
``openvocab.py`` delegates every decision to these functions so the logic is
exercised in the fast CPU lane while the model integration runs in the slow
lane.

Conventions: boxes are (x0, y0, x1, y1) float arrays; scores in [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, (a[2] - a[0])) * max(0.0, (a[3] - a[1]))
    area_b = max(0.0, (b[2] - b[0])) * max(0.0, (b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def center(box: np.ndarray) -> np.ndarray:
    return np.array([(box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0])


LOST_AFTER = 6  # consecutive misses after which a track is declared lost


@dataclass
class LinkState:
    """Per-role tracker state across sampled frames."""

    box: np.ndarray | None = None  # last confirmed box
    misses: int = 0  # consecutive frames without a confirmation


def select_detection(
    boxes: np.ndarray,  # (N, 4)
    scores: np.ndarray,  # (N,)
    state: LinkState,
    *,
    det_threshold: float,
    iou_weight: float = 0.3,
    max_jump_frac: float = 0.35,  # of image diagonal — teleport rejection
    image_diag: float = 1.0,
) -> tuple[int, float] | None:
    """Pick the detection continuing this role's track, or None.

    Score = det_score + iou_weight * IoU(prev_box). A candidate that teleports
    farther than ``max_jump_frac`` of the image diagonal from the previous box
    is rejected even if its raw score is high. The leash loosens with
    consecutive misses, and after ``LOST_AFTER`` misses the track is declared
    LOST: acquisition becomes unanchored (a permanently-unreachable object
    would otherwise never be re-acquired; the visibility gap has already
    recorded the loss for the permanence gate).
    """
    if len(boxes) == 0:
        return None
    scores = np.asarray(scores, dtype=float)
    valid = scores >= det_threshold
    if not valid.any():
        return None
    combined = scores.copy()
    prev_box = state.box
    anchored = prev_box is not None and state.misses < LOST_AFTER
    if anchored and prev_box is not None:
        leash = max_jump_frac * image_diag * (1.0 + 0.5 * state.misses)
        prev_c = center(prev_box)
        for i in range(len(boxes)):
            if not valid[i]:
                continue
            combined[i] += iou_weight * iou(boxes[i], prev_box)
            if np.linalg.norm(center(boxes[i]) - prev_c) > leash:
                valid[i] = False
    if not valid.any():
        return None
    combined[~valid] = -np.inf
    best = int(np.argmax(combined))
    return best, float(scores[best])


def interpolate_track(
    sample_idxs: np.ndarray,  # (S,) frame indices that were actually sampled
    centers: np.ndarray,  # (S, 2) detected centers, NaN where unobserved
    n_frames: int,
) -> np.ndarray:
    """Dense (T, 2) track: linear interpolation BETWEEN observed samples.

    Honesty boundary: frames before the first or after the last observation
    stay NaN (extrapolation would manufacture evidence — the C2 lesson).
    Interior gaps are interpolated and the matching visibility handling marks
    them as inferred, not observed.
    """
    track = np.full((n_frames, 2), np.nan, np.float32)
    obs = ~np.isnan(centers[:, 0])
    if not obs.any():
        return track
    oi = sample_idxs[obs]
    oc = centers[obs]
    lo, hi = int(oi[0]), int(oi[-1])
    xs = np.arange(lo, hi + 1)
    track[lo : hi + 1, 0] = np.interp(xs, oi, oc[:, 0])
    track[lo : hi + 1, 1] = np.interp(xs, oi, oc[:, 1])
    return track


def dense_visibility(
    sample_idxs: np.ndarray,
    scores: np.ndarray,  # (S,) det scores, 0 where unobserved
    n_frames: int,
) -> np.ndarray:
    """Dense (T,) visibility: hold each sample's score until the next sample.

    Unobserved samples contribute 0 (the permanence signal sees real dips —
    holding the previous score across a miss would hide exactly the collapse
    the gate must catch).
    """
    vis = np.zeros(n_frames, np.float32)
    s = np.asarray(scores, np.float32)
    for j, idx in enumerate(sample_idxs):
        nxt = sample_idxs[j + 1] if j + 1 < len(sample_idxs) else n_frames
        vis[int(idx) : int(nxt)] = s[j]
    return vis


def motion_consistency(
    centers: np.ndarray,  # (S, 2) observed centers, NaN where unobserved
    motion: str,  # spec Role.motion signature
    image_diag: float,
    *,
    moving_min_frac: float = 0.03,
    static_max_frac: float = 0.25,
) -> tuple[bool, float]:
    """Does the observed track's spatial RANGE match the role's motion signature?

    Returns ``(consistent, range_px)``. The binding-study finding behind this:
    a detector can latch onto static background clutter with high confidence
    (100% "bind rate", ~constant score) — but a role declared
    ``co_moves_with_effector`` whose track never moves is a FALSE LATCH, and a
    ``static`` role whose track wanders is a relatch/morph. Geometry catches
    what confidence cannot (confidence inversion is measured, not theoretical).
    """
    obs = centers[np.isfinite(centers[:, 0])]
    if len(obs) < 2:
        return True, 0.0  # too little evidence to judge — not an inconsistency
    rng = float(np.linalg.norm(obs.max(axis=0) - obs.min(axis=0)))
    if motion in ("co_moves_with_effector", "actuated"):
        return rng >= moving_min_frac * image_diag, rng
    if motion == "static":
        return rng <= static_max_frac * image_diag, rng
    return True, rng  # "free" roles carry no expectation
