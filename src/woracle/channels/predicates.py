"""Generic predicate evaluation over grounded artifacts (the seed of S4).

Operates ONLY on contract-level data (tracks, masks, visibility) — never on
models. The same engine will evaluate predicates for any grounder's output;
the blob plugins are merely its first supplier.

Conventions:
* tracks: (T, 2) float32 (x, y); NaN where the role was unobserved.
* masks:  (T, H, W) uint8/bool occupancy.
* All evaluators return ``(holds, margin)`` where margin > 0 quantifies how
  comfortably the predicate holds (signed; negative = violated by that much).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from woracle.errors import SpecError

if TYPE_CHECKING:
    from woracle.contracts import Predicate


@dataclass
class RoleData:
    track: np.ndarray | None = None  # (T, 2)
    mask: np.ndarray | None = None  # (T, H, W)
    visibility: np.ndarray | None = None  # (T,)


@dataclass
class PredicateResult:
    predicate: Predicate
    holds: bool
    margin: float
    reason: str = ""


def _bbox_from_masks(mask: np.ndarray) -> tuple[float, float, float, float]:
    """Static bbox of a (T, H, W) occupancy stack (union over time)."""
    occ = mask.any(axis=0) if mask.ndim == 3 else mask.astype(bool)
    ys, xs = np.nonzero(occ)
    if len(xs) == 0:
        raise SpecError("empty mask — cannot derive region bbox")
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def _window(track: np.ndarray, window: slice) -> np.ndarray:
    w = track[window]
    if w.size == 0:
        raise SpecError("empty evaluation window")
    return w


def eval_predicate(
    pred: Predicate,
    roles: dict[str, RoleData],
    window: slice,
    fps: float = 10.0,
) -> PredicateResult:
    sub = roles.get(pred.subject)
    if sub is None:
        return PredicateResult(pred, False, -1.0, f"no data for role '{pred.subject}'")

    if pred.kind == "present":
        vis = sub.visibility
        if vis is None:
            return PredicateResult(pred, False, -1.0, "no visibility data")
        frac = float(np.mean(vis[window] > 0))
        return PredicateResult(pred, frac >= 0.5, frac - 0.5)

    if sub.track is None:
        return PredicateResult(pred, False, -1.0, f"no track for role '{pred.subject}'")
    s = _window(sub.track, window)
    if np.isnan(s).any():
        return PredicateResult(pred, False, -1.0, f"'{pred.subject}' unobserved in window")

    if pred.kind == "stationary":
        tol = pred.params.get("tol", 1.5)
        if len(s) < 2:
            return PredicateResult(pred, True, tol)
        speed = float(np.max(np.linalg.norm(np.diff(s, axis=0), axis=1)))
        return PredicateResult(pred, speed <= tol, tol - speed)

    # Binary kinds need the object role.
    if pred.object is None:
        raise SpecError(f"predicate {pred.kind} requires an object role")
    obj = roles.get(pred.object)
    if obj is None:
        return PredicateResult(pred, False, -1.0, f"no data for role '{pred.object}'")

    if pred.kind == "contained":
        if obj.mask is None:
            return PredicateResult(pred, False, -1.0, f"no mask for role '{pred.object}'")
        x0, y0, x1, y1 = _bbox_from_masks(obj.mask)
        e = pred.params.get("erode_px", 5.0)
        e_top = pred.params.get("erode_top_px", 0.0)  # open-top containers
        ix0, iy0, ix1, iy1 = x0 + e, y0 + e_top, x1 - e, y1 - e
        margins = np.minimum.reduce([s[:, 0] - ix0, ix1 - s[:, 0], s[:, 1] - iy0, iy1 - s[:, 1]])
        m = float(margins.min())
        return PredicateResult(pred, m > 0, m)

    if pred.object not in roles or roles[pred.object].track is None:
        return PredicateResult(pred, False, -1.0, f"no track for role '{pred.object}'")
    o = _window(roles[pred.object].track, window)  # type: ignore[union-attr]
    if np.isnan(o).any():
        # Static roles may have constant tracks; NaN means truly unobserved.
        return PredicateResult(pred, False, -1.0, f"'{pred.object}' unobserved in window")
    d = np.linalg.norm(s - o, axis=1)

    if pred.kind == "co_located":
        tol = pred.params.get("tol", 12.0)
        m = float(tol - d.max())
        return PredicateResult(pred, m > 0, m)
    if pred.kind == "separated":
        tol = pred.params.get("tol", 12.0)
        m = float(d.min() - tol)
        return PredicateResult(pred, m > 0, m)
    if pred.kind == "approaching":
        if len(d) < 3:
            return PredicateResult(pred, False, 0.0, "window too short")
        slope = float(np.polyfit(np.arange(len(d)), d, 1)[0])
        return PredicateResult(pred, slope < 0, -slope)

    raise SpecError(f"unhandled predicate kind '{pred.kind}'")


def eval_conjunction(
    preds: list[Predicate],
    roles: dict[str, RoleData],
    window: slice,
    fps: float = 10.0,
) -> list[PredicateResult]:
    return [eval_predicate(p, roles, window, fps) for p in preds]
