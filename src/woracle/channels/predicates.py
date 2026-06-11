"""Generic predicate evaluation over grounded artifacts (the seed of S4).

Operates ONLY on contract-level data (tracks, masks, visibility) — never on
models. The same engine will evaluate predicates for any grounder's output;
the blob plugins are merely its first supplier.

Conventions:
* tracks: (T, 2) float32 (x, y); NaN where the role was unobserved.
* masks:  (T, H, W) uint8/bool occupancy.
* Distances/speeds are in pixels and pixels-per-frame (time-normalization is a
  P3 concern, applied where specs declare physical tolerances).

Honesty semantics (the C2 rule): a predicate that CANNOT be evaluated —
missing role data, NaN track in the window, empty mask, empty window — is
``status="evidence_missing"``, never a definitive ``holds=False``. "The object
was unobservable at the decisive moment" is grounds for ABSTAIN, not FAIL.
``SpecError`` is reserved for malformed specs (wrong arity, unknown kind).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from woracle.errors import SpecError

if TYPE_CHECKING:
    from woracle.contracts import Predicate

PredStatus = Literal["ok", "evidence_missing"]


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
    status: PredStatus = "ok"
    reason: str = ""

    @property
    def evaluable(self) -> bool:
        return self.status == "ok"


def _missing(pred: Predicate, reason: str) -> PredicateResult:
    return PredicateResult(pred, False, 0.0, status="evidence_missing", reason=reason)


def _bbox_from_masks(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    """Static bbox of a (T, H, W) occupancy stack (union over time); None if empty."""
    occ = mask.any(axis=0) if mask.ndim == 3 else mask.astype(bool)
    ys, xs = np.nonzero(occ)
    if len(xs) == 0:
        return None
    return float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)


def eval_predicate(
    pred: Predicate,
    roles: dict[str, RoleData],
    window: slice,
) -> PredicateResult:
    sub = roles.get(pred.subject)
    if sub is None:
        return _missing(pred, f"no data for role '{pred.subject}'")

    if pred.kind == "present":
        vis = sub.visibility
        if vis is None:
            return _missing(pred, f"no visibility data for role '{pred.subject}'")
        w = vis[window]
        if w.size == 0:
            return _missing(pred, "empty evaluation window")
        frac = float(np.mean(w > 0))
        return PredicateResult(pred, frac >= 0.5, frac - 0.5)

    if sub.track is None:
        return _missing(pred, f"no track for role '{pred.subject}'")
    s = sub.track[window]
    if s.size == 0:
        return _missing(pred, "empty evaluation window")
    if np.isnan(s).any():
        return _missing(pred, f"'{pred.subject}' unobserved in evaluation window")

    if pred.kind == "stationary":
        tol = pred.params.get("tol", 1.5)  # px/frame
        if len(s) < 2:
            return PredicateResult(pred, True, tol)
        speed = float(np.max(np.linalg.norm(np.diff(s, axis=0), axis=1)))
        return PredicateResult(pred, speed <= tol, tol - speed)

    # Binary kinds need the object role.
    if pred.object is None:
        raise SpecError(f"predicate {pred.kind} requires an object role")
    obj = roles.get(pred.object)
    if obj is None:
        return _missing(pred, f"no data for role '{pred.object}'")

    if pred.kind == "contained":
        if obj.mask is None:
            return _missing(pred, f"no mask for role '{pred.object}'")
        bbox = _bbox_from_masks(obj.mask)
        if bbox is None:
            return _missing(pred, f"role '{pred.object}' mask is empty — region unknowable")
        x0, y0, x1, y1 = bbox
        e = pred.params.get("erode_px", 5.0)
        e_top = pred.params.get("erode_top_px", 0.0)  # open-top containers
        ix0, iy0, ix1, iy1 = x0 + e, y0 + e_top, x1 - e, y1 - e
        margins = np.minimum.reduce([s[:, 0] - ix0, ix1 - s[:, 0], s[:, 1] - iy0, iy1 - s[:, 1]])
        m = float(margins.min())
        return PredicateResult(pred, m > 0, m)

    if obj.track is None:
        return _missing(pred, f"no track for role '{pred.object}'")
    o = obj.track[window]
    if o.size == 0:
        return _missing(pred, "empty evaluation window")
    if np.isnan(o).any():
        return _missing(pred, f"'{pred.object}' unobserved in evaluation window")
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
            return _missing(pred, "window too short to assess approach")
        slope = float(np.polyfit(np.arange(len(d)), d, 1)[0])
        return PredicateResult(pred, slope < 0, -slope)

    raise SpecError(f"unhandled predicate kind '{pred.kind}'")


def eval_conjunction(
    preds: list[Predicate],
    roles: dict[str, RoleData],
    window: slice,
) -> list[PredicateResult]:
    return [eval_predicate(p, roles, window) for p in preds]
