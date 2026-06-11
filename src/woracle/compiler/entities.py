"""Model-free entity induction: who's in the scene and how does it move?

Shared by the spec compiler (role discovery from demos) and the relational
grounder (role binding at grade time). Color-quantizes frames (numpy k-means,
seeded), tracks each color cluster, and classifies motion — no models, no
appearance priors, fully deterministic.

This is the v1 perception floor: adequate for blobworld-class scenes and as
the relational fallback/cross-check for real scenes; open-vocab + embedding
induction layer on top of the SAME Entity contract later.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Small named-color palette for candidate phrases (nearest-neighbor naming).
_PALETTE = {
    "red": (210, 50, 50),
    "green": (50, 180, 70),
    "blue": (50, 80, 220),
    "yellow": (230, 200, 50),
    "purple": (160, 50, 200),
    "orange": (230, 140, 40),
    "white": (240, 240, 240),
    "gray": (128, 128, 128),
    "black": (30, 30, 30),
}


def color_name(rgb: np.ndarray) -> str:
    keys = list(_PALETTE)
    dists = [np.linalg.norm(np.asarray(rgb, float) - np.asarray(_PALETTE[k], float)) for k in keys]
    return keys[int(np.argmin(dists))]


@dataclass
class Entity:
    eid: int
    color: np.ndarray  # cluster center RGB
    track: np.ndarray  # (T, 2) centroid, NaN where absent
    area: np.ndarray  # (T,) pixel count
    mask_fn: object = None  # callable(t) -> (H, W) bool, lazily evaluated
    stats: dict[str, float] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return color_name(self.color)


def _kmeans(px: np.ndarray, k: int, iters: int = 12, seed: int = 0) -> np.ndarray:
    # Deterministic farthest-point init (k-means++ spirit): the first center is
    # the most saturated pixel; each next center maximizes min-distance to the
    # chosen set — tiny color modes get their own center instead of luck.
    centers = [px[int(np.argmax(np.linalg.norm(px - np.median(px, axis=0), axis=1)))]]
    for _ in range(k - 1):
        d = np.min(np.stack([np.linalg.norm(px - c, axis=1) for c in centers]), axis=0)
        centers.append(px[int(np.argmax(d))])
    centers = np.stack(centers).astype(np.float64)
    _ = seed  # init is deterministic; seed kept for signature stability
    for _ in range(iters):
        d = np.linalg.norm(px[:, None, :] - centers[None, :, :], axis=2)
        assign = d.argmin(axis=1)
        for j in range(k):
            sel = px[assign == j]
            if len(sel):
                centers[j] = sel.mean(axis=0)
    return centers


def induce_entities(
    frames: np.ndarray,
    *,
    k: int = 5,
    sample_stride: int = 4,
    min_area: int = 12,
    seed: int = 0,
) -> list[Entity]:
    """Cluster colors -> per-cluster tracks. Background clusters (huge, static
    coverage) are dropped; the rest are candidate task entities."""
    T, H, W = frames.shape[:3]
    f_idx = np.arange(0, T, max(1, T // 12))
    px = frames[f_idx][:, ::sample_stride, ::sample_stride, :].reshape(-1, 3).astype(np.float64)

    # SALIENCY-WEIGHTED sampling: task objects can be <1% of pixels; uniform
    # k-means hands every center to the background. Sample ∝ distance from
    # the background median color so small colored entities earn centers.
    bg_color = np.median(px, axis=0)
    sal = np.linalg.norm(px - bg_color, axis=1)
    w = sal + 1e-3
    rng = np.random.default_rng(seed)
    take = min(len(px), 6000)
    # replace=True: a weighted BOOTSTRAP. Without replacement the few hundred
    # colored pixels are exhausted immediately and gray noise floods the
    # sample again (measured failure mode, not hypothetical).
    sel = rng.choice(len(px), size=take, replace=True, p=w / w.sum())
    centers = _kmeans(px[sel], k, seed=seed)

    # Assign every pixel of every frame to nearest center (vectorized, frame-wise).
    flat_centers = centers[None, None, :, :]
    tracks = np.full((k, T, 2), np.nan, np.float32)
    areas = np.zeros((k, T), np.float32)
    assigns = np.empty((T, H, W), np.uint8)
    for t in range(T):
        d = np.linalg.norm(frames[t][:, :, None, :].astype(np.float64) - flat_centers, axis=3)
        a = d.argmin(axis=2).astype(np.uint8)
        assigns[t] = a
        for j in range(k):
            ys, xs = np.nonzero(a == j)
            areas[j, t] = len(xs)
            if len(xs) >= min_area:
                tracks[j, t] = (xs.mean(), ys.mean())

    diag = float(np.hypot(H, W))
    entities: list[Entity] = []
    for j in range(k):
        mean_area_frac = float(areas[j].mean()) / (H * W)
        obs = np.isfinite(tracks[j, :, 0])
        if mean_area_frac > 0.25 or not obs.any():
            continue  # background / never-present
        if float(np.linalg.norm(centers[j] - bg_color)) < 30.0:
            continue  # background-colored cluster (texture noise band)
        pts = tracks[j][obs]
        rng_px = float(np.linalg.norm(pts.max(0) - pts.min(0))) if len(pts) > 1 else 0.0
        ent = Entity(
            eid=j,
            color=centers[j],
            track=tracks[j],
            area=areas[j],
            mask_fn=(lambda t, j=j: assigns[t] == j),
            stats={
                "range_px": rng_px,
                "range_frac": rng_px / diag,
                "persistence": float(obs.mean()),
                "mean_area": float(areas[j][areas[j] > 0].mean()) if (areas[j] > 0).any() else 0.0,
            },
        )
        entities.append(ent)
    return entities


def classify_relational(
    entities: list[Entity],
    *,
    moving_frac: float = 0.08,
    settle_window_frac: float = 0.2,
) -> dict[str, Entity | None]:
    """Map induced entities to relational roles by HOW THEY MOVE AND END —
    never by appearance:

    * carried_object — a mover that ends stationary nearest a static entity
    * effector       — the other mover (ties broken by velocity correlation
                       with the carried object during transport)
    * receptacle     — the static entity nearest the carried object's end

    Returns possibly-None slots. (Never-separating mover pairs are detected
    and quality-penalized by the relational GROUNDER, not here.)
    """
    movers = [e for e in entities if e.stats["range_frac"] >= moving_frac]
    statics = [e for e in entities if e.stats["range_frac"] < moving_frac]
    out: dict[str, Entity | None] = {
        "carried_object": None,
        "effector": None,
        "receptacle": None,
    }
    if not movers or not statics:
        return out

    def final_pos(e: Entity) -> np.ndarray | None:
        obs = np.isfinite(e.track[:, 0])
        if not obs.any():
            return None
        T = len(e.track)
        w0 = int(T * (1 - settle_window_frac))
        idx = np.flatnonzero(obs & (np.arange(T) >= w0))
        if len(idx) == 0:
            idx = np.flatnonzero(obs)
        return e.track[idx].mean(axis=0)

    # carried = mover whose end position is closest to some static entity
    best = None
    for m in movers:
        fp = final_pos(m)
        if fp is None:
            continue
        for st in statics:
            sp = final_pos(st)
            if sp is None:
                continue
            d = float(np.linalg.norm(fp - sp))
            if best is None or d < best[0]:
                best = (d, m, st)
    if best is None:
        return out
    _, carried, receptacle = best
    out["carried_object"] = carried
    out["receptacle"] = receptacle

    others = [m for m in movers if m.eid != carried.eid]
    if others:
        # effector = remaining mover; if several, the one most velocity-
        # correlated with the carried object during transport
        if len(others) == 1:
            out["effector"] = others[0]
        else:

            def vel_corr(a: Entity, b: Entity) -> float:
                va = np.diff(np.nan_to_num(a.track, nan=0.0), axis=0).ravel()
                vb = np.diff(np.nan_to_num(b.track, nan=0.0), axis=0).ravel()
                denom = np.linalg.norm(va) * np.linalg.norm(vb)
                return float(va @ vb / denom) if denom > 1e-9 else 0.0

            out["effector"] = max(others, key=lambda o: vel_corr(o, carried))
    return out
