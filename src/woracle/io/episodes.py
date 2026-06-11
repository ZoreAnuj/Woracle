"""Episode I/O — P0 supports the dependency-free ``frames.npz`` payload.

Episode directory layout::

    <dir>/
        rollout.json      # RolloutRef (relative ArtifactRef paths)
        frames.npz        # key "frames": (T, H, W, 3) uint8
        actions.npz       # optional, key "actions": (T, A) float32

mp4 ingestion arrives in P1 behind an extra — the kernel never grows a video
decoder dependency.
"""

from __future__ import annotations

import json
import os

import numpy as np

from woracle.contracts import ArtifactRef, RolloutRef, digest_file, migrate
from woracle.errors import StoreError


def save_episode(
    dir_path: str,
    rollout_id: str,
    frames: np.ndarray,
    *,
    fps: float = 10.0,
    policy: str = "",
    source: str = "",
    actions: np.ndarray | None = None,
    meta: dict[str, str] | None = None,
) -> RolloutRef:
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.dtype != np.uint8:
        raise StoreError(f"frames must be (T, H, W, 3) uint8, got {frames.shape} {frames.dtype}")
    os.makedirs(dir_path, exist_ok=True)
    fpath = os.path.join(dir_path, "frames.npz")
    np.savez_compressed(fpath, frames=frames)
    refs: dict[str, ArtifactRef] = {
        "frames": ArtifactRef(path="frames.npz", sha256=digest_file(fpath), kind="frames.npz")
    }
    if actions is not None:
        apath = os.path.join(dir_path, "actions.npz")
        np.savez_compressed(apath, actions=actions.astype(np.float32))
        refs["actions"] = ArtifactRef(
            path="actions.npz", sha256=digest_file(apath), kind="actions.npz"
        )
    ref = RolloutRef(
        id=rollout_id,
        frames=refs["frames"],
        fps=fps,
        n_frames=int(frames.shape[0]),
        policy=policy,
        source=source,
        actions=refs.get("actions"),
        meta=meta or {},
    )
    with open(os.path.join(dir_path, "rollout.json"), "w", encoding="utf-8") as f:
        f.write(ref.to_json())
    return ref


def load_rollout(dir_path: str) -> RolloutRef:
    path = os.path.join(dir_path, "rollout.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data = migrate("RolloutRef", data)
    ref = RolloutRef.model_validate(data)
    ref.meta = {**ref.meta, "_dir": os.path.abspath(dir_path)}
    return ref


def load_frames(ref: RolloutRef) -> np.ndarray:
    base = ref.meta.get("_dir", "")
    path = ref.frames.resolve(base) if base else ref.frames.path
    if not os.path.isfile(path):
        raise StoreError(f"frames payload not found: {path}")
    with np.load(path) as z:
        return z["frames"]


def list_rollouts(root: str) -> list[RolloutRef]:
    """Find every episode directory (contains rollout.json) under ``root``."""
    out: list[RolloutRef] = []
    for dirpath, _dirnames, filenames in sorted(os.walk(root)):
        if "rollout.json" in filenames:
            out.append(load_rollout(dirpath))
    return out
