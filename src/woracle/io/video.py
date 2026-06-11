"""mp4 ingestion behind the [video] extra — decode once, cache as npz.

The kernel never grows a video dependency: imageio/imageio-ffmpeg import
lazily at call time and fail with the exact extra name. Decoded frames are
cached content-addressed (file digest + decode params), so every downstream
consumer shares one decode (the VBench re-decode-per-metric anti-pattern is
the cautionary tale).
"""

from __future__ import annotations

import os

import numpy as np

from woracle.contracts import ArtifactRef, RolloutRef, digest_file, digest_json
from woracle.errors import MissingDependencyError, StoreError

_CACHE_ENV = "WORACLE_CACHE_DIR"


def _cache_dir() -> str:
    root = os.environ.get(_CACHE_ENV) or os.path.join(os.path.expanduser("~"), ".cache", "woracle")
    d = os.path.join(root, "decode")
    os.makedirs(d, exist_ok=True)
    return d


def _imageio():
    try:
        import imageio.v3 as iio
    except ImportError as e:
        raise MissingDependencyError("mp4 decoding", "video") from e
    return iio


def decode_video(path: str, *, max_side: int | None = None) -> tuple[np.ndarray, float]:
    """Decode an mp4 to (T, H, W, 3) uint8 + fps, with a content-addressed cache."""
    if not os.path.isfile(path):
        raise StoreError(f"video not found: {path}")
    key = digest_json({"file": digest_file(path), "max_side": max_side, "codec": "v1"})
    cpath = os.path.join(_cache_dir(), f"{key}.npz")
    if os.path.isfile(cpath):
        with np.load(cpath) as z:
            return z["frames"], float(z["fps"])

    iio = _imageio()
    frames = iio.imread(path, plugin="pyav") if _has_pyav() else iio.imread(path)
    frames = np.asarray(frames)
    if frames.ndim == 3:  # single frame video
        frames = frames[None]
    if frames.shape[-1] == 4:
        frames = frames[..., :3]
    frames = frames.astype(np.uint8, copy=False)
    fps = _probe_fps(path)
    if max_side is not None and max(frames.shape[1:3]) > max_side:
        frames = _resize_batch(frames, max_side)
    tmp = cpath + ".tmp.npz"  # numpy appends .npz to names lacking it — be explicit
    np.savez_compressed(tmp, frames=frames, fps=np.float64(fps))
    os.replace(tmp, cpath)
    return frames, fps


def _has_pyav() -> bool:
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False


def _probe_fps(path: str) -> float:
    iio = _imageio()
    try:
        meta = iio.immeta(path)
        fps = float(meta.get("fps", 10.0))
        return fps if np.isfinite(fps) and fps > 0 else 10.0
    except Exception:
        return 10.0


def _resize_batch(frames: np.ndarray, max_side: int) -> np.ndarray:
    """Nearest-neighbor batch resize (numpy-only; adequate for detection input
    which the grounder's models re-resize anyway)."""
    t, h, w, _ = frames.shape
    scale = max_side / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    yi = np.clip((np.arange(nh) / scale).astype(int), 0, h - 1)
    xi = np.clip((np.arange(nw) / scale).astype(int), 0, w - 1)
    return frames[:, yi[:, None], xi[None, :], :]


def rollout_from_video(
    path: str,
    *,
    rollout_id: str | None = None,
    policy: str = "",
    source: str = "",
    fps: float | None = None,
    meta: dict[str, str] | None = None,
) -> RolloutRef:
    """Build a RolloutRef directly from an mp4 (no episode dir required)."""
    path = os.path.abspath(path)
    if fps is None:
        try:
            fps = _probe_fps(path)
        except MissingDependencyError:
            fps = 10.0
    ref = RolloutRef(
        id=rollout_id or os.path.splitext(os.path.basename(path))[0],
        frames=ArtifactRef(path=os.path.basename(path), sha256=digest_file(path), kind="mp4"),
        fps=fps,
        policy=policy,
        source=source,
        meta={**(meta or {}), "_dir": os.path.dirname(path)},
    )
    return ref
