"""mp4 ingestion tests (skip cleanly when the [video] extra is absent)."""

from __future__ import annotations

import numpy as np
import pytest

iio = pytest.importorskip("imageio.v3", reason="requires the [video] extra")

from woracle.io import load_frames, rollout_from_video  # noqa: E402
from woracle.io.video import decode_video  # noqa: E402
from woracle.testing.blobworld import make_episode  # noqa: E402


@pytest.fixture(scope="module")
def blob_mp4(tmp_path_factory) -> str:
    frames, _ = make_episode("success", seed=0, n_frames=30)
    path = str(tmp_path_factory.mktemp("vid") / "success.mp4")
    iio.imwrite(path, frames, fps=10, codec="libx264")
    return path


def test_decode_roundtrip_and_cache(blob_mp4, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORACLE_CACHE_DIR", str(tmp_path / "cache"))
    frames, fps = decode_video(blob_mp4)
    assert frames.ndim == 4 and frames.shape[-1] == 3 and frames.dtype == np.uint8
    assert frames.shape[0] >= 28  # codecs may drop a trailing frame or two
    assert 8.0 <= fps <= 12.0
    # second call must come from cache and be byte-identical
    frames2, _ = decode_video(blob_mp4)
    assert np.array_equal(frames, frames2)
    cache_files = list((tmp_path / "cache" / "decode").glob("*.npz"))
    assert len(cache_files) == 1


def test_decode_max_side(blob_mp4, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORACLE_CACHE_DIR", str(tmp_path / "cache"))
    frames, _ = decode_video(blob_mp4, max_side=64)
    assert max(frames.shape[1:3]) == 64


def test_rollout_from_video_and_load_frames(blob_mp4, monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORACLE_CACHE_DIR", str(tmp_path / "cache"))
    ref = rollout_from_video(blob_mp4, policy="testpol", source="wm:test")
    assert ref.id == "success" and ref.frames.sha256
    frames = load_frames(ref)
    assert frames.shape[-1] == 3
    # lossy codec: content must still be recognizably blobworld (red square area)
    red = np.all((frames >= [120, 0, 0]) & (frames <= [255, 110, 110]), axis=-1)
    assert red.sum() > 0


def test_decode_missing_file_is_loud(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("WORACLE_CACHE_DIR", str(tmp_path / "cache"))
    from woracle.errors import StoreError

    with pytest.raises(StoreError):
        decode_video(str(tmp_path / "nope.mp4"))
