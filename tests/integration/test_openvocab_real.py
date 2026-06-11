"""Slow lane: the REAL open-vocab grounder on real encoded video.

Downloads grounding-dino-tiny (~700MB) + sam-vit-base (~360MB) on first run.
RUN_SLOW=1 required; uses GPU when available, CPU otherwise.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.slow, pytest.mark.enable_socket]  # model downloads need net

iio = pytest.importorskip("imageio.v3", reason="requires the [video] extra")
pytest.importorskip("transformers", reason="requires the [ground] extra")

from woracle.grounders.openvocab import OpenVocabGrounder  # noqa: E402
from woracle.io import rollout_from_video  # noqa: E402
from woracle.testing.blobworld import blob_spec, make_episode  # noqa: E402
from woracle.testing.conformance import grounder_checks  # noqa: E402


@pytest.fixture(scope="module")
def grounder() -> OpenVocabGrounder:
    return OpenVocabGrounder(stride=6, det_threshold=0.2)


@pytest.fixture(scope="module")
def blob_video(tmp_path_factory):
    frames, truth = make_episode("success", seed=0, n_frames=48)
    path = str(tmp_path_factory.mktemp("rv") / "ep.mp4")
    iio.imwrite(path, frames, fps=10, codec="libx264")
    return path, truth


def test_real_grounder_binds_blobworld(blob_video, tmp_path) -> None:
    """End-to-end honesty check on ground truth we control: the real detector
    must find the red square and green cup in real encoded video, and the
    track must follow the truth trajectory within a loose pixel budget."""
    path, truth = blob_video
    ref = rollout_from_video(path, source="blobworld")
    out = str(tmp_path / "g")
    os.makedirs(out, exist_ok=True)
    g = OpenVocabGrounder(stride=6, det_threshold=0.2)
    grounded = g.ground(ref, blob_spec(), out)

    carried = grounded.binding("carried_object")
    receptacle = grounded.binding("receptacle")
    assert carried.bound, carried.reason
    assert receptacle.bound, receptacle.reason
    assert carried.quality > 0.5, f"detected only {carried.quality:.0%} of samples"

    with np.load(carried.tracks.resolve(out)) as z:
        track = z["track"]
    obs = np.isfinite(track[:, 0])
    assert obs.sum() > len(track) * 0.5
    # truth.carried is (T,2); codec may trim trailing frames
    n = min(len(track), len(truth.carried))
    err = np.linalg.norm(track[:n][obs[:n]] - truth.carried[:n][obs[:n]], axis=1)
    assert np.median(err) < 12.0, f"median track error {np.median(err):.1f}px"


@pytest.mark.xfail(
    strict=False,
    reason="MEASURED GDINO-tiny confidence inversion (2026-06-11 probe): absent "
    "'purple elephant' scores 0.606 > present 'red block' 0.511 — no score "
    "threshold can separate them. Mitigation is P2's appearance-consistency "
    "gate signal (crop-embedding drift along the track), not detector confidence.",
)
def test_real_grounder_reports_absent_role_unbound(blob_video, tmp_path) -> None:
    """ASPIRATION (currently xfail): a role whose object does not exist should
    come back unbound or rock-bottom quality. The detector alone cannot
    deliver this — kept as the canary for the P2 mitigation."""
    path, _ = blob_video
    spec = blob_spec()
    spec.role("carried_object").candidates = ["purple elephant"]
    ref = rollout_from_video(path, source="blobworld")
    out = str(tmp_path / "g2")
    os.makedirs(out, exist_ok=True)
    grounded = OpenVocabGrounder(stride=12, det_threshold=0.2).ground(ref, spec, out)
    b = grounded.binding("carried_object")
    assert (not b.bound) or b.quality < 0.3, (
        f"detector confidently bound a purple elephant (quality={b.quality:.2f}) — "
        "known GDINO absent-object failure; threshold/prompt mitigation regressed"
    )


def test_real_grounder_passes_conformance(tmp_path) -> None:
    for name, check in grounder_checks(lambda: OpenVocabGrounder(stride=8)):  # type: ignore[arg-type]
        check()
        _ = name
