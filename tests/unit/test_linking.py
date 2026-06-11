"""Pure-function tests for detection linking (the logic behind openvocab)."""

from __future__ import annotations

import numpy as np

from woracle.grounders.linking import (
    LinkState,
    dense_visibility,
    interpolate_track,
    iou,
    select_detection,
)


def test_iou_basic() -> None:
    a = np.array([0, 0, 10, 10], float)
    assert iou(a, a) == 1.0
    assert iou(a, np.array([20, 20, 30, 30], float)) == 0.0
    assert abs(iou(a, np.array([5, 0, 15, 10], float)) - (50 / 150)) < 1e-9


def test_select_prefers_track_continuity_over_raw_score() -> None:
    state = LinkState(box=np.array([10, 10, 20, 20], float))
    boxes = np.array([[11, 11, 21, 21], [80, 80, 90, 90]], float)
    scores = np.array([0.5, 0.6])  # far box scores higher...
    pick = select_detection(
        boxes, scores, state, det_threshold=0.3, iou_weight=0.5, image_diag=141.0
    )
    assert pick is not None and pick[0] == 0  # ...but continuity wins


def test_select_rejects_teleports_but_allows_reacquire() -> None:
    state = LinkState(box=np.array([0, 0, 10, 10], float))
    far = np.array([[120, 120, 130, 130]], float)
    s = np.array([0.9])
    assert (
        select_detection(far, s, state, det_threshold=0.3, max_jump_frac=0.3, image_diag=141.0)
        is None
    )
    state.misses = 6  # LOST_AFTER misses: track declared lost -> unanchored re-acquisition
    assert (
        select_detection(far, s, state, det_threshold=0.3, max_jump_frac=0.3, image_diag=141.0)
        is not None
    )


def test_select_threshold_and_empty() -> None:
    state = LinkState()
    assert select_detection(np.zeros((0, 4)), np.zeros(0), state, det_threshold=0.3) is None
    assert (
        select_detection(np.array([[0, 0, 1, 1]], float), np.array([0.1]), state, det_threshold=0.3)
        is None
    )


def test_interpolate_never_extrapolates() -> None:
    idxs = np.array([2, 4, 8])
    centers = np.array([[0, 0], [2, 2], [6, 6]], np.float32)
    track = interpolate_track(idxs, centers, n_frames=12)
    assert np.isnan(track[:2]).all() and np.isnan(track[9:]).all()  # honesty boundary
    assert np.allclose(track[2], [0, 0]) and np.allclose(track[8], [6, 6])
    assert np.allclose(track[3], [1, 1]) and np.allclose(track[6], [4, 4])  # interior interp


def test_interpolate_with_unobserved_samples() -> None:
    idxs = np.array([0, 5, 10])
    centers = np.array([[0, 0], [np.nan, np.nan], [10, 10]], np.float32)
    track = interpolate_track(idxs, centers, n_frames=11)
    assert np.allclose(track[5], [5, 5])  # interior gap bridged from observed neighbors


def test_interpolate_all_unobserved() -> None:
    track = interpolate_track(np.array([0, 1]), np.full((2, 2), np.nan, np.float32), 5)
    assert np.isnan(track).all()


def test_dense_visibility_shows_misses() -> None:
    vis = dense_visibility(np.array([0, 3, 6]), np.array([0.9, 0.0, 0.8]), n_frames=9)
    assert vis[0] == vis[2] == np.float32(0.9)
    assert vis[3] == vis[5] == 0.0  # the miss is visible to the permanence gate
    assert vis[6] == vis[8] == np.float32(0.8)


def test_motion_consistency_catches_false_latch() -> None:
    from woracle.grounders.linking import motion_consistency

    static = np.tile(np.array([[50.0, 50.0]], np.float32), (10, 1))
    static += np.random.default_rng(0).normal(0, 0.1, static.shape).astype(np.float32)
    moving = np.stack([np.linspace(0, 60, 10), np.linspace(0, 40, 10)], 1).astype(np.float32)

    ok, _ = motion_consistency(static, "co_moves_with_effector", image_diag=160.0)
    assert not ok  # static track for a co-moving role = false latch
    ok, _ = motion_consistency(moving, "co_moves_with_effector", image_diag=160.0)
    assert ok
    ok, _ = motion_consistency(static, "static", image_diag=160.0)
    assert ok
    ok, _ = motion_consistency(moving, "static", image_diag=160.0)
    assert not ok  # wandering track for a static role = relatch/morph
    # free roles and tiny evidence carry no expectation
    ok, _ = motion_consistency(moving, "free", image_diag=160.0)
    assert ok
    ok, _ = motion_consistency(static[:1], "co_moves_with_effector", image_diag=160.0)
    assert ok
