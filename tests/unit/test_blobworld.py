from __future__ import annotations

import numpy as np
import pytest

from woracle.contracts import digest_array
from woracle.testing.blobworld import INTERIOR, KINDS, make_episode


def test_deterministic_per_seed() -> None:
    f1, _ = make_episode("success", seed=3)
    f2, _ = make_episode("success", seed=3)
    f3, _ = make_episode("success", seed=4)
    assert digest_array(f1) == digest_array(f2)
    assert digest_array(f1) != digest_array(f3)


def test_frames_shape_and_dtype() -> None:
    frames, _ = make_episode("success", seed=0, n_frames=24)
    assert frames.shape == (24, 96, 128, 3) and frames.dtype == np.uint8


def test_success_truth_contained_and_settled() -> None:
    _, truth = make_episode("success", seed=0)
    t = truth.events["contained_from"]
    end = truth.carried[-1]
    assert INTERIOR[0] < end[0] < INTERIOR[2]
    assert INTERIOR[1] < end[1] < INTERIOR[3] + 6 + 1  # settles low in the cup
    tail = truth.carried[t:]
    assert np.linalg.norm(np.diff(tail, axis=0), axis=1).max() < 1e-6


def test_fail_miss_ends_outside_interior() -> None:
    _, truth = make_episode("fail_miss", seed=0)
    end = truth.carried[-1]
    assert not (INTERIOR[0] < end[0] < INTERIOR[2])


def test_vanish_removes_red_pixels() -> None:
    frames, truth = make_episode("vanish", seed=0)
    k = truth.events["vanish_frame"]
    red_before = np.all((frames[k - 1] >= [150, 0, 0]) & (frames[k - 1] <= [255, 90, 90]), -1)
    red_after = np.all((frames[k] >= [150, 0, 0]) & (frames[k] <= [255, 90, 90]), -1)
    assert red_before.sum() > 0
    assert red_after.sum() == 0
    assert np.isnan(truth.carried[k:]).all()


def test_drop_freezes_carried_object() -> None:
    _, truth = make_episode("fail_drop", seed=0)
    t = truth.events["drop_frame"]
    frozen = truth.carried[t:]
    assert np.linalg.norm(np.diff(frozen, axis=0), axis=1).max() < 1e-6
    # gripper keeps moving after the drop
    assert np.linalg.norm(np.diff(truth.gripper[t : t + 5], axis=0), axis=1).max() > 0.1


@pytest.mark.parametrize("kind", KINDS)
def test_all_kinds_render(kind: str) -> None:
    frames, truth = make_episode(kind, seed=1, n_frames=30)
    assert frames.shape[0] == 30
    assert truth.label in ("success", "fail")
