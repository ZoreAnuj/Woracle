from __future__ import annotations

import numpy as np
import pytest

from woracle.channels.predicates import RoleData, eval_predicate
from woracle.contracts import Predicate
from woracle.errors import SpecError


def _roles(sub_track, obj_mask=None, obj_track=None, sub_vis=None):
    roles = {"sub": RoleData(track=np.asarray(sub_track, np.float32), visibility=sub_vis)}
    if obj_mask is not None or obj_track is not None:
        roles["obj"] = RoleData(
            track=None if obj_track is None else np.asarray(obj_track, np.float32),
            mask=obj_mask,
        )
    return roles


def test_contained_inside_and_outside() -> None:
    mask = np.zeros((1, 50, 50), np.uint8)
    mask[0, 10:40, 10:40] = 1
    pred = Predicate(kind="contained", subject="sub", object="obj", params={"erode_px": 5})
    inside = eval_predicate(pred, _roles([[25, 25]], obj_mask=mask), slice(0, 1))
    outside = eval_predicate(pred, _roles([[12, 12]], obj_mask=mask), slice(0, 1))
    assert inside.holds and inside.margin > 0
    assert not outside.holds  # within bbox but inside the eroded wall band


def test_stationary_margins() -> None:
    still = eval_predicate(
        Predicate(kind="stationary", subject="sub", params={"tol": 1.0}),
        _roles([[5, 5], [5.2, 5.0], [5.1, 5.1]]),
        slice(0, 3),
    )
    moving = eval_predicate(
        Predicate(kind="stationary", subject="sub", params={"tol": 1.0}),
        _roles([[0, 0], [4, 0], [8, 0]]),
        slice(0, 3),
    )
    assert still.holds and not moving.holds


def test_approaching() -> None:
    obj = [[10.0, 0.0]] * 5
    toward = eval_predicate(
        Predicate(kind="approaching", subject="sub", object="obj"),
        _roles([[0, 0], [2, 0], [4, 0], [6, 0], [8, 0]], obj_track=obj),
        slice(0, 5),
    )
    away = eval_predicate(
        Predicate(kind="approaching", subject="sub", object="obj"),
        _roles([[8, 0], [6, 0], [4, 0], [2, 0], [0, 0]], obj_track=obj),  # receding from x=10
        slice(0, 5),
    )
    assert toward.holds and not away.holds


def test_nan_window_is_evidence_missing_not_fail() -> None:
    track = np.array([[1.0, 1.0], [np.nan, np.nan]], np.float32)
    res = eval_predicate(
        Predicate(kind="stationary", subject="sub"), {"sub": RoleData(track=track)}, slice(0, 2)
    )
    assert res.status == "evidence_missing" and not res.holds
    assert "unobserved" in res.reason


def test_present_uses_visibility() -> None:
    vis = np.array([1.0, 1.0, 0.0, 0.0], np.float32)
    roles = {"sub": RoleData(track=np.zeros((4, 2), np.float32), visibility=vis)}
    early = eval_predicate(Predicate(kind="present", subject="sub"), roles, slice(0, 2))
    late = eval_predicate(Predicate(kind="present", subject="sub"), roles, slice(2, 4))
    assert early.holds and not late.holds


def test_missing_role_is_evidence_missing() -> None:
    res = eval_predicate(Predicate(kind="stationary", subject="ghost"), {}, slice(0, 1))
    assert res.status == "evidence_missing" and not res.holds
    assert "no data" in res.reason


def test_empty_window_is_evidence_missing() -> None:
    res = eval_predicate(
        Predicate(kind="stationary", subject="sub"),
        {"sub": RoleData(track=np.zeros((4, 2), np.float32))},
        slice(4, 4),
    )
    assert res.status == "evidence_missing"
    assert "empty" in res.reason


def test_empty_mask_is_evidence_missing() -> None:
    res = eval_predicate(
        Predicate(kind="contained", subject="sub", object="obj"),
        {
            "sub": RoleData(track=np.zeros((2, 2), np.float32)),
            "obj": RoleData(mask=np.zeros((2, 8, 8), np.uint8)),
        },
        slice(0, 2),
    )
    assert res.status == "evidence_missing"
    assert "empty" in res.reason


def test_binary_kind_without_object_is_spec_error() -> None:
    with pytest.raises(SpecError):
        eval_predicate(
            Predicate(kind="co_located", subject="sub"),
            _roles([[0, 0]]),
            slice(0, 1),
        )
