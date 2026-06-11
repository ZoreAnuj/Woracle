"""P2 gate signals on blobworld + corruptions (ground truth by construction)."""

from __future__ import annotations

import os

import pytest

from woracle.gate.signals import (
    ActionVideoConsistencySignal,
    AppearanceConsistencySignal,
    BackgroundDriftSignal,
    TrackContinuitySignal,
)
from woracle.io import load_rollout, save_episode
from woracle.testing.blobworld import blob_spec, make_episode
from woracle.testing.corruptions import corrupt
from woracle.testing.plugins import BlobColorGrounder


def _ground(frames, tmp_path, name, actions=None):
    ep = str(tmp_path / name)
    save_episode(ep, name, frames, source="blobworld", actions=actions)
    ref = load_rollout(ep)
    out = str(tmp_path / f"{name}_g")
    os.makedirs(out, exist_ok=True)
    return BlobColorGrounder().ground(ref, blob_spec(), out)


@pytest.fixture(scope="module")
def clean(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("sig")
    frames, truth = make_episode("success", seed=0)
    return _ground(frames, tmp, "clean", actions=truth.actions), frames, truth, tmp


def test_background_drift_clean_vs_morph(clean, tmp_path) -> None:
    grounded, frames, _t, _ = clean
    v_clean = BackgroundDriftSignal().measure(grounded)
    assert v_clean.status == "ok" and v_clean.value is not None and v_clean.value > 0.7

    morphed = corrupt(frames, "bg_morph", seed=1)
    g2 = _ground(morphed, tmp_path, "morph")
    v_morph = BackgroundDriftSignal().measure(g2)
    assert v_morph.status == "ok" and v_morph.value is not None
    assert v_morph.value < v_clean.value - 0.2, (v_morph.value, v_clean.value)


def test_background_drift_catches_noise_ramp(clean, tmp_path) -> None:
    _g, frames, _t, _ = clean
    noisy = corrupt(frames, "noise_ramp", seed=2)
    g = _ground(noisy, tmp_path, "noise")
    v = BackgroundDriftSignal().measure(g)
    assert v.status == "ok" and v.value is not None and v.value < 0.5


def test_appearance_consistency_clean_vs_morph(clean, tmp_path) -> None:
    grounded, frames, _t, _ = clean
    v = AppearanceConsistencySignal().measure(grounded)
    assert v.status == "ok" and v.value is not None and v.value > 0.4
    # the large static receptacle is the stable anchor (binding-study F4)
    assert v.details["receptacle"] > 0.8

    # bg_morph mutates everything except the carried object: the receptacle's
    # appearance must degrade hard relative to clean (the discriminative claim)
    morphed = corrupt(frames, "bg_morph", seed=4)
    g2 = _ground(morphed, tmp_path, "amorph")
    v2 = AppearanceConsistencySignal().measure(g2)
    assert v2.status == "ok"
    assert v2.details["receptacle"] < v.details["receptacle"] - 0.3


def test_track_continuity_flags_teleports(clean, tmp_path) -> None:
    grounded, frames, _t, _ = clean
    v_clean = TrackContinuitySignal().measure(grounded)
    assert v_clean.status == "ok" and v_clean.value is not None and v_clean.value > 0.5

    tele = corrupt(frames, "teleport", seed=3)
    g2 = _ground(tele, tmp_path, "tele")
    v_tele = TrackContinuitySignal().measure(g2)
    assert v_tele.status == "ok" and v_tele.value is not None
    assert v_tele.value < v_clean.value - 0.3, (v_tele.value, v_clean.value)


def test_action_video_consistency_true_vs_scrambled(clean, tmp_path) -> None:
    grounded, frames, truth, _ = clean
    v_true = ActionVideoConsistencySignal().measure(grounded)
    assert v_true.status == "ok" and v_true.value is not None and v_true.value > 0.8

    # Same video, CONTRADICTORY actions (negated commands): the WM "ignored"
    # the policy — the signal must drop hard.
    g2 = _ground(frames, tmp_path, "scrambled", actions=-truth.actions)
    v_bad = ActionVideoConsistencySignal().measure(g2)
    assert v_bad.status == "ok" and v_bad.value is not None
    assert v_bad.value < 0.3, v_bad.value


def test_action_video_missing_actions_is_evidence_missing(clean, tmp_path) -> None:
    _g, frames, _t, _ = clean
    g = _ground(frames, tmp_path, "noact", actions=None)
    v = ActionVideoConsistencySignal().measure(g)
    assert v.status == "evidence_missing" and "no action stream" in v.reason
