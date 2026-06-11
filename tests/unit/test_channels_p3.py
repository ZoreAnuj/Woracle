"""P3 channels: DTW trajectory, ordered phase coverage, TL-DTMC, GVL protocol."""

from __future__ import annotations

import os

import numpy as np
import pytest

from woracle.channels.phases import OrderedPhaseCoverageChannel
from woracle.channels.tlcheck import TLSatisfactionChannel, dtmc_eventually_sustained
from woracle.channels.trajectory import TrajectoryDTWChannel, dtw_distance
from woracle.io import load_rollout, save_episode
from woracle.judges.base import parse_progress_reply, value_order_correlation
from woracle.judges.progress_gvl import GVLProgressChannel
from woracle.testing.blobworld import blob_spec, make_episode
from woracle.testing.oracle_backend import BlobProgressOracleBackend
from woracle.testing.plugins import BlobColorGrounder


def _ground(frames, tmp_path, name):
    ep = str(tmp_path / name)
    save_episode(ep, name, frames, source="blobworld")
    out = str(tmp_path / f"{name}_g")
    os.makedirs(out, exist_ok=True)
    return BlobColorGrounder().ground(load_rollout(ep), blob_spec(), out)


# ---------------------------------------------------------------- DTW -------
def test_dtw_identity_and_ordering() -> None:
    a = np.stack([np.linspace(0, 10, 20), np.zeros(20)], 1)
    b = a[::2]  # same path, different sampling
    c = a + np.array([0.0, 8.0])  # parallel path, offset (numpy broadcast)
    assert dtw_distance(a, a) == 0.0
    assert dtw_distance(a, b) < 0.5
    assert dtw_distance(a, c) > dtw_distance(a, b)


def test_trajectory_channel_ranks_success_over_failures(tmp_path) -> None:
    spec = blob_spec()
    # demo reference: the truth trajectory of a DIFFERENT seed's success episode
    _, demo_truth = make_episode("success", seed=11)
    chan = TrajectoryDTWChannel(demo_tracks=[demo_truth.carried], role="carried_object")

    scores = {}
    for kind in ("success", "fail_miss", "random"):
        frames, _ = make_episode(kind, seed=3)
        g = _ground(frames, tmp_path, f"traj_{kind}")
        sc = chan.score(g, spec)
        assert sc.status == "ok", sc.reason
        scores[kind] = sc.value
    assert scores["success"] > scores["fail_miss"] > scores["random"], scores


def test_trajectory_channel_without_refs_is_evidence_missing(tmp_path) -> None:
    frames, _ = make_episode("success", seed=0)
    g = _ground(frames, tmp_path, "norefs")
    sc = TrajectoryDTWChannel().score(g, blob_spec())
    assert sc.status == "evidence_missing" and "no demo reference" in sc.reason


# ---------------------------------------------------------- phase order -----
def test_phase_coverage_full_on_success_partial_on_miss(tmp_path) -> None:
    spec = blob_spec()
    chan = OrderedPhaseCoverageChannel()
    f_succ, _ = make_episode("success", seed=0)
    f_miss, _ = make_episode("fail_miss", seed=0)
    s_succ = chan.score(_ground(f_succ, tmp_path, "ph_s"), spec)
    s_miss = chan.score(_ground(f_miss, tmp_path, "ph_m"), spec)
    assert s_succ.status == "ok" and s_succ.value == 1.0, s_succ
    assert s_miss.status == "ok" and s_miss.value is not None
    assert s_miss.value < 1.0  # insert phase never satisfied
    assert s_miss.details["t_first:insert"] == -1.0


# ---------------------------------------------------------------- DTMC ------
def test_dtmc_math_limits() -> None:
    assert dtmc_eventually_sustained(np.ones(10), 3) == pytest.approx(1.0)
    assert dtmc_eventually_sustained(np.zeros(10), 1) == 0.0
    # exact small case: q=0.5, k=1, T=2  ->  1 - 0.25
    assert dtmc_eventually_sustained(np.array([0.5, 0.5]), 1) == pytest.approx(0.75)
    # exact: k=2, T=2, q=0.5 -> both must succeed = 0.25
    assert dtmc_eventually_sustained(np.array([0.5, 0.5]), 2) == pytest.approx(0.25)
    # monotone in k
    q = np.full(20, 0.6)
    assert dtmc_eventually_sustained(q, 2) > dtmc_eventually_sustained(q, 5)


def test_tl_channel_orders_success_over_miss(tmp_path) -> None:
    spec = blob_spec()
    chan = TLSatisfactionChannel()
    f_succ, _ = make_episode("success", seed=0)
    f_miss, _ = make_episode("fail_miss", seed=0)
    p_succ = chan.score(_ground(f_succ, tmp_path, "tl_s"), spec)
    p_miss = chan.score(_ground(f_miss, tmp_path, "tl_m"), spec)
    assert p_succ.status == "ok" and p_succ.value is not None and p_succ.value > 0.9
    assert p_miss.status == "ok" and p_miss.value is not None and p_miss.value < 0.1
    assert not chan.caps.verdict_eligible  # achieved-then-undone must not pass verdicts


# ---------------------------------------------------------------- GVL -------
def test_parse_progress_reply_formats() -> None:
    assert parse_progress_reply("Frame 1: 30%\nFrame 2: 60%", 2) == [0.3, 0.6]
    assert parse_progress_reply("1 = 45\n2 - 80%", 2) == [0.45, 0.8]
    out = parse_progress_reply("about 20% then 90% I think", 2)
    assert out == [0.2, 0.9]
    assert parse_progress_reply("no numbers here", 2) == [None, None]
    assert parse_progress_reply("Frame 1: 250%", 1) == [1.0]  # clamped


def test_voc_directions() -> None:
    assert value_order_correlation([0.1, 0.4, 0.9], [0, 1, 2]) == pytest.approx(1.0)
    assert value_order_correlation([0.9, 0.4, 0.1], [0, 1, 2]) == pytest.approx(-1.0)
    assert value_order_correlation([0.5, 0.5, 0.5], [0, 1, 2]) == 0.0


def test_gvl_channel_end_to_end_with_oracle(tmp_path) -> None:
    """Full protocol against the geometry oracle: success ends high with
    positive VOC; fail_miss ends lower; degenerate replies abstain."""
    spec = blob_spec()
    f_succ, _ = make_episode("success", seed=0)
    f_miss, _ = make_episode("fail_miss", seed=0)
    g_succ = _ground(f_succ, tmp_path, "gvl_s")
    g_miss = _ground(f_miss, tmp_path, "gvl_m")

    chan = GVLProgressChannel(backend=BlobProgressOracleBackend(), n_frames=10)
    s = chan.score(g_succ, spec)
    m = chan.score(g_miss, spec)
    assert s.status == "ok" and s.value is not None and s.value > 0.75, s
    assert s.details["voc"] > 0.8  # ordered progress recovered
    assert s.details["reshuffle_agreement"] > 0.9  # deterministic oracle agrees
    assert m.status == "ok" and m.value is not None
    assert s.value > m.value + 0.15

    class GarbageBackend:
        name = "test.garbage"
        version = "0"

        def complete(self, images, prompt):
            return "I cannot help with that."

    g = GVLProgressChannel(backend=GarbageBackend(), n_frames=8).score(g_succ, spec)
    assert g.status == "evidence_missing" and "unparseable" in g.reason


def test_gvl_without_backend_is_evidence_missing(tmp_path) -> None:
    frames, _ = make_episode("success", seed=0)
    g = _ground(frames, tmp_path, "gvl_nb")
    sc = GVLProgressChannel().score(g, blob_spec())
    assert sc.status == "evidence_missing" and "backend" in sc.reason
