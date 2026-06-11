"""Regression tests for the P0 critic findings (C1, C2, I1-I7, N10).

Each test reproduces the critic's verified failure scenario and asserts the
fixed behavior. These are honesty/caching invariants — do not weaken.
"""

from __future__ import annotations

import os

import numpy as np
import pydantic
import pytest

import woracle
from woracle.contracts import ChannelScore, GradeCard, Provenance, SuccessReport
from woracle.contracts.gate import GateReport
from woracle.errors import PluginError, SpecError, StoreError
from woracle.gate import DEFAULT_POLICY, compose_gate
from woracle.io import load_rollout, save_episode
from woracle.pipeline import blob_profile, grade_rollouts
from woracle.pipeline.run import _frames_digest, _ground_cached, _success_from_channels
from woracle.reporting import build_leaderboard
from woracle.store import ContentStore
from woracle.testing.blobworld import blob_spec, make_episode
from woracle.testing.plugins import (
    BindingHealthSignal,
    BlobColorGrounder,
    PermanenceSignal,
    PredicateSuccessChannel,
    role_data,
)


# ---------------------------------------------------------------- C1 --------
def test_c1_empty_digest_never_reaches_cache_key() -> None:
    with pytest.raises(StoreError, match="empty digest"):
        ContentStore.key_for(
            stage="ground", component="g", component_version="1", inputs={"frames": ""}
        )


def test_c1_blank_recorded_digest_recomputed_from_payload(tmp_path) -> None:
    frames_a, _ = make_episode("success", seed=0)
    frames_b, _ = make_episode("fail_miss", seed=1)
    ra = save_episode(str(tmp_path / "a"), "a", frames_a, source="blobworld")
    rb = save_episode(str(tmp_path / "b"), "b", frames_b, source="blobworld")
    ra, rb = load_rollout(str(tmp_path / "a")), load_rollout(str(tmp_path / "b"))
    ra.frames.sha256 = ""  # hostile/hand-written rollout.json
    rb.frames.sha256 = ""
    da, db = _frames_digest(ra), _frames_digest(rb)
    assert da and db and da != db  # distinct keys -> no cross-rollout aliasing


def test_c1_unreadable_payload_with_blank_digest_is_loud(tmp_path) -> None:
    frames, _ = make_episode("success", seed=0)
    save_episode(str(tmp_path / "a"), "a", frames, source="blobworld")
    ref = load_rollout(str(tmp_path / "a"))
    ref.frames.sha256 = ""
    ref.meta["_dir"] = str(tmp_path / "nonexistent")
    with pytest.raises(StoreError, match="cannot build a safe cache key"):
        _frames_digest(ref)


# ---------------------------------------------------------------- C2 --------
def _grounded_for(frames: np.ndarray, tmp_path, name: str):
    ep = str(tmp_path / name)
    save_episode(ep, name, frames, source="blobworld")
    ref = load_rollout(ep)
    out = str(tmp_path / f"{name}_grounded")
    os.makedirs(out, exist_ok=True)
    return BlobColorGrounder().ground(ref, blob_spec(), out)


def test_c2_late_vanish_abstains_instead_of_failing(tmp_path) -> None:
    """Object deleted only in the decisive window: permanence stays above the
    hard floor, so the GATE passes — the CHANNEL must abstain, not fail."""
    frames, _truth = make_episode("success", seed=0)
    lo, hi = np.array([150, 0, 0]), np.array([255, 90, 90])
    for t in (len(frames) - 2, len(frames) - 1):
        red = np.all((frames[t] >= lo) & (frames[t] <= hi), axis=-1)
        frames[t][red] = (200, 200, 200)  # WM glitch at the decisive moment
    grounded = _grounded_for(frames, tmp_path, "late_vanish")

    perm = PermanenceSignal().measure(grounded)
    assert perm.status == "ok" and perm.value is not None and perm.value > 0.5  # gate passes

    score = PredicateSuccessChannel().score(grounded, blob_spec())
    assert score.status == "evidence_missing"
    assert "unobserved" in score.reason

    success = _success_from_channels([PredicateSuccessChannel()], [score], 0.5)
    assert success.verdict == "abstain"
    assert not success.violated  # never attribute unevaluable predicates as violations


# ---------------------------------------------------------------- I1 --------
def test_i1_optional_unbound_role_does_not_gate(tmp_path) -> None:
    spec = blob_spec()
    gripper = spec.role("gripper")
    assert not gripper.required
    gripper.candidates = ["invisible widget"]  # blob grounder cannot bind this
    frames, _ = make_episode("success", seed=0)
    ep = str(tmp_path / "ep")
    save_episode(ep, "ep", frames, source="blobworld")
    ref = load_rollout(ep)
    out = str(tmp_path / "g")
    os.makedirs(out, exist_ok=True)
    grounded = BlobColorGrounder().ground(ref, spec, out)
    assert not grounded.binding("gripper").bound
    assert not grounded.binding("gripper").required

    health = BindingHealthSignal().measure(grounded)
    assert health.status == "ok"  # optional role never produces evidence_missing

    gate = compose_gate(grounded, [health], DEFAULT_POLICY)
    assert gate.verdict == "gradeable"
    assert any("no gate impact" in r for r in gate.reasons)  # annotated, not punished


# ---------------------------------------------------------------- I2 --------
def test_i2_nonfinite_channel_values_rejected_at_contract() -> None:
    for bad in (float("nan"), float("inf")):
        with pytest.raises(pydantic.ValidationError):
            ChannelScore(channel="c", value=bad)


def test_i2_verdict_fold_treats_nonfinite_as_missing() -> None:
    class Eligible:
        name = "e"
        version = "0"

        class caps:
            verdict_eligible = True

    # Bypass the contract deliberately (model_construct skips validation) to
    # prove the fold has its own defense in depth.
    sc = ChannelScore.model_construct(
        channel="e",
        status="ok",
        value=float("nan"),
        confidence=None,
        reason="",
        series={},
        details={},
    )
    out = _success_from_channels([Eligible()], [sc], 0.5)
    assert out.verdict == "abstain"


# ---------------------------------------------------------------- I3 --------
def test_i3_window_from_artifacts_not_metadata(tmp_path) -> None:
    frames, _ = make_episode("success", seed=0)
    grounded = _grounded_for(frames, tmp_path, "meta")
    grounded.rollout.n_frames = 0  # hostile metadata
    score = PredicateSuccessChannel().score(grounded, blob_spec())
    assert score.status == "ok" and score.value == 1.0  # artifacts know the length


# ---------------------------------------------------------------- I4 --------
def test_i4_forgetful_grounder_does_not_poison_cache(tmp_path) -> None:
    class ForgetfulGrounder:
        name = "blob.color"  # same identity as the real one -> same cache key
        version = "0.1.0"

        def ground(self, rollout, spec, out_dir):  # writes NOTHING
            return None

    frames, _ = make_episode("success", seed=0)
    ep = str(tmp_path / "ep")
    save_episode(ep, "ep", frames, source="blobworld")
    ref = load_rollout(ep)
    store = ContentStore(str(tmp_path / "store"))
    spec = blob_spec()
    with pytest.raises(PluginError, match=r"no grounded\.json"):
        _ground_cached(store, ForgetfulGrounder(), ref, spec)
    # No half-entry committed: the REAL grounder now succeeds on the same key.
    grounded, _key, hit = _ground_cached(store, BlobColorGrounder(), ref, spec)
    assert not hit and grounded.binding("carried_object").bound


# ---------------------------------------------------------------- I5 --------
def test_i5_grade_never_mutates_config_template(blob_dataset, tmp_path) -> None:
    template = blob_profile()
    before = (template.out_dir, template.store_root)
    spec_path = os.path.join(blob_dataset, "spec.yaml")
    woracle.grade(blob_dataset, spec_path, out_dir=str(tmp_path / "a"), config=template)
    woracle.grade(blob_dataset, spec_path, out_dir=str(tmp_path / "b"), config=template)
    assert (template.out_dir, template.store_root) == before
    assert os.path.isdir(tmp_path / "a" / "store")
    assert os.path.isdir(tmp_path / "b" / "store")  # second run got its own store


# ---------------------------------------------------------------- I6 --------
def test_i6_duplicate_rollout_ids_rejected(tmp_path) -> None:
    frames, _ = make_episode("success", seed=0)
    a = save_episode(str(tmp_path / "a"), "ep", frames, source="blobworld")
    b = save_episode(str(tmp_path / "b"), "ep", frames, source="blobworld")
    a, b = load_rollout(str(tmp_path / "a")), load_rollout(str(tmp_path / "b"))
    cfg = blob_profile()
    cfg.store_root = str(tmp_path / "store")
    cfg.out_dir = str(tmp_path / "out")
    with pytest.raises(PluginError, match="duplicate rollout ids"):
        grade_rollouts([a, b], blob_spec(), cfg)


def test_i6_card_filenames_sanitized(tmp_path) -> None:
    frames, _ = make_episode("success", seed=0)
    save_episode(str(tmp_path / "weird"), "run-3/init 07", frames, source="blobworld")
    ref = load_rollout(str(tmp_path / "weird"))
    cfg = blob_profile()
    cfg.store_root = str(tmp_path / "store")
    cfg.out_dir = str(tmp_path / "out")
    cards = grade_rollouts([ref], blob_spec(), cfg)
    assert cards[0].rollout_id == "run-3/init 07"  # identity preserved in the card
    assert os.path.isfile(tmp_path / "out" / "cards" / "run-3_init_07.json")


# ---------------------------------------------------------------- I7 --------
def _mini_card(spec_hash: str) -> GradeCard:
    return GradeCard(
        rollout_id="r",
        policy="p",
        spec_name="s",
        spec_version=1,
        spec_hash=spec_hash,
        gate=GateReport(verdict="gradeable"),
        success=SuccessReport(verdict="pass"),
        provenance=Provenance(),
    )


def test_i7_leaderboard_rejects_mixed_specs() -> None:
    with pytest.raises(ValueError, match="ONE spec"):
        build_leaderboard([_mini_card("aaa"), _mini_card("bbb")])


# ---------------------------------------------------------------- N10 -------
def test_n10_migration_version_jump_rejected() -> None:
    from woracle.contracts.migrations import _MIGRATIONS, CURRENT_VERSIONS, migrate, migration

    CURRENT_VERSIONS["_TestJump"] = 3
    try:

        @migration("_TestJump", from_version=1)
        def _v1(d: dict) -> dict:
            d["schema_version"] = 3  # illegal jump over v2
            return d

        with pytest.raises(SpecError, match="exactly one version"):
            migrate("_TestJump", {"schema_version": 1})
    finally:
        del CURRENT_VERSIONS["_TestJump"]
        _MIGRATIONS.pop("_TestJump", None)


# ------------------------------------------------- verdict isolation --------
def test_verdict_isolation_rank_only_channels_cannot_touch_success() -> None:
    class RankOnly:
        name = "rank"
        version = "0"

        class caps:
            verdict_eligible = False

    class Eligible:
        name = "elig"
        version = "0"

        class caps:
            verdict_eligible = True

    terrible_rank = ChannelScore(channel="rank", value=0.0)
    missing_rank = ChannelScore(channel="rank", status="evidence_missing", reason="x")
    good_verdict = ChannelScore(channel="elig", value=1.0, details={"pred:ok()": 1.0})

    out1 = _success_from_channels([RankOnly(), Eligible()], [terrible_rank, good_verdict], 0.5)
    out2 = _success_from_channels([RankOnly(), Eligible()], [missing_rank, good_verdict], 0.5)
    assert out1.verdict == "pass"  # rank-only zero cannot flip the verdict
    assert out2.verdict == "pass"  # rank-only missing cannot force abstain


# ----------------------------------------- vanish progress honesty ----------
def test_vanished_track_progress_is_not_confident(tmp_path) -> None:
    frames, _ = make_episode("vanish", seed=0)
    grounded = _grounded_for(frames, tmp_path, "vanish")
    roles = role_data(grounded)
    track = roles["carried_object"].track
    assert track is not None and np.isnan(track).any()  # grounder recorded the absence
    from woracle.testing.plugins import GoalDistanceProgress

    score = GoalDistanceProgress().score(grounded, blob_spec())
    if score.status == "ok":
        assert score.confidence is not None and score.confidence < 1.0


# ------------------------------------------- re-verification NEW issues -----
def test_new1_sanitized_filename_collision_rejected(tmp_path) -> None:
    frames, _ = make_episode("success", seed=0)
    save_episode(str(tmp_path / "x"), "a/b", frames, source="blobworld")
    save_episode(str(tmp_path / "y"), "a_b", frames, source="blobworld")
    a, b = load_rollout(str(tmp_path / "x")), load_rollout(str(tmp_path / "y"))
    cfg = blob_profile()
    cfg.store_root = str(tmp_path / "store")
    cfg.out_dir = str(tmp_path / "out")
    with pytest.raises(PluginError, match="collide after filename sanitization"):
        grade_rollouts([a, b], blob_spec(), cfg)


def test_new2_report_function_survives_subpackage_import(blob_dataset, tmp_path) -> None:
    """The reporting subpackage must never shadow the woracle.report function."""
    import woracle.reporting

    cards = woracle.grade(
        blob_dataset, os.path.join(blob_dataset, "spec.yaml"), out_dir=str(tmp_path / "o")
    )
    board1 = woracle.report(cards)
    board2 = woracle.report(cards)  # second call used to hit a module, not a function
    assert callable(woracle.report)
    assert board1.spec_hash == board2.spec_hash


def test_new3_fresh_honors_explicit_overrides() -> None:
    from woracle.contracts import GatePolicy

    template = blob_profile()
    custom_policy = GatePolicy(required_signals=["only_this"])
    out = template.fresh(policy=custom_policy, signals=["s1"], channels=["c1"])
    assert out.policy.required_signals == ["only_this"]
    assert out.signals == ["s1"] and out.channels == ["c1"]
    # and non-overridden fields are still deep copies, not shared refs
    plain = template.fresh(out_dir="elsewhere")
    plain.policy.required_signals.append("mutant")
    assert "mutant" not in template.policy.required_signals
