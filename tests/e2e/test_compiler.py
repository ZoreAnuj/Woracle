"""P4 exit gates.

1. compile_spec on Scene-A demos -> accepted spec whose self-test report shows
   demos PASS + minted negatives FAIL (the VLM-CaR gate, run for real).
2. The compiled spec grades Scene-A test episodes correctly via the
   RELATIONAL grounder (never the color grounder, never blob_spec()).
3. CROSS-SCENE: the SAME spec grades Scene-B episodes (different colors,
   mirrored layout) correctly — compile once, bind later, demonstrated.
4. REFUSE: demos with no recoverable task structure raise SpecError.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from woracle.compiler import compile_spec, mint_negatives
from woracle.contracts import TaskSpec
from woracle.errors import SpecError
from woracle.grounders.relational import RelationalMotionGrounder
from woracle.io import load_rollout, save_episode
from woracle.testing.blobworld import SCENE_B, make_episode
from woracle.testing.plugins import PredicateSuccessChannel


@pytest.fixture(scope="module")
def compiled() -> TaskSpec:
    demos = [make_episode("success", seed=s)[0] for s in (0, 1, 2)]
    return compile_spec(demos, "put the small object into the container", name="blob-family")


def _verdict(spec: TaskSpec, frames: np.ndarray, tmp_path, tag: str) -> str:
    ep = str(tmp_path / f"ep_{tag}")
    save_episode(ep, tag, frames, source="compiler-test")
    ref = load_rollout(ep)
    out = str(tmp_path / f"g_{tag}")
    os.makedirs(out, exist_ok=True)
    grounded = RelationalMotionGrounder().ground(ref, spec, out)
    score = PredicateSuccessChannel().score(grounded, spec)
    if score.status != "ok" or score.value is None:
        return "abstain"
    return "pass" if score.value >= 0.5 else "fail"


def test_compiler_accepts_with_full_selftest(compiled: TaskSpec) -> None:
    st = compiled.spec_provenance.self_test
    assert st.ran and st.accepted
    assert st.demos_passed == st.demos_total == 3
    assert st.negatives_failed == st.negatives_total == 8  # 4 mints x 2 demos
    assert compiled.spec_provenance.compiler.startswith("woracle.compile")
    assert len(compiled.spec_provenance.demo_digests) == 3
    # roles are RELATIONAL: definitions speak of motion/relations, not colors
    for role in compiled.roles:
        assert role.name in ("carried_object", "receptacle", "gripper")
        assert "red" not in role.definition and "color" not in role.definition


def test_compiled_spec_grades_scene_a(compiled: TaskSpec, tmp_path) -> None:
    succ, _ = make_episode("success", seed=7)  # unseen seed
    miss, _ = make_episode("fail_miss", seed=7)
    drop, _ = make_episode("fail_drop", seed=7)
    assert _verdict(compiled, succ, tmp_path, "a_succ") == "pass"
    assert _verdict(compiled, miss, tmp_path, "a_miss") == "fail"
    assert _verdict(compiled, drop, tmp_path, "a_drop") == "fail"


def test_cross_scene_transfer_to_scene_b(compiled: TaskSpec, tmp_path) -> None:
    """THE compile-once/bind-later claim: same spec, new appearance + layout."""
    succ, _ = make_episode("success", seed=5, scene=SCENE_B)
    miss, _ = make_episode("fail_miss", seed=5, scene=SCENE_B)
    assert _verdict(compiled, succ, tmp_path, "b_succ") == "pass"
    assert _verdict(compiled, miss, tmp_path, "b_miss") == "fail"


def test_mints_are_real_negatives() -> None:
    frames, _ = make_episode("success", seed=0)
    mints = mint_negatives(frames)
    assert {k for k, _ in mints} == {"reversed", "frozen", "truncated", "stalled"}
    for _kind, mf in mints:
        assert mf.shape == frames.shape  # length never the discriminator
    rev = dict(mints)["reversed"]
    assert np.array_equal(rev[0], frames[-1])


def test_compiler_refuses_structureless_demos() -> None:
    """Random walks have no settle-at-static structure: REFUSE, don't emit."""
    demos = [make_episode("random", seed=s)[0] for s in (0, 1, 2)]
    with pytest.raises(SpecError, match="REFUSE"):
        compile_spec(demos, "do something random")


def test_compiler_needs_two_demos() -> None:
    frames, _ = make_episode("success", seed=0)
    with pytest.raises(SpecError, match="at least 2 demos"):
        compile_spec([frames], "task")


def test_cross_scene_b_more_failure_modes(compiled: TaskSpec, tmp_path) -> None:
    drop, _ = make_episode("fail_drop", seed=9, scene=SCENE_B)
    rnd, _ = make_episode("random", seed=9, scene=SCENE_B)
    assert _verdict(compiled, drop, tmp_path, "b_drop") == "fail"
    assert _verdict(compiled, rnd, tmp_path, "b_rnd") in ("fail", "abstain")


def test_relational_grounder_ignores_appearance_candidates(compiled: TaskSpec, tmp_path) -> None:
    """Regression guard: poisoned candidates must change NOTHING relationally."""
    poisoned = compiled.model_copy(deep=True)
    for role in poisoned.roles:
        role.candidates = ["flying spaghetti monster"]
    succ, _ = make_episode("success", seed=5, scene=SCENE_B)
    assert _verdict(poisoned, succ, tmp_path, "b_poisoned") == "pass"
