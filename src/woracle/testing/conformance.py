"""Plugin conformance suite (sklearn ``check_estimator`` pattern, ARCH §7).

Plugin authors run woracle's own compliance checks in THEIR CI::

    from woracle.testing.conformance import channel_checks

    @pytest.mark.parametrize("name,check", channel_checks(MyChannel))
    def test_conformance(name, check):
        check()

Checks execute against a deterministic blobworld fixture, so they need no
GPU, no network, and no user data.
"""

from __future__ import annotations

import atexit
import shutil
import tempfile
from typing import TYPE_CHECKING

import numpy as np

from woracle.contracts import (
    ChannelScore,
    GateSignalValue,
    GroundedRollout,
    RoleBinding,
    TaskSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


def _tmpdir(prefix: str) -> str:
    d = tempfile.mkdtemp(prefix=prefix)
    atexit.register(shutil.rmtree, d, True)
    return d


def _fixture() -> tuple[GroundedRollout, TaskSpec]:
    """A grounded blobworld success episode (built once per process)."""
    global _FIXTURE
    try:
        return _FIXTURE  # type: ignore[name-defined]
    except NameError:
        pass
    import os

    from woracle.io import save_episode
    from woracle.testing.blobworld import blob_spec, make_episode
    from woracle.testing.plugins import BlobColorGrounder

    tmp = _tmpdir("woracle-conformance-")
    frames, _truth = make_episode("success", seed=7)
    ep_dir = os.path.join(tmp, "ep")
    ref = save_episode(ep_dir, "conformance_success", frames, source="blobworld")
    ref.meta["_dir"] = ep_dir
    spec = blob_spec()
    out = os.path.join(tmp, "grounded")
    os.makedirs(out, exist_ok=True)
    grounded = BlobColorGrounder().ground(ref, spec, out)
    _FIXTURE = (grounded, spec)  # type: ignore[name-defined]
    return _FIXTURE  # type: ignore[name-defined]


def _degenerate_fixture() -> tuple[GroundedRollout, TaskSpec]:
    """A bundle with NOTHING usable: all roles unbound, zero artifacts.

    Components must respond with recorded evidence-missing statuses — never an
    exception, never a confident value (the honesty floor for plugins).
    """
    global _DEGENERATE
    try:
        return _DEGENERATE  # type: ignore[name-defined]
    except NameError:
        pass
    from woracle.testing.blobworld import blob_spec

    grounded_ok, _ = _fixture()
    spec = blob_spec()
    grounded = GroundedRollout(
        rollout=grounded_ok.rollout,
        spec_name=spec.name,
        spec_hash=spec.content_hash(),
        bindings=[
            RoleBinding(role=r.name, bound=False, required=r.required, reason="degenerate fixture")
            for r in spec.roles
        ],
        grounder="degenerate@0",
        bundle_dir=_tmpdir("woracle-degenerate-"),
    )
    _DEGENERATE = (grounded, spec)  # type: ignore[name-defined]
    return _DEGENERATE  # type: ignore[name-defined]


def _identity_checks(obj: object, kind: str) -> Iterator[tuple[str, Callable[[], None]]]:
    def check_identity() -> None:
        assert isinstance(getattr(obj, "name", None), str) and obj.name, (  # type: ignore[attr-defined]
            f"{kind} must define a non-empty str `name`"
        )
        assert isinstance(getattr(obj, "version", None), str) and obj.version, (  # type: ignore[attr-defined]
            f"{kind} must define a non-empty str `version` (it participates in cache keys)"
        )

    yield f"{kind}_identity", check_identity


def channel_checks(channel_cls: type) -> list[tuple[str, Callable[[], None]]]:
    ch = channel_cls()
    checks = list(_identity_checks(ch, "channel"))

    def check_caps() -> None:
        caps = getattr(ch, "caps", None)
        assert caps is not None, "channel must declare `caps: ChannelCaps`"
        lo, hi = caps.value_range
        assert lo < hi, "caps.value_range must be a non-empty (lo, hi) interval"

    def check_score_contract() -> None:
        grounded, spec = _fixture()
        score = ch.score(grounded, spec)
        assert isinstance(score, ChannelScore), "score() must return a ChannelScore"
        assert score.channel == ch.name, "ChannelScore.channel must equal channel.name"
        if score.status == "ok":
            assert score.value is not None, "ok scores must carry a value"
            lo, hi = ch.caps.value_range
            assert lo <= score.value <= hi, (
                f"value {score.value} outside declared range [{lo}, {hi}]"
            )
        else:
            assert score.value is None, "non-ok scores must not carry a value"
            assert score.reason, "non-ok scores must explain themselves in `reason`"

    def check_deterministic() -> None:
        grounded, spec = _fixture()
        a, b = ch.score(grounded, spec), ch.score(grounded, spec)
        assert a.model_dump() == b.model_dump(), (
            "score() must be deterministic for identical inputs (cache correctness)"
        )

    def check_does_not_mutate() -> None:
        grounded, spec = _fixture()
        before = grounded.model_dump()
        ch.score(grounded, spec)
        assert grounded.model_dump() == before, "score() must not mutate its inputs"

    def check_degenerate_honesty() -> None:
        grounded, spec = _degenerate_fixture()
        try:
            score = ch.score(grounded, spec)
        except Exception as e:
            raise AssertionError(
                f"channel raised {type(e).__name__} on a degenerate bundle — missing "
                "evidence must be RECORDED (status='evidence_missing'), never raised"
            ) from e
        assert score.status != "ok" or score.value is None or score.confidence == 0.0, (
            "channel returned a confident ok-value with zero usable evidence — honesty violation"
        )
        if score.status != "ok":
            assert score.reason, "non-ok scores must explain themselves"

    checks += [
        ("channel_caps", check_caps),
        ("channel_score_contract", check_score_contract),
        ("channel_deterministic", check_deterministic),
        ("channel_no_mutation", check_does_not_mutate),
        ("channel_degenerate_honesty", check_degenerate_honesty),
    ]
    return checks


def gate_signal_checks(signal_cls: type) -> list[tuple[str, Callable[[], None]]]:
    sig = signal_cls()
    checks = list(_identity_checks(sig, "gate_signal"))

    def check_measure_contract() -> None:
        grounded, _spec = _fixture()
        v = sig.measure(grounded)
        assert isinstance(v, GateSignalValue), "measure() must return a GateSignalValue"
        assert v.name == sig.name
        if v.status == "ok":
            assert v.value is not None and np.isfinite(v.value), "ok signals carry finite values"
        else:
            assert v.value is None and v.reason, (
                "evidence_missing signals carry no value and must give a reason"
            )

    def check_deterministic() -> None:
        grounded, _spec = _fixture()
        a, b = sig.measure(grounded), sig.measure(grounded)
        assert a.model_dump() == b.model_dump(), "measure() must be deterministic"

    def check_degenerate_honesty() -> None:
        grounded, _spec = _degenerate_fixture()
        try:
            v = sig.measure(grounded)
        except Exception as e:
            raise AssertionError(
                f"gate signal raised {type(e).__name__} on a degenerate bundle — "
                "missing evidence must be RECORDED, never raised"
            ) from e
        assert v.status == "evidence_missing", (
            "a degenerate bundle has no evidence; the signal must say so"
        )

    checks += [
        ("gate_signal_measure_contract", check_measure_contract),
        ("gate_signal_deterministic", check_deterministic),
        ("gate_signal_degenerate_honesty", check_degenerate_honesty),
    ]
    return checks


def grounder_checks(grounder_cls: type) -> list[tuple[str, Callable[[], None]]]:
    g = grounder_cls()
    checks = list(_identity_checks(g, "grounder"))

    def check_ground_contract() -> None:
        import os

        from woracle.io import save_episode
        from woracle.testing.blobworld import blob_spec, make_episode

        tmp = _tmpdir("woracle-conf-grounder-")
        frames, _ = make_episode("success", seed=11)
        ep = os.path.join(tmp, "ep")
        ref = save_episode(ep, "conf_g", frames, source="blobworld")
        ref.meta["_dir"] = ep
        spec = blob_spec()
        out = os.path.join(tmp, "out")
        os.makedirs(out, exist_ok=True)
        grounded = g.ground(ref, spec, out)
        assert isinstance(grounded, GroundedRollout)
        produced = {b.role for b in grounded.bindings}
        expected = set(spec.role_names())
        assert produced == expected, (
            f"grounder must emit a RoleBinding for EVERY spec role "
            f"(bound or not); missing: {expected - produced}"
        )
        assert os.path.isfile(os.path.join(out, "grounded.json")), (
            "grounder must persist grounded.json into out_dir"
        )

    def check_params_property() -> None:
        params = getattr(g, "params", None)
        assert isinstance(params, dict), (
            "grounder must expose a `params` dict property — it feeds the ground-"
            "stage cache key; without it two differently-configured instances "
            "silently share cache entries (cache-aliasing failure class)"
        )
        import json

        json.dumps(params, sort_keys=True)  # must be canonical-JSON-able

    checks += [
        ("grounder_ground_contract", check_ground_contract),
        ("grounder_params_property", check_params_property),
    ]
    return checks
