"""Drop-missing verdict ensemble — the generalizable-grounding fix.

A verdict-eligible channel that cannot judge (e.g. object-grounded predicates
when the manipulated object is un-detectable) must be DROPPED, not veto the
whole verdict — so a detection-free channel can still decide. Abstain only if
NO channel could judge, or if those that could DISAGREE.
"""

from __future__ import annotations

from woracle.contracts import ChannelScore
from woracle.pipeline.run import _success_from_channels


class _Chan:
    def __init__(self, name, eligible):
        self.name = name
        self.version = "0"

        class _Caps:
            verdict_eligible = eligible

        self.caps = _Caps()


PRED = _Chan("success.predicates", True)
DEMO = _Chan("success.demo_match", True)
RANK = _Chan("progress.gvl", False)


def _sc(name, status="ok", value=None, reason=""):
    return ChannelScore(channel=name, status=status, value=value, reason=reason)


def test_blind_predicate_dropped_demo_decides() -> None:
    """THE fix: predicates can't ground the object -> dropped; demo_match decides."""
    out = _success_from_channels(
        [PRED, DEMO],
        [
            _sc("success.predicates", "evidence_missing", reason="tip unobserved"),
            _sc("success.demo_match", value=0.8),
        ],
        0.5,
    )
    assert out.verdict == "pass"
    assert any("dropped" in r for r in out.reasons)

    out_fail = _success_from_channels(
        [PRED, DEMO],
        [
            _sc("success.predicates", "evidence_missing", reason="tip unobserved"),
            _sc("success.demo_match", value=0.2),
        ],
        0.5,
    )
    assert out_fail.verdict == "fail"


def test_conflict_abstains_never_guesses() -> None:
    out = _success_from_channels(
        [PRED, DEMO],
        [_sc("success.predicates", value=1.0), _sc("success.demo_match", value=0.1)],
        0.5,
    )
    assert out.verdict == "abstain"
    assert any("conflict" in r for r in out.reasons)


def test_all_missing_abstains() -> None:
    out = _success_from_channels(
        [PRED, DEMO],
        [
            _sc("success.predicates", "evidence_missing", reason="a"),
            _sc("success.demo_match", "evidence_missing", reason="b"),
        ],
        0.5,
    )
    assert out.verdict == "abstain"


def test_single_channel_unchanged() -> None:
    # one verdict channel -> behaves exactly as before (no ensemble effect)
    assert (
        _success_from_channels([PRED], [_sc("success.predicates", value=1.0)], 0.5).verdict
        == "pass"
    )
    assert (
        _success_from_channels([PRED], [_sc("success.predicates", value=0.0)], 0.5).verdict
        == "fail"
    )
    assert (
        _success_from_channels(
            [PRED], [_sc("success.predicates", "evidence_missing", reason="x")], 0.5
        ).verdict
        == "abstain"
    )


def test_rank_only_channels_never_vote() -> None:
    out = _success_from_channels(
        [RANK, DEMO], [_sc("progress.gvl", value=0.0), _sc("success.demo_match", value=0.9)], 0.5
    )
    assert out.verdict == "pass"  # rank-only 0.0 cannot drag the verdict to fail
