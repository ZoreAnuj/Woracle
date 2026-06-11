"""First-party components must pass the same conformance suite we publish
for third-party plugin authors (sklearn check_estimator discipline)."""

from __future__ import annotations

import pytest

from woracle.testing.conformance import (
    channel_checks,
    gate_signal_checks,
    grounder_checks,
)
from woracle.testing.plugins import (
    BindingHealthSignal,
    BlobColorGrounder,
    GoalDistanceProgress,
    MotionSanitySignal,
    PermanenceSignal,
    PredicateSuccessChannel,
)

_ALL = (
    [(f"grounder:{c[0]}", c[1]) for c in grounder_checks(BlobColorGrounder)]
    + [(f"sig.binding:{c[0]}", c[1]) for c in gate_signal_checks(BindingHealthSignal)]
    + [(f"sig.permanence:{c[0]}", c[1]) for c in gate_signal_checks(PermanenceSignal)]
    + [(f"sig.motion:{c[0]}", c[1]) for c in gate_signal_checks(MotionSanitySignal)]
    + [(f"ch.progress:{c[0]}", c[1]) for c in channel_checks(GoalDistanceProgress)]
    + [(f"ch.predicates:{c[0]}", c[1]) for c in channel_checks(PredicateSuccessChannel)]
)


@pytest.mark.parametrize("name,check", _ALL, ids=[n for n, _ in _ALL])
def test_conformance(name: str, check) -> None:
    check()
