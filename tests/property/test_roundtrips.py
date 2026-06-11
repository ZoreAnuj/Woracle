"""Property tests: serialization round-trips hold for arbitrary valid content."""

from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from woracle.contracts import (
    GateSignalValue,
    Predicate,
    Role,
    TaskSpec,
)

_name = st.text(
    alphabet=st.characters(whitelist_categories=("Ll", "Lu", "Nd"), whitelist_characters="_-"),
    min_size=1,
    max_size=24,
)
# Free text: anything except what the Text contract rejects (C0/C1 controls,
# U+2028/29 — YAML-hostile; the validator has its own unit test).
_text = lambda **kw: st.text(  # noqa: E731
    alphabet=st.characters(blacklist_categories=("Cc", "Cs"), blacklist_characters="\u2028\u2029"),
    **kw,
)
_params = st.dictionaries(
    _name, st.floats(min_value=-1e6, max_value=1e6, allow_nan=False), max_size=3
)

_role = st.builds(
    Role,
    name=_name,
    definition=_text(max_size=60),
    motion=st.sampled_from(["co_moves_with_effector", "static", "actuated", "free"]),
    candidates=st.lists(_text(min_size=1, max_size=20), max_size=3),
    required=st.booleans(),
)

_unary = st.builds(
    Predicate,
    kind=st.sampled_from(["stationary", "present"]),
    subject=_name,
    params=_params,
)
_binary = st.builds(
    Predicate,
    kind=st.sampled_from(["contained", "co_located", "approaching", "separated"]),
    subject=_name,
    object=_name,
    params=_params,
)
_pred = st.one_of(_unary, _binary)

_spec = st.builds(
    TaskSpec,
    name=_name,
    prompt=_text(min_size=1, max_size=80),
    roles=st.lists(_role, min_size=1, max_size=4),
    success=st.lists(_pred, max_size=4),
    success_sustain_frames=st.integers(min_value=1, max_value=30),
    version=st.integers(min_value=1, max_value=99),
)


@settings(max_examples=60, deadline=None)
@given(_spec)
def test_spec_yaml_roundtrip(spec: TaskSpec) -> None:
    assert TaskSpec.from_yaml(spec.to_yaml()) == spec


@settings(max_examples=60, deadline=None)
@given(_spec)
def test_spec_json_roundtrip_preserves_hash(spec: TaskSpec) -> None:
    back = TaskSpec.model_validate_json(spec.model_dump_json())
    assert back.content_hash() == spec.content_hash()


@settings(max_examples=40, deadline=None)
@given(
    st.builds(
        GateSignalValue,
        name=_name,
        value=st.floats(min_value=0, max_value=1, allow_nan=False),
        reason=st.text(max_size=40),
    )
)
def test_signal_roundtrip(sig: GateSignalValue) -> None:
    assert GateSignalValue.model_validate_json(sig.model_dump_json()) == sig
