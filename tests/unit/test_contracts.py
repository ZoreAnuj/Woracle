from __future__ import annotations

import numpy as np
import pydantic
import pytest

from woracle.contracts import (
    ArtifactRef,
    GateSignalValue,
    TaskSpec,
    canonical_json,
    digest_array,
    digest_json,
)
from woracle.errors import SpecError


def test_spec_yaml_roundtrip(spec: TaskSpec) -> None:
    text = spec.to_yaml()
    back = TaskSpec.from_yaml(text)
    assert back == spec
    assert back.content_hash() == spec.content_hash()


def test_content_hash_excludes_provenance(spec: TaskSpec) -> None:
    h0 = spec.content_hash()
    mutated = spec.model_copy(deep=True)
    mutated.spec_provenance.compiler = "something else"
    assert mutated.content_hash() == h0
    semantic = spec.model_copy(deep=True)
    semantic.prompt = "a different task"
    assert semantic.content_hash() != h0


def test_extra_keys_forbidden() -> None:
    with pytest.raises(pydantic.ValidationError):
        ArtifactRef.model_validate({"path": "x.npz", "nonsense": 1})


def test_spec_role_lookup(spec: TaskSpec) -> None:
    assert spec.role("carried_object").motion == "co_moves_with_effector"
    with pytest.raises(SpecError):
        spec.role("nope")


def test_canonical_json_is_order_invariant() -> None:
    a = {"b": 1, "a": [1, 2], "c": {"y": 2, "x": 1}}
    b = {"c": {"x": 1, "y": 2}, "a": [1, 2], "b": 1}
    assert canonical_json(a) == canonical_json(b)
    assert digest_json(a) == digest_json(b)


def test_canonical_json_rejects_nan() -> None:
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_digest_array_sensitive_to_dtype_and_shape() -> None:
    x = np.zeros((4, 4), np.float32)
    assert digest_array(x) != digest_array(x.astype(np.float64))
    assert digest_array(x) != digest_array(x.reshape(2, 8))
    assert digest_array(x) == digest_array(np.zeros((4, 4), np.float32))


@pytest.mark.parametrize("bad", ["\x85", "a\x00b", "line\u2028sep", "\x1b[31m"])
def test_yaml_hostile_text_rejected_in_specs(bad: str, spec: TaskSpec) -> None:
    """C0/C1 controls + U+2028/29 break YAML round-trips -> contract rejects them."""
    with pytest.raises(pydantic.ValidationError, match="control character"):
        TaskSpec(name="t", prompt=bad, roles=spec.roles)


def test_newline_and_tab_allowed_in_spec_text(spec: TaskSpec) -> None:
    s = TaskSpec(name="t", prompt="line one\nline two\tend", roles=spec.roles)
    assert TaskSpec.from_yaml(s.to_yaml()) == s


def test_signal_value_honesty_shape() -> None:
    ok = GateSignalValue(name="s", value=0.7)
    missing = GateSignalValue(name="s", status="evidence_missing", reason="no track")
    assert ok.status == "ok" and ok.value == 0.7
    assert missing.value is None and missing.reason
