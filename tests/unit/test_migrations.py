from __future__ import annotations

import pytest

from woracle.contracts import migrate
from woracle.contracts.migrations import _MIGRATIONS, CURRENT_VERSIONS, migration
from woracle.errors import SpecError


def test_identity_at_current_version(spec) -> None:
    data = spec.model_dump()
    assert migrate("TaskSpec", data) == data


def test_future_version_rejected() -> None:
    with pytest.raises(SpecError, match="upgrade woracle"):
        migrate("TaskSpec", {"schema_version": 999})


def test_unknown_model_rejected() -> None:
    with pytest.raises(SpecError, match="unknown persisted model"):
        migrate("NotAModel", {})


def test_missing_migration_path_is_loud() -> None:
    CURRENT_VERSIONS["_TestDoc"] = 2
    try:
        with pytest.raises(SpecError, match="no migration path"):
            migrate("_TestDoc", {"schema_version": 1})
    finally:
        del CURRENT_VERSIONS["_TestDoc"]


def test_migration_chain_walks_to_current() -> None:
    CURRENT_VERSIONS["_TestDoc2"] = 3
    try:

        @migration("_TestDoc2", from_version=1)
        def _v1(d: dict) -> dict:
            d["a"] = 1
            d["schema_version"] = 2
            return d

        @migration("_TestDoc2", from_version=2)
        def _v2(d: dict) -> dict:
            d["b"] = 2
            d["schema_version"] = 3
            return d

        out = migrate("_TestDoc2", {"schema_version": 1})
        assert out == {"schema_version": 3, "a": 1, "b": 2}
    finally:
        del CURRENT_VERSIONS["_TestDoc2"]
        _MIGRATIONS.pop("_TestDoc2", None)


def test_duplicate_migration_rejected() -> None:
    CURRENT_VERSIONS["_TestDoc3"] = 2
    try:

        @migration("_TestDoc3", from_version=1)
        def _v1(d: dict) -> dict:
            return d

        with pytest.raises(SpecError, match="duplicate migration"):

            @migration("_TestDoc3", from_version=1)
            def _v1b(d: dict) -> dict:
                return d

    finally:
        del CURRENT_VERSIONS["_TestDoc3"]
        _MIGRATIONS.pop("_TestDoc3", None)
