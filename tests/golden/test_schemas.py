"""JSON-Schema goldens: persisted models cannot drift without a visible diff.

Changing a contract model changes its JSON Schema; this test fails until the
golden is regenerated (WORACLE_REGEN_SCHEMAS=1) AND the schema_version /
migration story is updated in the same commit. That friction is the feature.
"""

from __future__ import annotations

import json
import os

import pytest

from woracle.contracts import (
    DemoSet,
    GateReport,
    GradeCard,
    GroundedRollout,
    Leaderboard,
    RolloutRef,
    RunManifest,
    TaskSpec,
)

GOLDEN_DIR = os.path.join(os.path.dirname(__file__), "schemas")
PERSISTED = [
    TaskSpec,
    RolloutRef,
    DemoSet,
    GroundedRollout,
    GateReport,
    GradeCard,
    Leaderboard,
    RunManifest,
]


@pytest.mark.parametrize("model", PERSISTED, ids=lambda m: m.__name__)
def test_schema_golden(model) -> None:
    schema = model.model_json_schema()
    path = os.path.join(GOLDEN_DIR, f"{model.__name__}.json")
    if os.environ.get("WORACLE_REGEN_SCHEMAS") == "1":
        os.makedirs(GOLDEN_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(schema, f, indent=2, sort_keys=True)
            f.write("\n")
    if not os.path.isfile(path):
        pytest.fail(
            f"no golden schema for {model.__name__}; run with WORACLE_REGEN_SCHEMAS=1 "
            "and commit the result"
        )
    with open(path, encoding="utf-8") as f:
        golden = json.load(f)
    assert schema == golden, (
        f"{model.__name__} JSON schema drifted from golden. If intentional: bump "
        "schema_version, add a migration, regenerate with WORACLE_REGEN_SCHEMAS=1, "
        "and commit all three together."
    )
