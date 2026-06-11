"""Schema migrations: pure up-migration functions chained at load time.

Every persisted document carries ``schema_version``. On load we walk
``v -> v+1`` migrations until the document reaches the model's current
version. Migrations are pure dict->dict functions registered per model name.

Adding a migration:

    @migration("TaskSpec", from_version=1)
    def _spec_v1_to_v2(data: dict) -> dict:
        data["new_field"] = "default"
        return data

and bump the model's ``schema_version`` default to 2 in the same commit.
JSON-Schema goldens (tests/golden/schemas) fail CI if the model changed
without a version bump.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from woracle.errors import SpecError

Migration = Callable[[dict[str, Any]], dict[str, Any]]

# model name -> {from_version: migration}
_MIGRATIONS: dict[str, dict[int, Migration]] = {}

# Current schema version per persisted model. Single source of truth used by
# both `migrate` and the golden tests.
CURRENT_VERSIONS: dict[str, int] = {
    "TaskSpec": 1,
    "RolloutRef": 1,
    "DemoSet": 1,
    "GroundedRollout": 1,
    "GateReport": 1,
    "GradeCard": 1,
    "Leaderboard": 1,
    "RunManifest": 1,
}


def migration(model: str, from_version: int) -> Callable[[Migration], Migration]:
    def deco(fn: Migration) -> Migration:
        slot = _MIGRATIONS.setdefault(model, {})
        if from_version in slot:
            raise SpecError(f"duplicate migration for {model} v{from_version}")
        slot[from_version] = fn
        return fn

    return deco


def migrate(model: str, data: dict[str, Any]) -> dict[str, Any]:
    """Walk ``data`` up to the current schema version for ``model``."""
    if model not in CURRENT_VERSIONS:
        raise SpecError(f"unknown persisted model '{model}'")
    target = CURRENT_VERSIONS[model]
    version = int(data.get("schema_version", 1))
    if version > target:
        raise SpecError(
            f"{model} document has schema_version={version} but this woracle "
            f"only understands <= {target}; upgrade woracle."
        )
    while version < target:
        step = _MIGRATIONS.get(model, {}).get(version)
        if step is None:
            raise SpecError(f"no migration path for {model} v{version} -> v{version + 1}")
        data = step(dict(data))
        new_version = int(data.get("schema_version", version))
        if new_version <= version:
            # Migration forgot to bump — enforce, don't loop forever.
            version += 1
            data["schema_version"] = version
        else:
            version = new_version
    return data
