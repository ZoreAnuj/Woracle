"""RunManifest — provenance for every pipeline run ("MLflow without the server").

Written into each run directory so any number in any GradeCard is auditable:
which package version, which components at which versions, which inputs (by
digest), which cache keys, which seeds.
"""

from __future__ import annotations

from pydantic import Field

from woracle.contracts.base import VersionedModel


class StageRecord(VersionedModel):
    schema_version: int = 1
    stage: str  # "ground" | "gate" | "grade" | ...
    component: str = ""  # name@version
    cache_key: str = ""
    cache_hit: bool = False
    wall_s: float = 0.0


class RunManifest(VersionedModel):
    run_id: str
    created_at: str = ""  # ISO-8601
    package_version: str = ""
    git_sha: str = ""
    python: str = ""
    platform: str = ""
    seeds: dict[str, int] = Field(default_factory=dict)
    config: dict = Field(default_factory=dict)  # full config snapshot
    inputs: dict[str, str] = Field(default_factory=dict)  # logical name -> digest
    stages: list[StageRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
