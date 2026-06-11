"""Woracle contracts — every stage boundary is one of these models.

Pure data layer: numpy + pydantic only. Nothing here may import model
libraries, plugins, or pipeline code.
"""

from woracle.contracts.base import (
    ArtifactRef,
    Provenance,
    VersionedModel,
    WoracleModel,
    canonical_json,
    digest_array,
    digest_bytes,
    digest_file,
    digest_json,
)
from woracle.contracts.binding import GroundedRollout, RoleBinding
from woracle.contracts.gate import (
    GatePolicy,
    GateReport,
    GateSignalValue,
    GateThreshold,
    GateVerdict,
    SignalStatus,
)
from woracle.contracts.manifest import RunManifest, StageRecord
from woracle.contracts.migrations import CURRENT_VERSIONS, migrate, migration
from woracle.contracts.rollout import DemoSet, RolloutRef
from woracle.contracts.score import (
    ChannelCaps,
    ChannelScore,
    ChannelStatus,
    GradeCard,
    Leaderboard,
    PolicySummary,
    SuccessReport,
    SuccessVerdict,
)
from woracle.contracts.spec import (
    FailureMode,
    MotionSig,
    Phase,
    Predicate,
    PredicateKind,
    Role,
    SelfTestReport,
    SpecProvenance,
    TaskSpec,
    load_spec,
)

__all__ = [
    "CURRENT_VERSIONS",
    "ArtifactRef",
    "ChannelCaps",
    "ChannelScore",
    "ChannelStatus",
    "DemoSet",
    "FailureMode",
    "GatePolicy",
    "GateReport",
    "GateSignalValue",
    "GateThreshold",
    "GateVerdict",
    "GradeCard",
    "GroundedRollout",
    "Leaderboard",
    "MotionSig",
    "Phase",
    "PolicySummary",
    "Predicate",
    "PredicateKind",
    "Provenance",
    "Role",
    "RoleBinding",
    "RolloutRef",
    "RunManifest",
    "SelfTestReport",
    "SignalStatus",
    "SpecProvenance",
    "StageRecord",
    "SuccessReport",
    "SuccessVerdict",
    "TaskSpec",
    "VersionedModel",
    "WoracleModel",
    "canonical_json",
    "digest_array",
    "digest_bytes",
    "digest_file",
    "digest_json",
    "load_spec",
    "migrate",
    "migration",
]
