"""TaskSpec — the portable, scene-invariant task contract (THE product).

A spec stores the TASK, never the scene: roles are defined relationally
("the thing that co-moves with the end-effector"), success is a conjunction of
predicates over roles, and phases describe what should happen in what order.

P0 ships the schema + (de)serialization + hashing. The compiler that *emits*
specs from demos lands in P4; until then specs are hand-written YAML (which the
schema is explicitly designed to keep human-readable and human-editable —
autonomy is the ceiling, review is the floor).
"""

from __future__ import annotations

from typing import Literal

import yaml
from pydantic import Field

from woracle.contracts.base import (
    Provenance,
    Text,
    VersionedModel,
    WoracleModel,
    digest_json,
)

MotionSig = Literal["co_moves_with_effector", "static", "actuated", "free"]

PredicateKind = Literal[
    "contained",  # subject centroid inside object region
    "co_located",  # subject within `tol` of object
    "stationary",  # subject speed below `tol` for `frames`
    "approaching",  # distance(subject, object) decreasing
    "separated",  # subject NOT within `tol` of object
    "present",  # subject visible in frame
]


class Role(WoracleModel):
    """A relational participant in the task — never an appearance."""

    name: Text  # e.g. "carried_object"
    definition: Text  # human-readable relational definition
    motion: MotionSig = "free"
    candidates: list[Text] = Field(default_factory=list)  # open-vocab binding hints
    required: bool = True  # binding failure on a required role
    #                                          # => rollout is ungradeable


class Predicate(WoracleModel):
    kind: PredicateKind
    subject: Text  # role name
    object: Text | None = None  # role name (None for unary kinds)
    params: dict[str, float] = Field(default_factory=dict)

    def describe(self) -> str:
        if self.object:
            return f"{self.kind}({self.subject}, {self.object})"
        return f"{self.kind}({self.subject})"


class Phase(WoracleModel):
    name: Text
    description: Text = ""
    order: int = 0
    # Evidence expected during this phase (used by progress/phase channels).
    active: list[Predicate] = Field(default_factory=list)


class FailureMode(WoracleModel):
    name: Text  # e.g. "dropped", "missed"
    description: Text = ""
    signature: list[Predicate] = Field(default_factory=list)


class SelfTestReport(WoracleModel):
    """Result of the compile-time self-test (P4): demos must pass, minted
    negatives must fail. Recorded so every spec carries its own validation."""

    ran: bool = False
    demos_passed: int = 0
    demos_total: int = 0
    negatives_failed: int = 0
    negatives_total: int = 0
    accepted: bool = False
    notes: Text = ""


class SpecProvenance(WoracleModel):
    demo_digests: list[str] = Field(default_factory=list)
    compiler: str = "handwritten"  # "handwritten" | "woracle.compile vX"
    self_test: SelfTestReport = Field(default_factory=SelfTestReport)
    provenance: Provenance = Field(default_factory=Provenance)


class TaskSpec(VersionedModel):
    name: Text
    prompt: Text
    roles: list[Role]
    phases: list[Phase] = Field(default_factory=list)
    success: list[Predicate] = Field(default_factory=list)  # conjunction
    failure_modes: list[FailureMode] = Field(default_factory=list)
    # Sustained-success requirement: success predicates must hold for this many
    # consecutive frames at the end of the rollout (guards single-frame flukes).
    success_sustain_frames: int = 3
    tl_formula: str = ""  # compiled temporal-logic form (P3/P4)
    version: int = 1  # bumped on ANY semantic change
    spec_provenance: SpecProvenance = Field(default_factory=SpecProvenance)

    # -- helpers ------------------------------------------------------------
    def role(self, name: str) -> Role:
        for r in self.roles:
            if r.name == name:
                return r
        from woracle.errors import SpecError

        raise SpecError(f"spec '{self.name}' has no role '{name}'")

    def role_names(self) -> list[str]:
        return [r.name for r in self.roles]

    def content_hash(self) -> str:
        """Hash of the semantic content (excludes provenance)."""
        data = self.model_dump(exclude={"spec_provenance"})
        return digest_json(data)

    # -- (de)serialization ----------------------------------------------------
    def to_yaml(self) -> str:
        return yaml.safe_dump(self.model_dump(), sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, text: str) -> TaskSpec:
        from woracle.contracts.migrations import migrate
        from woracle.errors import SpecError

        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:  # pragma: no cover - exercised via SpecError path
            raise SpecError(f"spec is not valid YAML: {e}") from e
        if not isinstance(data, dict):
            raise SpecError("spec YAML must be a mapping")
        data = migrate("TaskSpec", data)
        return cls.model_validate(data)


def load_spec(path: str) -> TaskSpec:
    """Load a TaskSpec from a YAML file (the public entry point)."""
    with open(path, encoding="utf-8") as f:
        return TaskSpec.from_yaml(f.read())
