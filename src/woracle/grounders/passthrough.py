"""frames.passthrough — the no-grounding grounder for object-free judging.

Produces a GroundedRollout with NO role bindings. This is the entry point for
the detection-free verdict path (success.demo_match, VLM/reward judges): when
the manipulated object is un-detectable, you don't ground it at all — you judge
the whole-frame evidence against demos. Gate signals that need role artifacts
return evidence_missing (handled by a role-free GatePolicy); frame-level
signals (background_drift) still work.
"""

from __future__ import annotations

import os

from woracle.contracts import GroundedRollout, RoleBinding, RolloutRef, TaskSpec
from woracle.registry import register


@register("grounder", "frames.passthrough")
class PassthroughGrounder:
    name = "frames.passthrough"
    version = "0.1.0"

    @property
    def params(self) -> dict:
        return {}

    def ground(self, rollout: RolloutRef, spec: TaskSpec, out_dir: str) -> GroundedRollout:
        # Emit an unbound binding per spec role (honest: "we did not ground this"),
        # so required-role gates can still report what was skipped.
        bindings = [
            RoleBinding(
                role=r.name,
                bound=False,
                required=r.required,
                reason="frames.passthrough does not ground objects (object-free judging)",
            )
            for r in spec.roles
        ]
        grounded = GroundedRollout(
            rollout=rollout,
            spec_name=spec.name,
            spec_hash=spec.content_hash(),
            bindings=bindings,
            grounder=f"{self.name}@{self.version}",
            bundle_dir=out_dir,
        )
        with open(os.path.join(out_dir, "grounded.json"), "w", encoding="utf-8") as f:
            f.write(grounded.to_json())
        return grounded
