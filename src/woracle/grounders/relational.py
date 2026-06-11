"""Relational grounder: bind roles by HOW THINGS MOVE, not how they look.

The other half of compile-once/bind-later: a spec whose roles are relational
("the mover that settles at a static entity") can be bound in a scene with
completely different appearance — no candidates, no detector, no models.
Scene B (yellow square -> blue cup, mirrored) binds with the same spec that
was compiled on Scene A.

Honesty notes baked in:
* two movers that NEVER separate are relationally ambiguous — both bindings
  are emitted with quality 0.5 and an explicit reason (the gate degrades);
  in successful episodes the effector separates, resolving the ambiguity.
* perception floor is the model-free entity induction; real-world scenes
  should pair this with the open-vocab grounder and cross-check (P4+ roadmap).
"""

from __future__ import annotations

import os

import numpy as np

from woracle.compiler.entities import Entity, classify_relational, induce_entities
from woracle.contracts import (
    ArtifactRef,
    GroundedRollout,
    RoleBinding,
    RolloutRef,
    TaskSpec,
    digest_file,
)
from woracle.io import load_frames
from woracle.registry import register

# spec role name -> relational slot produced by classify_relational
_SLOT = {"carried_object": "carried_object", "receptacle": "receptacle", "gripper": "effector"}


@register("grounder", "relational.motion")
class RelationalMotionGrounder:
    name = "relational.motion"
    version = "0.1.0"

    def __init__(self, k: int = 5, seed: int = 0, moving_frac: float = 0.08) -> None:
        self.k = int(k)
        self.seed = int(seed)
        self.moving_frac = float(moving_frac)

    @property
    def params(self) -> dict:
        return {"k": self.k, "seed": self.seed, "moving_frac": self.moving_frac}

    def _write_entity(
        self, out_dir: str, role: str, ent: Entity, T: int, H: int, W: int
    ) -> RoleBinding:
        track = ent.track.astype(np.float32)
        vis = (ent.area / max(float(ent.area.max()), 1.0)).astype(np.float32)
        # mask stack at a coarse stride (occupancy for region predicates)
        stride = max(1, T // 24)
        idxs = np.arange(0, T, stride)
        masks = np.zeros((len(idxs), H, W), np.uint8)
        for n, t in enumerate(idxs):
            masks[n] = np.asarray(ent.mask_fn(int(t)), dtype=np.uint8)  # type: ignore[misc]
        tpath = os.path.join(out_dir, f"{role}.track.npz")
        mpath = os.path.join(out_dir, f"{role}.mask.npz")
        vpath = os.path.join(out_dir, f"{role}.vis.npz")
        np.savez_compressed(tpath, track=track)
        np.savez_compressed(mpath, mask=masks, sample_idxs=idxs)
        np.savez_compressed(vpath, vis=vis)
        return RoleBinding(
            role=role,
            bound=True,
            quality=float(ent.stats["persistence"]),
            reason=f"relational bind: {ent.name} entity, range {ent.stats['range_px']:.0f}px",
            tracks=ArtifactRef(
                path=os.path.basename(tpath), sha256=digest_file(tpath), kind="track.npz"
            ),
            masks=ArtifactRef(
                path=os.path.basename(mpath), sha256=digest_file(mpath), kind="mask.npz"
            ),
            visibility=ArtifactRef(
                path=os.path.basename(vpath), sha256=digest_file(vpath), kind="vis.npz"
            ),
        )

    def ground(self, rollout: RolloutRef, spec: TaskSpec, out_dir: str) -> GroundedRollout:
        frames = load_frames(rollout)
        T, H, W = frames.shape[:3]
        entities = induce_entities(frames, k=self.k, seed=self.seed)
        slots = classify_relational(entities, moving_frac=self.moving_frac)

        # Ambiguity detection: carried & effector that never separate
        ambiguous = False
        c, e = slots.get("carried_object"), slots.get("effector")
        if c is not None and e is not None:
            oc, oe = np.isfinite(c.track[:, 0]), np.isfinite(e.track[:, 0])
            tail = np.arange(T) >= int(T * 0.8)
            ic, ie = np.flatnonzero(oc & tail), np.flatnonzero(oe & tail)
            if len(ic) and len(ie):
                sep = float(np.linalg.norm(c.track[ic].mean(axis=0) - e.track[ie].mean(axis=0)))
                ambiguous = sep < 16.0  # never separated -> roles interchangeable

        bindings: list[RoleBinding] = []
        for role in spec.roles:
            ent = slots.get(_SLOT.get(role.name, ""))
            if ent is None:
                bindings.append(
                    RoleBinding(
                        role=role.name,
                        bound=False,
                        required=role.required,
                        reason="no entity matches this role's relational signature",
                    )
                )
                continue
            b = self._write_entity(out_dir, role.name, ent, T, H, W)
            b.required = role.required
            if ambiguous and role.name in ("carried_object", "gripper"):
                b.quality = min(b.quality, 0.5)
                b.reason = (
                    "AMBIGUOUS: two movers never separate — carried/effector "
                    "assignment is by final-window stability only; " + b.reason
                )
            bindings.append(b)

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
