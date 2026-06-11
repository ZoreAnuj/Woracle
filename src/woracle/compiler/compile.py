"""S1: compile demos + prompt -> TaskSpec, with self-test or REFUSE.

The product's core loop (proposal §4): UNDERSTAND (entities, roles, phases,
fitted tolerances) -> EMIT spec -> MINT negatives -> SELF-TEST (held-out-style
check: every demo must PASS, every minted negative must FAIL) -> repair
tolerances by grid search -> accept, or REFUSE with the evidence.

v1 honesty box (stated, not hidden): perception is the model-free entity
induction (color clustering + motion) — adequate for blob-class scenes and as
the relational layer for real ones; phases/predicates are fitted from
RELATIONS (distances, motion), so the emitted spec is appearance-free except
for the open-vocab candidate HINTS, which carry color words only as hints.
"""

from __future__ import annotations

import numpy as np

from woracle._version import __version__
from woracle.compiler.entities import classify_relational, induce_entities
from woracle.compiler.negatives import mint_negatives
from woracle.compiler.selftest import SelfTestOutcome, run_selftest
from woracle.contracts import (
    Phase,
    Predicate,
    Role,
    SelfTestReport,
    SpecProvenance,
    TaskSpec,
    digest_array,
)
from woracle.errors import SpecError


def _fit_tolerances(demos_entities: list[dict]) -> dict[str, float]:
    """Fit success tolerances from the demos' final-state RELATIONS."""
    final_dists_rel = []
    final_speeds = []
    for ents in demos_entities:
        carried, receptacle = ents["carried_object"], ents["receptacle"]
        if carried is None or receptacle is None:
            continue
        obs_c = np.isfinite(carried.track[:, 0])
        obs_r = np.isfinite(receptacle.track[:, 0])
        if not obs_c.any() or not obs_r.any():
            continue
        T = len(carried.track)
        w = np.arange(T) >= int(T * 0.85)
        cw = carried.track[np.flatnonzero(obs_c & w)]
        rp = receptacle.track[np.flatnonzero(obs_r)].mean(axis=0)
        if len(cw) == 0:
            continue
        d = float(np.linalg.norm(cw.mean(axis=0) - rp))
        size = float(np.sqrt(max(receptacle.stats["mean_area"], 1.0)))
        final_dists_rel.append(d / size)
        if len(cw) > 1:
            final_speeds.append(float(np.linalg.norm(np.diff(cw, axis=0), axis=1).max()))
    if not final_dists_rel:
        raise SpecError("could not fit tolerances: no demo had carried+receptacle relations")
    return {
        # generous initial fit; the self-test grid search tightens/widens it
        "tol_rel": float(np.percentile(final_dists_rel, 95) * 1.5),
        "stationary_tol": float(max(np.percentile(final_speeds, 95) * 2.0, 1.0))
        if final_speeds
        else 1.5,
    }


def _phases_from_relations() -> list[Phase]:
    """The transport-task phase template, fitted predicates filled at emit."""
    return [
        Phase(
            name="approach",
            order=0,
            description="carried object approaches the receptacle",
            active=[Predicate(kind="approaching", subject="carried_object", object="receptacle")],
        ),
        Phase(
            name="arrive",
            order=1,
            description="carried object reaches the receptacle region",
            active=[
                Predicate(
                    kind="co_located",
                    subject="carried_object",
                    object="receptacle",
                    params={"tol_rel": 2.5},
                )
            ],
        ),
        Phase(
            name="settle",
            order=2,
            description="carried object rests at the receptacle",
            active=[Predicate(kind="stationary", subject="carried_object")],
        ),
    ]


def compile_spec(
    demo_frames: list[np.ndarray],
    prompt: str,
    *,
    name: str = "compiled-task",
    grounder: str = "relational.motion",
    seed: int = 0,
    max_rounds: int = 6,
) -> TaskSpec:
    """Compile a TaskSpec from demo episodes; raises SpecError (REFUSE) when
    the compiled spec cannot separate demos from minted negatives."""
    if len(demo_frames) < 2:
        raise SpecError("need at least 2 demos to compile (and to fit + sanity-check)")

    # 1) UNDERSTAND: induce entities per demo, map to relational roles
    demos_entities = []
    for i, frames in enumerate(demo_frames):
        ents = induce_entities(frames, seed=seed + i)
        demos_entities.append(classify_relational(ents))
    n_ok = sum(
        1 for e in demos_entities if e["carried_object"] is not None and e["receptacle"] is not None
    )
    if n_ok < max(2, len(demo_frames) // 2):
        raise SpecError(
            f"REFUSE: relational role induction succeeded on only {n_ok}/{len(demo_frames)} "
            "demos (need a mover that settles at a static entity) — task structure "
            "not recoverable from these demos"
        )

    # 2) candidates: color-word HINTS from induced entities (hints, not truth)
    def hint(role: str) -> list[str]:
        names = [e[role].name for e in demos_entities if e[role] is not None]  # type: ignore[union-attr]
        if not names:
            return []
        top = max(set(names), key=names.count)
        return [f"{top} object"]

    tols = _fit_tolerances(demos_entities)
    spec = TaskSpec(
        name=name,
        prompt=prompt,
        roles=[
            Role(
                name="carried_object",
                definition="the mover that ends settled at the static receptacle",
                motion="co_moves_with_effector",
                candidates=hint("carried_object"),
                required=True,
            ),
            Role(
                name="receptacle",
                definition="the static entity the carried object settles at",
                motion="static",
                candidates=hint("receptacle"),
                required=True,
            ),
            Role(
                name="gripper",
                definition="the effector that transports the carried object then separates",
                motion="actuated",
                candidates=hint("effector"),
                required=False,
            ),
        ],
        phases=_phases_from_relations(),
        success=[
            Predicate(
                kind="co_located",
                subject="carried_object",
                object="receptacle",
                params={"tol_rel": tols["tol_rel"]},
            ),
            Predicate(
                kind="stationary", subject="carried_object", params={"tol": tols["stationary_tol"]}
            ),
        ],
        success_sustain_frames=5,
        version=1,
    )

    # 3) MINT negatives + 4) SELF-TEST with tolerance repair (grid search)
    negatives = [neg for frames in demo_frames[:2] for neg in mint_negatives(frames)]
    outcome: SelfTestOutcome = run_selftest(
        spec, demo_frames, negatives, grounder=grounder, max_rounds=max_rounds
    )
    spec = outcome.spec  # possibly tolerance-repaired
    spec.spec_provenance = SpecProvenance(
        demo_digests=[digest_array(f) for f in demo_frames],
        compiler=f"woracle.compile v{__version__} (relational v1)",
        self_test=SelfTestReport(
            ran=True,
            demos_passed=outcome.demos_passed,
            demos_total=len(demo_frames),
            negatives_failed=outcome.negatives_failed,
            negatives_total=len(negatives),
            accepted=outcome.accepted,
            notes=outcome.notes,
        ),
    )
    if not outcome.accepted:
        raise SpecError(
            "REFUSE: compiled spec cannot separate demos from minted negatives "
            f"(demos passed {outcome.demos_passed}/{len(demo_frames)}, negatives "
            f"failed {outcome.negatives_failed}/{len(negatives)}). {outcome.notes} "
            "— refusing to emit an oracle that cannot tell success from failure."
        )
    return spec
