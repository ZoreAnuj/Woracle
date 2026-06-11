"""Compile-time self-test: demos must PASS, minted negatives must FAIL.

The VLM-CaR acceptance gate (expert-pass / random-fail), upgraded with a
tolerance repair loop: a 1-D grid search over the success co-location
tolerance picks the value that separates demos from negatives, if any does.
No separation -> the compiler REFUSES (honest failure beats silent garbage).

Verdicts here run the REAL grading path (grounder -> PredicateSuccessChannel),
not a shortcut — the self-test certifies the spec under the same machinery
that will grade rollouts.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass

import numpy as np

from woracle.contracts import TaskSpec
from woracle.io import load_rollout, save_episode
from woracle.registry import get as reg_get


@dataclass
class SelfTestOutcome:
    spec: TaskSpec
    accepted: bool
    demos_passed: int
    negatives_failed: int
    notes: str = ""


def _verdict(spec: TaskSpec, frames: np.ndarray, grounder, workdir: str, tag: str) -> bool | None:
    """True=pass, False=fail, None=unevaluable (counts as fail for demos and
    as 'failed' for negatives — an oracle that can't read its own demos is
    not accepted, and an unreadable negative is at least not a false pass)."""
    from woracle.testing.plugins import PredicateSuccessChannel

    ep = os.path.join(workdir, f"ep_{tag}")
    save_episode(ep, f"st_{tag}", frames, source="selftest")
    ref = load_rollout(ep)
    gdir = os.path.join(workdir, f"g_{tag}")
    os.makedirs(gdir, exist_ok=True)
    grounded = grounder.ground(ref, spec, gdir)
    score = PredicateSuccessChannel().score(grounded, spec)
    if score.status != "ok" or score.value is None:
        return None
    return bool(score.value >= 0.5)


def run_selftest(
    spec: TaskSpec,
    demos: list[np.ndarray],
    negatives: list[tuple[str, np.ndarray]],
    *,
    grounder: str = "relational.motion",
    max_rounds: int = 6,
) -> SelfTestOutcome:
    g = reg_get("grounder", grounder)()
    base_tol = None
    for pred in spec.success:
        if pred.kind == "co_located" and "tol_rel" in pred.params:
            base_tol = float(pred.params["tol_rel"])
    grid = [1.0] if base_tol is None else [0.6, 0.8, 1.0, 1.3, 1.7, 2.2][:max_rounds]

    best = None  # (demos_passed, negatives_failed, tol_mult, notes)
    with tempfile.TemporaryDirectory(prefix="woracle-selftest-") as workdir:
        # Ground once per episode per tolerance? Grounding is tolerance-
        # independent — ground ONCE, re-evaluate predicates per tolerance.
        grounded_demos = []
        for i, frames in enumerate(demos):
            grounded_demos.append(("demo", i, frames))
        all_eps = grounded_demos + [("neg", kind, fr) for kind, fr in negatives]

        verdict_cache: dict[tuple[str, str, float], bool | None] = {}
        for mult in grid:
            trial = spec.model_copy(deep=True)
            for pred in trial.success:
                if pred.kind == "co_located" and "tol_rel" in pred.params and base_tol:
                    pred.params["tol_rel"] = base_tol * mult
            dp = nf = 0
            for kind, tag, frames in all_eps:
                key = (kind, str(tag), mult)
                if key not in verdict_cache:
                    verdict_cache[key] = _verdict(trial, frames, g, workdir, f"{kind}{tag}_{mult}")
                v = verdict_cache[key]
                if kind == "demo":
                    dp += 1 if v is True else 0
                else:
                    nf += 1 if (v is False or v is None) else 0
            cand = (dp, nf, mult)
            if best is None or (dp, nf) > (best[0], best[1]):
                best = cand
            if dp == len(demos) and nf == len(negatives):
                final = spec.model_copy(deep=True)
                for pred in final.success:
                    if pred.kind == "co_located" and "tol_rel" in pred.params and base_tol:
                        pred.params["tol_rel"] = base_tol * mult
                return SelfTestOutcome(
                    spec=final,
                    accepted=True,
                    demos_passed=dp,
                    negatives_failed=nf,
                    notes=f"separated at tol_rel x{mult}",
                )

    assert best is not None
    dp, nf, mult = best
    return SelfTestOutcome(
        spec=spec,
        accepted=False,
        demos_passed=dp,
        negatives_failed=nf,
        notes=f"best separation at tol_rel x{mult}: {dp} demos passed, {nf} negatives failed",
    )
