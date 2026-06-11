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


def _ground_once(spec: TaskSpec, frames: np.ndarray, grounder, workdir: str, tag: str):
    """Ground one episode through the REAL grading path (no shortcuts).

    Verdict semantics downstream: unevaluable counts as not-passed for demos
    and as not-a-false-pass for negatives — an oracle that can't read its own
    demos is never accepted.
    """
    ep = os.path.join(workdir, f"ep_{tag}")
    save_episode(ep, f"st_{tag}", frames, source="selftest")
    ref = load_rollout(ep)
    gdir = os.path.join(workdir, f"g_{tag}")
    os.makedirs(gdir, exist_ok=True)
    return grounder.ground(ref, spec, gdir)


def run_selftest(
    spec: TaskSpec,
    demos: list[np.ndarray],
    negatives: list[tuple[str, np.ndarray]],
    *,
    grounder: str = "relational.motion",
    max_rounds: int = 6,
) -> SelfTestOutcome:
    g = reg_get("grounder", grounder)()
    from woracle.channels.verdict import PredicateSuccessChannel  # real channel

    base_tol = None
    for pred in spec.success:
        if pred.kind == "co_located" and "tol_rel" in pred.params:
            base_tol = float(pred.params["tol_rel"])
    grid = [1.0] if base_tol is None else [0.6, 0.8, 1.0, 1.3, 1.7, 2.2][:max_rounds]

    best = None  # (demos_passed, negatives_failed, tol_mult)
    with tempfile.TemporaryDirectory(prefix="woracle-selftest-") as workdir:
        # Ground each episode exactly ONCE (grounding is tolerance-independent);
        # only the predicate evaluation re-runs per tolerance. Every minted
        # negative gets a UNIQUE tag — the C-1 lesson: a provenance count must
        # never be aggregated from aliased cache entries.
        episodes: list[tuple[str, str]] = []  # (kind, unique_tag)
        grounded_by_tag: dict[str, object] = {}
        for i, frames in enumerate(demos):
            tag = f"demo{i}"
            grounded_by_tag[tag] = _ground_once(spec, frames, g, workdir, tag)
            episodes.append(("demo", tag))
        for i, (kind, frames) in enumerate(negatives):
            tag = f"neg{i}_{kind}"
            grounded_by_tag[tag] = _ground_once(spec, frames, g, workdir, tag)
            episodes.append(("neg", tag))

        for mult in grid:
            trial = spec.model_copy(deep=True)
            for pred in trial.success:
                if pred.kind == "co_located" and "tol_rel" in pred.params and base_tol:
                    pred.params["tol_rel"] = base_tol * mult
            dp = nf = 0
            for kind, tag in episodes:
                grounded = grounded_by_tag[tag]
                score = PredicateSuccessChannel().score(grounded, trial)
                v = (
                    None
                    if (score.status != "ok" or score.value is None)
                    else bool(score.value >= 0.5)
                )
                if kind == "demo":
                    dp += 1 if v is True else 0
                else:
                    nf += 1 if (v is False or v is None) else 0
            if best is None or (dp, nf) > (best[0], best[1]):
                best = (dp, nf, mult)
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
