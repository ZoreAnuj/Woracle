"""P2 exit gate: composed validity-gate AUROC ≥ 0.8 on the labeled
generative-corruption benchmark (clean vs broken blobworld episodes).

Also calibrates a conformal threshold on half the benchmark and verifies the
risk guarantee on the held-out half — the full P2 calibration loop, end to
end, on ground truth we control.
"""

from __future__ import annotations

import os

import numpy as np

from woracle.gate.signals import (
    AppearanceConsistencySignal,
    BackgroundDriftSignal,
    TrackContinuitySignal,
)
from woracle.io import load_rollout, save_episode
from woracle.stats.conformal import auroc, calibrate_threshold
from woracle.testing.blobworld import blob_spec
from woracle.testing.corruptions import gate_benchmark
from woracle.testing.plugins import (
    BlobColorGrounder,
    MotionSanitySignal,
    PermanenceSignal,
)

SIGNALS = [
    PermanenceSignal,
    MotionSanitySignal,
    BackgroundDriftSignal,
    AppearanceConsistencySignal,
    TrackContinuitySignal,
]


def _composed_score(grounded) -> float:
    """min over available signal values; missing evidence scores 0 (a rollout
    whose health cannot be measured is not healthy)."""
    vals = []
    for cls in SIGNALS:
        v = cls().measure(grounded)
        vals.append(v.value if (v.status == "ok" and v.value is not None) else 0.0)
    return float(min(vals))


def test_gate_auroc_and_conformal_loop(tmp_path) -> None:
    spec = blob_spec()
    grounder = BlobColorGrounder()
    names, scores, broken = [], [], []
    for name, frames, is_broken in gate_benchmark(seeds=range(3), n_frames=60):
        ep = str(tmp_path / name)
        save_episode(ep, name, frames, source="blobworld")
        ref = load_rollout(ep)
        out = str(tmp_path / f"{name}_g")
        os.makedirs(out, exist_ok=True)
        grounded = grounder.ground(ref, spec, out)
        names.append(name)
        scores.append(_composed_score(grounded))
        broken.append(is_broken)

    scores_arr = np.array(scores)
    broken_arr = np.array(broken)
    n_broken, n_clean = int(broken_arr.sum()), int((~broken_arr).sum())
    assert n_broken >= 12 and n_clean >= 6  # benchmark actually has both classes

    # ---- exit gate: AUROC ≥ 0.8 ----
    a = auroc(scores_arr[~broken_arr], scores_arr[broken_arr])
    per_kind = {
        k: round(float(np.mean(scores_arr[[k in n for n in names]])), 3)
        for k in (
            "success",
            "fail_miss",
            "vanish_late",
            "freeze",
            "noise_ramp",
            "teleport",
            "bg_morph",
        )
    }
    assert a >= 0.8, f"gate AUROC {a:.3f} < 0.8; per-kind mean scores: {per_kind}"

    # ---- conformal loop: calibrate on even seeds, verify risk on odd ----
    cal = np.array([("s0" in n) or ("s2" in n) for n in names])
    res = calibrate_threshold(scores_arr[cal], broken_arr[cal], alpha=0.34)
    held_broken = scores_arr[~cal & broken_arr]
    if len(held_broken):
        admitted = float((held_broken >= res.threshold).mean())
        # One-shot check, small n: allow generous slack over the bound; the
        # distributional guarantee itself is Monte-Carlo-verified in
        # tests/unit/test_conformal.py.
        assert admitted <= 0.6, f"held-out broken admission {admitted:.2f} wildly over bound"
    assert res.clean_retention >= 0.5, "calibrated gate rejects most CLEAN rollouts"
