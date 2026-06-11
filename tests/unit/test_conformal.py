"""Conformal risk-control calibration math (P2)."""

from __future__ import annotations

import numpy as np
import pytest

from woracle.errors import SpecError
from woracle.stats.conformal import auroc, calibrate_threshold


def test_separable_distributions_calibrate_cleanly() -> None:
    rng = np.random.default_rng(0)
    clean = rng.uniform(0.7, 1.0, 200)
    broken = rng.uniform(0.0, 0.3, 60)
    values = np.concatenate([clean, broken])
    labels = np.concatenate([np.zeros(200, bool), np.ones(60, bool)])
    res = calibrate_threshold(values, labels, alpha=0.1)
    # CRC picks the MOST PERMISSIVE valid threshold: with 60 broken examples it
    # may admit a few (8.3% here) while the corrected bound still meets alpha.
    assert res.guaranteed_risk <= 0.1
    assert res.achieved_risk <= 0.1
    assert res.clean_retention == 1.0
    # tightening alpha forces the threshold into the separation gap
    strict = calibrate_threshold(values, labels, alpha=0.025)
    assert broken.max() < strict.threshold <= clean.min() + 1e-9
    assert strict.achieved_risk == 0.0


def test_guarantee_holds_empirically_under_overlap() -> None:
    """Monte-Carlo check of the CRC bound: fresh broken draws are admitted at
    a rate within the guaranteed risk (averaged over trials)."""
    rng = np.random.default_rng(1)
    rates = []
    for _ in range(60):
        clean = rng.normal(0.75, 0.12, 80)
        broken = rng.normal(0.45, 0.15, 40)
        res = calibrate_threshold(
            np.concatenate([clean, broken]),
            np.concatenate([np.zeros(80, bool), np.ones(40, bool)]),
            alpha=0.2,
        )
        fresh_broken = rng.normal(0.45, 0.15, 400)
        rates.append((fresh_broken >= res.threshold).mean())
    assert float(np.mean(rates)) <= 0.2 + 0.03  # bound + MC slack


def test_alpha_floor_is_refused_not_faked() -> None:
    values = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([False, False, True, True])
    with pytest.raises(SpecError, match="refusing to fake a guarantee"):
        calibrate_threshold(values, labels, alpha=0.05)  # floor = 1/3 > 0.05


def test_no_broken_examples_refused() -> None:
    with pytest.raises(SpecError, match="without broken examples"):
        calibrate_threshold(np.array([0.5, 0.6]), np.array([False, False]))


def test_all_admitting_impossible_returns_admit_nothing() -> None:
    # Broken scores ABOVE clean (pathological): only "admit nothing" satisfies.
    values = np.array([0.2, 0.3, 0.9, 0.95] + [0.9] * 12)
    labels = np.array([False, False] + [True] * 14)
    res = calibrate_threshold(values, labels, alpha=0.1)
    assert res.threshold > values.max()
    assert res.clean_retention == 0.0  # honest: nothing passes, and it says so


def test_auroc_basics() -> None:
    assert auroc(np.array([1.0, 0.9]), np.array([0.1, 0.2])) == 1.0
    assert auroc(np.array([0.5]), np.array([0.5])) == 0.5
    assert abs(auroc(np.array([0.4, 0.8]), np.array([0.6])) - 0.5) < 1e-9
