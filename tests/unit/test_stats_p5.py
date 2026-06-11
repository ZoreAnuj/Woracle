"""P5 statistics: PPI (unbiasedness + CI coverage), MNAR sensitivity, VOC ties."""

from __future__ import annotations

import numpy as np
import pytest

from woracle.errors import SpecError
from woracle.judges.base import parse_progress_reply, value_order_correlation
from woracle.stats.mnar import abstention_sensitivity
from woracle.stats.ppi import ppi_mean


def _world(rng, N, n, judge_noise=0.25, judge_bias=0.15):
    y = rng.binomial(1, 0.6, N + n).astype(float)
    f = np.clip(y + judge_bias + rng.normal(0, judge_noise, N + n), 0, 1)
    return f[:N], f[N:], y[N:], float(y.mean())  # unlabeled f, labeled f, gold, ~truth


def test_ppi_unbiased_and_covering() -> None:
    rng = np.random.default_rng(0)
    errs, cover, widths_ppi, widths_classical = [], [], [], []
    true_p = 0.6
    for _ in range(400):
        f_unlab, f_lab, y_lab, _ = _world(rng, N=400, n=40)
        est = ppi_mean(f_unlab, f_lab, y_lab, alpha=0.1)
        errs.append(est.estimate - true_p)
        cover.append(est.ci_low <= true_p <= est.ci_high)
        widths_ppi.append(est.width)
        widths_classical.append(est.classical_width)
    assert abs(float(np.mean(errs))) < 0.01  # unbiased despite judge bias
    assert float(np.mean(cover)) >= 0.875  # ~nominal 0.90 coverage
    # a correlated judge must BUY precision over gold-only
    assert float(np.mean(widths_ppi)) < float(np.mean(widths_classical))


def test_ppi_lambda_zero_on_useless_judge() -> None:
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.5, 60).astype(float)
    junk_lab = rng.uniform(0, 1, 60)
    junk_unlab = rng.uniform(0, 1, 500)
    est = ppi_mean(junk_unlab, junk_lab, y, alpha=0.1)
    assert est.lam < 0.3  # power-tuning refuses to lean on noise


def test_ppi_input_validation() -> None:
    with pytest.raises(SpecError):
        ppi_mean(np.ones(5), np.ones(3), np.ones(4))
    with pytest.raises(SpecError):
        ppi_mean(np.array([np.nan, 1.0]), np.ones(3), np.ones(3))


def test_mnar_bounds_and_flips() -> None:
    rep = abstention_sensitivity(
        {
            "good": ["pass"] * 8 + ["fail"] * 1 + ["abstain"] * 1,
            "bad": ["pass"] * 1 + ["fail"] * 8 + ["abstain"] * 1,
            "shy": ["pass"] * 2 + ["abstain"] * 8,
        }
    )
    by = {b.policy: b for b in rep.bounds}
    assert by["good"].rate_low == 0.8 and by["good"].rate_high == 0.9
    assert by["shy"].rate_low == 0.2 and by["shy"].rate_high == 1.0
    assert ("good", "bad") in rep.robust_pairs  # survives worst case
    assert ("good", "shy") in rep.undetermined_pairs  # abstains could flip it
    assert not rep.ranking_is_robust


def test_voc_average_ranks_under_ties() -> None:
    # critic I-3 reproductions: ordinal ranks said 1.0 / -0.2; truth is ±0.775
    assert value_order_correlation([0, 0, 0, 1], [0, 1, 2, 3]) == pytest.approx(0.7746, abs=1e-3)
    assert value_order_correlation([1, 0, 0, 0], [0, 1, 2, 3]) == pytest.approx(-0.7746, abs=1e-3)


def test_parser_anchor_counted_as_frame_one() -> None:
    # model counted the anchor: shuffled frames reported as Frames 2..5
    reply = "Frame 2: 10%\nFrame 3: 40%\nFrame 4: 70%\nFrame 5: 90%"
    assert parse_progress_reply(reply, 4) == [0.1, 0.4, 0.7, 0.9]
