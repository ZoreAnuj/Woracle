"""Prediction-powered inference (PPI) for judge-rectified success estimates.

The P5 statistics core: many judge-scored rollouts + a few gold labels give
an UNBIASED estimate of the true success rate with a valid CI — the judge's
systematic error is measured on the labeled subset and subtracted
(Angelopoulos et al., PPI; power-tuned lam per PPI++).

    theta(lam) = lam*mean(f(X_unlab)) + mean(Y_lab - lam*f(X_lab))
    Var   = lam^2*var(f_unlab)/N + var(Y_lab - lam*f_lab)/n
    (sets are DISJOINT — overlapping them breaks the independence the
    variance formula needs; measured: coverage 0.88 vs nominal 0.90)
    lam*    chosen to minimize Var (clipped to [0, 1]); lam=1 is classic PPI,
    lam=0 degenerates to the gold-only (classical) estimate.

Stdlib + numpy only (NormalDist for quantiles) — the kernel rule holds.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from woracle.errors import SpecError


@dataclass(frozen=True)
class PPIEstimate:
    estimate: float
    ci_low: float
    ci_high: float
    lam: float  # power-tuning weight actually used
    n_unlabeled: int
    n_labeled: int
    classical_width: float  # gold-only CI width, for the honesty comparison
    width: float

    @property
    def narrower_than_classical(self) -> bool:
        return self.width < self.classical_width


def ppi_mean(
    judge_unlabeled: np.ndarray,
    judge_labeled: np.ndarray,
    gold_labeled: np.ndarray,
    *,
    alpha: float = 0.1,
    tune_lambda: bool = True,
) -> PPIEstimate:
    """PPI estimate of E[Y]: judge scores on the UNLABELED set + judge & gold
    on the DISJOINT labeled subset.

    judge_unlabeled : (N,) judge scores on rollouts WITHOUT gold labels
    judge_labeled   : (n,) judge scores on the gold-labeled subset
    gold_labeled    : (n,) ground-truth outcomes on that subset
    """
    f_all = np.asarray(judge_unlabeled, float)
    f_lab = np.asarray(judge_labeled, float)
    y_lab = np.asarray(gold_labeled, float)
    if f_lab.shape != y_lab.shape or f_lab.ndim != 1:
        raise SpecError("judge_labeled and gold_labeled must be matching 1-D arrays")
    n, N = len(y_lab), len(f_all)
    if n < 2 or N < 2:
        raise SpecError("need at least 2 labeled and 2 judged rollouts")
    if not (np.isfinite(f_all).all() and np.isfinite(f_lab).all() and np.isfinite(y_lab).all()):
        raise SpecError("PPI inputs must be finite")

    if tune_lambda and n >= 4:
        cov = float(np.cov(f_lab, y_lab, ddof=1)[0, 1])
        var_f = float(np.var(f_lab, ddof=1))
        lam = 0.0 if var_f <= 1e-12 else float(np.clip(cov / var_f, 0.0, 1.0))
    else:
        lam = 1.0

    rectifier = y_lab - lam * f_lab
    theta = lam * float(f_all.mean()) + float(rectifier.mean())
    var = (lam**2) * float(np.var(f_all, ddof=1)) / N + float(np.var(rectifier, ddof=1)) / n
    z = NormalDist().inv_cdf(1 - alpha / 2)
    half = z * float(np.sqrt(max(var, 0.0)))

    classical_half = z * float(np.sqrt(np.var(y_lab, ddof=1) / n))
    return PPIEstimate(
        estimate=float(theta),
        ci_low=float(theta - half),
        ci_high=float(theta + half),
        lam=lam,
        n_unlabeled=N,
        n_labeled=n,
        classical_width=2 * classical_half,
        width=2 * half,
    )
