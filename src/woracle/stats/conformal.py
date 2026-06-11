"""Conformal calibration for gate thresholds (P2).

Implements split-conformal risk control for the gate's one-dimensional,
monotone setting: signals emit higher-is-healthier values; lowering a
threshold lam admits more rollouts but raises the risk of admitting BROKEN
ones. Given a labeled calibration set, ``calibrate_threshold`` returns the
most permissive lam whose finite-sample-corrected broken-admission risk stays
<= alpha — the conformal-risk-control bound (Angelopoulos et al., "Conformal Risk
Control": choose lhat = inf{lam : (n/(n+1))*Rhat(lam) + B/(n+1) <= alpha} for a bounded
monotone risk, B = 1 here).

This is the load-bearing math behind GatePolicy thresholds from P2 on;
numbers in DEFAULT_POLICY are explicitly uncalibrated defaults.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from woracle.errors import SpecError


@dataclass(frozen=True)
class CalibrationResult:
    threshold: float  # admit rollouts with signal value >= threshold
    achieved_risk: float  # empirical broken-admission rate at threshold
    guaranteed_risk: float  # finite-sample-corrected bound that was enforced
    n_broken: int
    n_clean: int
    clean_retention: float  # fraction of clean rollouts admitted (power)


def calibrate_threshold(
    values: np.ndarray,
    is_broken: np.ndarray,
    *,
    alpha: float = 0.1,
) -> CalibrationResult:
    """Most permissive admit-threshold with broken-admission risk <= alpha.

    values     : (n,) signal values (higher = healthier)
    is_broken  : (n,) bool — ground-truth "this rollout is broken/ungradeable"
    alpha      : target risk level for admitting a broken rollout

    Risk(lam) = P(broken admitted) = mean_{broken}[ value >= lam ] — monotone
    non-increasing in lam, bounded in [0,1] ⇒ CRC applies with B = 1:
        lam̂ = inf{ lam : (n_b/(n_b+1))*Rhat(lam) + 1/(n_b+1) <= alpha }
    computed over the candidate lam grid of observed values (+∞ fallback).
    """
    values = np.asarray(values, dtype=float)
    is_broken = np.asarray(is_broken, dtype=bool)
    if values.shape != is_broken.shape or values.ndim != 1:
        raise SpecError("values and is_broken must be 1-D arrays of equal length")
    if not np.isfinite(values).all():
        raise SpecError("calibration values must be finite")
    broken = values[is_broken]
    clean = values[~is_broken]
    n_b = len(broken)
    if n_b == 0:
        raise SpecError("cannot calibrate without broken examples — the guarantee would be vacuous")
    # Feasibility: even the strictest threshold gives bound 1/(n_b+1).
    if 1.0 / (n_b + 1) > alpha:
        raise SpecError(
            f"alpha={alpha} unattainable with only {n_b} broken examples "
            f"(finite-sample floor is {1.0 / (n_b + 1):.3f}); collect more labels "
            "or raise alpha — refusing to fake a guarantee"
        )
    # Candidate thresholds: descending unique values; pick the smallest lam
    # whose corrected risk meets alpha (most permissive valid choice).
    candidates = np.unique(values)[::-1]
    chosen = None
    for lam in candidates:
        risk_hat = float((broken >= lam).mean())
        bound = (n_b / (n_b + 1)) * risk_hat + 1.0 / (n_b + 1)
        if bound <= alpha:
            chosen = (float(lam), risk_hat, float(bound))
        else:
            break  # risk is monotone: lower lam only admits more broken rollouts
    if chosen is None:
        # No observed value satisfies the bound except "admit nothing".
        lam = float(np.nextafter(values.max(), np.inf))
        return CalibrationResult(
            threshold=lam,
            achieved_risk=0.0,
            guaranteed_risk=1.0 / (n_b + 1),
            n_broken=n_b,
            n_clean=len(clean),
            clean_retention=0.0,
        )
    lam, risk_hat, bound = chosen
    retention = float((clean >= lam).mean()) if len(clean) else 0.0
    return CalibrationResult(
        threshold=lam,
        achieved_risk=risk_hat,
        guaranteed_risk=bound,
        n_broken=n_b,
        n_clean=len(clean),
        clean_retention=retention,
    )


def auroc(healthy_scores: np.ndarray, broken_scores: np.ndarray) -> float:
    """Rank-based AUROC: P(healthy > broken) + 0.5·P(tie). Exit-gate metric."""
    h = np.asarray(healthy_scores, float)
    b = np.asarray(broken_scores, float)
    if len(h) == 0 or len(b) == 0:
        raise SpecError("AUROC needs both classes")
    wins = (h[:, None] > b[None, :]).sum() + 0.5 * (h[:, None] == b[None, :]).sum()
    return float(wins / (len(h) * len(b)))
