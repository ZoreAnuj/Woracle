"""Bootstrap rank intervals for leaderboards (P5).

A point ranking hides how fragile the order is; the bootstrap rank interval
says "policy A is rank 1-2 with 95% confidence". Graded (non-abstained)
verdicts only — abstention handling lives in stats.mnar, and the two are
reported side by side.
"""

from __future__ import annotations

import numpy as np

from woracle.errors import SpecError


def rank_intervals(
    scores_by_policy: dict[str, list[float]],
    *,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, dict[str, float]]:
    """Per policy: mean, rank_low, rank_high (1 = best, by bootstrap of means)."""
    names = sorted(scores_by_policy)
    if len(names) < 2:
        raise SpecError("rank intervals need at least 2 policies")
    arrs = []
    for n in names:
        a = np.asarray(scores_by_policy[n], float)
        if len(a) == 0 or not np.isfinite(a).all():
            raise SpecError(f"policy '{n}' has no finite graded scores")
        arrs.append(a)
    rng = np.random.default_rng(seed)
    P = len(names)
    ranks = np.empty((n_boot, P), int)
    for b in range(n_boot):
        means = [float(rng.choice(a, size=len(a), replace=True).mean()) for a in arrs]
        order = np.argsort([-m for m in means], kind="mergesort")
        r = np.empty(P, int)
        r[order] = np.arange(1, P + 1)
        ranks[b] = r
    lo_q, hi_q = 100 * alpha / 2, 100 * (1 - alpha / 2)
    out: dict[str, dict[str, float]] = {}
    for i, n in enumerate(names):
        out[n] = {
            "mean": float(np.mean(arrs[i])),
            "rank_low": float(np.percentile(ranks[:, i], lo_q)),
            "rank_high": float(np.percentile(ranks[:, i], hi_q)),
        }
    return out
