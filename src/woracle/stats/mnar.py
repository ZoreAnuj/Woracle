"""MNAR abstention sensitivity — the honesty statistic nobody ships.

Abstention is informative missingness: WM breakage correlates with policy
quality (WorldEval), so silently dropping abstained rollouts biases rankings.
This module computes the per-policy pass-rate BOUNDS under best/worst-case
imputation of every abstain, and reports which pairwise orderings survive the
whole bound interval ("robust") vs which can flip ("undetermined").
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PolicyBounds:
    policy: str
    n: int
    n_pass: int
    n_fail: int
    n_abstain: int
    rate_low: float  # every abstain counted as fail
    rate_high: float  # every abstain counted as pass

    @property
    def abstain_frac(self) -> float:
        return self.n_abstain / self.n if self.n else 0.0


@dataclass(frozen=True)
class SensitivityReport:
    bounds: list[PolicyBounds]
    robust_pairs: list[tuple[str, str]] = field(default_factory=list)  # a > b always
    undetermined_pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def ranking_is_robust(self) -> bool:
        return not self.undetermined_pairs


def abstention_sensitivity(verdicts_by_policy: dict[str, list[str]]) -> SensitivityReport:
    """verdicts: per policy, a list drawn from {'pass','fail','abstain'}."""
    bounds: list[PolicyBounds] = []
    for policy, vs in sorted(verdicts_by_policy.items()):
        n = len(vs)
        n_pass = sum(1 for v in vs if v == "pass")
        n_fail = sum(1 for v in vs if v == "fail")
        n_abstain = n - n_pass - n_fail
        bounds.append(
            PolicyBounds(
                policy=policy,
                n=n,
                n_pass=n_pass,
                n_fail=n_fail,
                n_abstain=n_abstain,
                rate_low=(n_pass / n) if n else 0.0,
                rate_high=((n_pass + n_abstain) / n) if n else 0.0,
            )
        )
    robust: list[tuple[str, str]] = []
    undet: list[tuple[str, str]] = []
    for i, a in enumerate(bounds):
        for b in bounds[i + 1 :]:
            if a.rate_low > b.rate_high:
                robust.append((a.policy, b.policy))
            elif b.rate_low > a.rate_high:
                robust.append((b.policy, a.policy))
            else:
                undet.append((a.policy, b.policy))
    return SensitivityReport(bounds=bounds, robust_pairs=robust, undetermined_pairs=undet)
