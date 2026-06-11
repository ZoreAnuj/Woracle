"""S5 statistics — kernel-pure (numpy + stdlib only, by construction)."""

from woracle.stats.conformal import CalibrationResult, auroc, calibrate_threshold
from woracle.stats.mnar import PolicyBounds, SensitivityReport, abstention_sensitivity
from woracle.stats.ppi import PPIEstimate, ppi_mean
from woracle.stats.rankboot import rank_intervals

__all__ = [
    "CalibrationResult",
    "PPIEstimate",
    "PolicyBounds",
    "SensitivityReport",
    "abstention_sensitivity",
    "auroc",
    "calibrate_threshold",
    "ppi_mean",
    "rank_intervals",
]
