"""Statistical evaluation helpers including Wilson score confidence intervals."""
from __future__ import annotations
import math

try:
    from statsmodels.stats.proportion import proportion_confint as _statsmodels_confint
    def proportion_confint(count: int, nobs: int, alpha: float = 0.05, method: str = "wilson") -> tuple[float, float]:
        lo, hi = _statsmodels_confint(count, nobs, alpha=alpha, method=method)
        return (round(float(lo), 4), round(float(hi), 4))
except ImportError:
    def proportion_confint(count: int, nobs: int, alpha: float = 0.05, method: str = "wilson") -> tuple[float, float]:
        """Compute two-sided Wilson score confidence interval matching statsmodels."""
        if nobs <= 0:
            return (0.0, 0.0)
        p = count / nobs
        if abs(alpha - 0.05) < 1e-4:
            z = 1.959963984540054
        elif abs(alpha - 0.01) < 1e-4:
            z = 2.5758293035489004
        elif abs(alpha - 0.10) < 1e-4:
            z = 1.6448536269514722
        else:
            z = 1.959963984540054

        z2 = z * z
        denom = 1 + z2 / nobs
        center = (p + z2 / (2 * nobs)) / denom
        margin = (z / denom) * math.sqrt((p * (1 - p) / nobs) + (z2 / (4 * nobs * nobs)))
        lo = max(0.0, center - margin)
        hi = min(1.0, center + margin)
        return (round(lo, 4), round(hi, 4))
