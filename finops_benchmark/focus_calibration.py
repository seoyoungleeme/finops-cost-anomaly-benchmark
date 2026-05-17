"""Extract real-world statistics from FOCUS daily series to calibrate the synthetic generator."""

from typing import Dict, Optional

import numpy as np
import pandas as pd


def fit_series_statistics(series: pd.Series) -> Dict:
    """Fit calibration parameters from a real daily cost series.

    Parameters
    ----------
    series : pd.Series
        Daily cost values indexed by DatetimeIndex (>= 7 observations required).

    Returns
    -------
    dict with keys:
        base_level      : mean daily cost
        monthly_growth  : estimated fractional monthly linear trend
        noise_pct       : coefficient of variation of de-trended residuals
        weekly_factor   : list of 7 DoW multipliers (Mon=0 … Sun=6),
                          normalized so they average to 1.0
    """
    if len(series) < 7:
        raise ValueError(f"Need >= 7 daily observations, got {len(series)}.")

    values = series.values.astype(float)
    mean_val = float(np.mean(values))
    if mean_val <= 0:
        raise ValueError("Mean cost is zero or negative — cannot calibrate.")

    n = len(values)
    t = np.arange(n, dtype=float)

    # Linear trend via least-squares
    slope, intercept = np.polyfit(t, values, 1)
    monthly_growth = float(np.clip(slope * 30.0 / mean_val, -0.10, 0.30))

    # Noise: std of de-trended residuals relative to mean
    residuals = values - (slope * t + intercept)
    noise_pct = float(np.clip(np.std(residuals) / mean_val, 0.01, 0.50))

    # Day-of-week multiplicative pattern using actual calendar dates
    dows = pd.DatetimeIndex(series.index).dayofweek.values  # 0=Mon, 6=Sun
    dow_sums = np.zeros(7)
    dow_counts = np.zeros(7)
    for i, d in enumerate(dows):
        dow_sums[d] += values[i]
        dow_counts[d] += 1
    # Fill days with no observations using the global mean
    for d in range(7):
        if dow_counts[d] == 0:
            dow_sums[d] = mean_val
            dow_counts[d] = 1
    dow_means = dow_sums / dow_counts
    weekly_factor = (dow_means / dow_means.mean()).tolist()

    return {
        "base_level": mean_val,
        "monthly_growth": monthly_growth,
        "noise_pct": noise_pct,
        "weekly_factor": weekly_factor,
    }


def calibrate_all_services(daily_dict: Dict[str, pd.Series]) -> Dict[str, Dict]:
    """Fit statistics for every service in *daily_dict*.

    Groups that cannot be calibrated (too few days, zero mean) are skipped
    without raising an exception.
    """
    result: Dict[str, Dict] = {}
    for label, series in daily_dict.items():
        try:
            result[label] = fit_series_statistics(series)
        except ValueError:
            pass
    return result
