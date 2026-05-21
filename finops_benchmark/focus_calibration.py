"""Extract real-world statistics from FOCUS daily series to calibrate the synthetic generator."""

from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

_MIN_FIT_DAYS = 21   # minimum days required to fit service-specific stats
_MONTHLY_GROWTH_BOUNDS = (-0.02, 0.10)
_NOISE_PCT_BOUNDS = (0.01, 0.15)
_WEEKLY_FACTOR_BOUNDS = (0.25, 2.50)

# A parameter is "clipping-saturated" when its raw value hits a bound exactly.
# Saturation means the real signal may be stronger than what the model sees.
def _is_saturated(raw: float, lo: float, hi: float) -> bool:
    return raw <= lo or raw >= hi


def _blend_weekly_factor(
    service_wf: Sequence[float],
    global_wf: Sequence[float],
    alpha: float = 0.5,
) -> list:
    """Return a 50/50 blend of service and global weekly factors, re-normalized to mean=1."""
    blended = [alpha * s + (1 - alpha) * g for s, g in zip(service_wf, global_wf)]
    blended = np.clip(blended, *_WEEKLY_FACTOR_BOUNDS)
    mean_b = sum(blended) / 7
    return [w / mean_b for w in blended]


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
        monthly_growth  : fractional monthly trend, clipped conservatively
        noise_pct       : CV of de-trended residuals, clipped conservatively
        weekly_factor   : list of 7 DoW multipliers (Mon=0 to Sun=6), mean=1
    """
    if len(series) < 7:
        raise ValueError(f"Need >= 7 daily observations, got {len(series)}.")

    # Credits and corrections can make EffectiveCost negative in short FOCUS
    # samples; synthetic cloud spend baselines should stay non-negative.
    values = np.clip(series.values.astype(float), 0.0, None)
    mean_val = float(np.mean(values))
    if mean_val <= 0:
        raise ValueError("Mean cost is zero or negative; cannot calibrate.")

    n = len(values)
    t = np.arange(n, dtype=float)

    # Conservative linear trend: tight clip because 30-day samples are noisy
    slope, intercept = np.polyfit(t, values, 1)
    monthly_growth_raw = float(slope * 30.0 / mean_val)
    monthly_growth = float(np.clip(monthly_growth_raw, *_MONTHLY_GROWTH_BOUNDS))

    # Noise from de-trended residuals
    residuals = values - (slope * t + intercept)
    noise_pct_raw = float(np.std(residuals) / mean_val)
    noise_pct = float(np.clip(noise_pct_raw, *_NOISE_PCT_BOUNDS))

    # Day-of-week pattern via actual calendar dates
    dows = pd.DatetimeIndex(series.index).dayofweek.values   # 0=Mon, 6=Sun
    dow_sums = np.zeros(7)
    dow_counts = np.zeros(7)
    for i, d in enumerate(dows):
        dow_sums[d] += values[i]
        dow_counts[d] += 1
    for d in range(7):
        if dow_counts[d] == 0:
            dow_sums[d] = mean_val
            dow_counts[d] = 1
    dow_means = dow_sums / dow_counts
    weekly_factor = np.clip(
        dow_means / dow_means.mean(), *_WEEKLY_FACTOR_BOUNDS
    )
    weekly_factor = (weekly_factor / weekly_factor.mean()).tolist()

    return {
        "base_level": mean_val,
        "monthly_growth": monthly_growth,
        "monthly_growth_raw": monthly_growth_raw,
        "monthly_growth_saturated": _is_saturated(
            monthly_growth_raw, *_MONTHLY_GROWTH_BOUNDS
        ),
        "noise_pct": noise_pct,
        "noise_pct_raw": noise_pct_raw,
        "noise_pct_saturated": _is_saturated(noise_pct_raw, *_NOISE_PCT_BOUNDS),
        "weekly_factor": weekly_factor,
    }


def compute_global_stats(daily_dict: Dict[str, pd.Series]) -> Dict:
    """Compute median statistics across all service groups for use as fallback.

    Uses any series with >= 7 days and mean > 0.  Weekly factors are averaged.
    Returns safe defaults if no series can be fit.
    """
    stats_list = []
    for series in daily_dict.values():
        if len(series) >= 7 and series.mean() > 0:
            try:
                stats_list.append(fit_series_statistics(series))
            except ValueError:
                pass

    if not stats_list:
        return {
            "base_level": 1000.0,
            "monthly_growth": 0.0,
            "noise_pct": 0.05,
            "weekly_factor": [1.0] * 7,
        }

    return {
        "base_level": float(np.median([s["base_level"] for s in stats_list])),
        "monthly_growth": float(
            np.clip(
                np.median([s["monthly_growth"] for s in stats_list]),
                *_MONTHLY_GROWTH_BOUNDS,
            )
        ),
        "noise_pct": float(
            np.clip(np.median([s["noise_pct"] for s in stats_list]), *_NOISE_PCT_BOUNDS)
        ),
        "weekly_factor": (
            lambda wf: (wf / wf.mean()).tolist()
        )(
            np.clip(
                np.mean(np.array([s["weekly_factor"] for s in stats_list]), axis=0),
                *_WEEKLY_FACTOR_BOUNDS,
            )
        ),
    }


def calibrate_all_services(
    daily_dict: Dict[str, pd.Series],
    global_stats: Optional[Dict] = None,
    min_fit_days: int = _MIN_FIT_DAYS,
) -> Dict[str, Dict]:
    """Fit statistics for every service in *daily_dict*.

    For each service:
    - If it has >= ``min_fit_days`` observations, fit service-specific stats and
      smooth the weekly factor 50/50 with global.
    - Otherwise, fall back to global stats and record the reason.

    Each returned dict includes extra metadata fields:
    ``days_observed``, ``nonzero_days``, ``used_fallback``, ``fallback_reason``.
    These extra fields are ignored by ``build_focus_calibrated_dataset``.
    """
    if global_stats is None:
        global_stats = compute_global_stats(daily_dict)

    result: Dict[str, Dict] = {}
    for label, series in daily_dict.items():
        days_observed = len(series)
        nonzero_days = int((series > 0).sum())
        used_fallback = False
        fallback_reason = ""

        try:
            if days_observed < min_fit_days:
                raise ValueError(
                    f"Only {days_observed} days observed (need {min_fit_days})"
                )
            stats = fit_series_statistics(series)
            # Smooth weekly factor to reduce over-fitting on short samples
            stats["weekly_factor"] = _blend_weekly_factor(
                stats["weekly_factor"], global_stats["weekly_factor"]
            )
        except ValueError as exc:
            stats = dict(global_stats)
            used_fallback = True
            fallback_reason = str(exc)

        stats.update({
            "service": label,
            "days_observed": days_observed,
            "nonzero_days": nonzero_days,
            "used_fallback": used_fallback,
            "fallback_reason": fallback_reason,
        })
        result[label] = stats

    return result


def save_calibration_stats(
    stats_all: Dict[str, Dict],
    output_path: str,
) -> None:
    """Write per-service calibration metadata to a CSV file.

    Columns: service, days_observed, nonzero_days, base_level, monthly_growth,
    noise_pct, weekly_factor (JSON list), used_fallback, fallback_reason.
    """
    import json

    rows = []
    for label, st in stats_all.items():
        mg_raw = st.get("monthly_growth_raw")
        np_raw = st.get("noise_pct_raw")
        rows.append({
            "service": label,
            "days_observed": st.get("days_observed", ""),
            "nonzero_days": st.get("nonzero_days", ""),
            "base_level": round(st["base_level"], 4),
            "monthly_growth_raw": round(mg_raw, 4) if mg_raw is not None else "",
            "monthly_growth": round(st["monthly_growth"], 4),
            "monthly_growth_saturated": (
                st.get("monthly_growth_saturated")
                if mg_raw is not None else ""
            ),
            "noise_pct_raw": round(np_raw, 4) if np_raw is not None else "",
            "noise_pct": round(st["noise_pct"], 4),
            "noise_pct_saturated": (
                st.get("noise_pct_saturated")
                if np_raw is not None else ""
            ),
            "weekly_factor": json.dumps([round(w, 4) for w in st["weekly_factor"]]),
            "used_fallback": st.get("used_fallback", False),
            "fallback_reason": st.get("fallback_reason", ""),
        })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)
