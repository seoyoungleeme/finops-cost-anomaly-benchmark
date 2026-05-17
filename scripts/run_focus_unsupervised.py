"""CLI: run unsupervised anomaly detection directly on real FOCUS daily cost series.

Approach B — real-data sanity check:
  1. Download FOCUS billing data and aggregate to daily cost per service.
  2. Apply a 7-day rolling z-score detector (no ground-truth labels needed).
  3. Report flagged dates and a per-service alert summary CSV.

Because each FOCUS sample covers ~30 days, this is a sanity check rather than
a rigorous benchmark.  It answers: "which real services have suspicious cost
spikes?" and validates that the models behave sensibly on production data.

Usage
-----
  python scripts/run_focus_unsupervised.py
  python scripts/run_focus_unsupervised.py --sigma 2.0 --window 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finops_benchmark.config import (
    FOCUS_CACHE_DIR,
    FOCUS_DATA_URL,
    FOCUS_GROUP_BY,
    RESULTS_DIR,
)
from finops_benchmark.focus_loader import aggregate_daily, download_focus_data, load_focus_data


def rolling_zscore(series: pd.Series, window: int = 7, sigma: float = 2.5):
    """Return z-scores and boolean alert flags using a rolling window."""
    rm = series.rolling(window, min_periods=3).mean()
    rs = series.rolling(window, min_periods=3).std().clip(lower=1e-9)
    z = (series - rm) / rs
    return z, z.abs() > sigma


def main(args: argparse.Namespace) -> None:
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Download & load ───────────────────────────────────────────────────────
    print("Downloading FOCUS sample data...")
    path = download_focus_data(url=args.url, cache_dir=args.cache_dir)
    raw_df = load_focus_data(str(path))
    print(f"  Rows: {len(raw_df):,}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    group_by = [g.strip() for g in args.group_by.split(",")]
    daily_dict = aggregate_daily(raw_df, group_by=group_by)
    print(f"  Service groups: {len(daily_dict)}")

    # ── Detect ────────────────────────────────────────────────────────────────
    alert_rows = []
    summary_rows = []

    for label, series in daily_dict.items():
        z_scores, alerts = rolling_zscore(series, window=args.window, sigma=args.sigma)

        flagged = series[alerts]
        n_alerts = int(alerts.sum())
        total_excess = float((series[alerts] - series.rolling(args.window, min_periods=3).mean()[alerts]).sum())

        summary_rows.append({
            "service": label,
            "days_observed": len(series),
            "mean_daily_cost": round(series.mean(), 4),
            "n_alerts": n_alerts,
            "alert_rate": round(n_alerts / len(series), 3),
            "flagged_dates": ";".join(str(d.date()) for d in flagged.index),
        })

        for dt, cost in flagged.items():
            alert_rows.append({
                "service": label,
                "date": dt.date(),
                "cost": round(cost, 6),
                "z_score": round(float(z_scores[dt]), 3),
            })

        if n_alerts:
            print(f"  {label}: {n_alerts} alert(s) on {[str(d.date()) for d in flagged.index]}")

    # ── Save ──────────────────────────────────────────────────────────────────
    alerts_path = output_dir / "focus_unsupervised_alerts.csv"
    summary_path = output_dir / "focus_unsupervised_summary.csv"
    pd.DataFrame(alert_rows).to_csv(alerts_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nDone.  Alerts: {alerts_path}")
    print(f"       Summary: {summary_path}")

    # Print summary table
    print("\n── Per-service summary ──")
    summary_df = pd.DataFrame(summary_rows)[["service", "days_observed", "mean_daily_cost", "n_alerts", "alert_rate"]]
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unsupervised anomaly detection on real FOCUS data (Approach B)"
    )
    parser.add_argument("--url", default=FOCUS_DATA_URL)
    parser.add_argument("--cache-dir", default=FOCUS_CACHE_DIR)
    parser.add_argument(
        "--group-by",
        default=",".join(FOCUS_GROUP_BY),
        help="Comma-separated FOCUS columns to group by",
    )
    parser.add_argument("--window", type=int, default=7, help="Rolling window size in days")
    parser.add_argument("--sigma", type=float, default=2.5, help="Alert threshold in std deviations")
    main(parser.parse_args())
