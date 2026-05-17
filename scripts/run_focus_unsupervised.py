"""CLI: run unsupervised anomaly detection directly on real FOCUS daily cost series.

Approach B - real-data sanity check:
  1. Download FOCUS billing data and aggregate to daily cost per service.
  2. Apply a look-ahead-free 7-day rolling z-score detector.
  3. Report flagged dates and a per-service alert summary CSV.

Because each FOCUS sample covers ~30 days, this is a sanity check rather than
a rigorous benchmark.  No precision/recall/F1/AUPRC/MCTD are computed.

Output files (outputs/results/):
  focus_unsupervised_alerts.csv    one row per flagged date per service
  focus_unsupervised_summary.csv   per-service alert counts and flagged dates

Usage
-----
  python scripts/run_focus_unsupervised.py
  python scripts/run_focus_unsupervised.py --sigma 2.0 --window 5
"""

import argparse
import sys
from pathlib import Path

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
    """Return z-scores and boolean alert flags using a look-ahead-free rolling window.

    The current day is compared against a baseline computed from the *previous*
    ``window`` days only (via ``shift(1)``), preventing data leakage.
    """
    baseline = series.shift(1)   # exclude today from its own baseline
    rm = baseline.rolling(window, min_periods=3).mean()
    rs = baseline.rolling(window, min_periods=3).std().clip(lower=1e-9)
    z = (series - rm) / rs
    return z, z.abs() > sigma


def main(args: argparse.Namespace) -> None:
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download & load
    print("Downloading FOCUS sample data...")
    path = download_focus_data(url=args.url, cache_dir=args.cache_dir)
    raw_df = load_focus_data(str(path))
    print(f"  Rows: {len(raw_df):,}")

    # Aggregate
    group_by = [g.strip() for g in args.group_by.split(",")]
    daily_dict = aggregate_daily(raw_df, group_by=group_by)
    print(f"  Service groups (after filtering): {len(daily_dict)}")

    if not daily_dict:
        print("No groups survived filtering. Try --group-by with broader columns.")
        return

    # Detect
    alert_rows = []
    summary_rows = []

    for label, series in daily_dict.items():
        z_scores, alerts = rolling_zscore(series, window=args.window, sigma=args.sigma)
        flagged = series[alerts]
        n_alerts = int(alerts.sum())

        # Top anomaly dates by absolute z-score magnitude
        top_dates = (
            z_scores[alerts].abs().nlargest(3).index
            if n_alerts > 0
            else pd.DatetimeIndex([])
        )

        summary_rows.append({
            "service": label,
            "days_observed": len(series),
            "nonzero_days": int((series > 0).sum()),
            "mean_daily_cost": round(float(series.mean()), 4),
            "alert_count": n_alerts,
            "alert_rate": round(n_alerts / max(len(series), 1), 3),
            "flagged_dates": ";".join(str(d.date()) for d in flagged.index),
            "top_anomaly_dates": ";".join(str(d.date()) for d in top_dates),
        })

        for dt, cost in flagged.items():
            alert_rows.append({
                "service": label,
                "date": dt.date(),
                "cost": round(float(cost), 6),
                "z_score": round(float(z_scores[dt]), 3),
            })

        if n_alerts:
            print(
                f"  {label}: {n_alerts} alert(s) - "
                f"{[str(d.date()) for d in flagged.index]}"
            )

    # Save
    alerts_path = output_dir / "focus_unsupervised_alerts.csv"
    summary_path = output_dir / "focus_unsupervised_summary.csv"
    pd.DataFrame(alert_rows).to_csv(alerts_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nDone.  Alerts:  {alerts_path}")
    print(f"       Summary: {summary_path}")

    # Print summary table
    cols = ["service", "days_observed", "mean_daily_cost", "alert_count", "alert_rate"]
    print("\n-- Per-service summary --")
    print(pd.DataFrame(summary_rows)[cols].to_string(index=False))


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
    parser.add_argument("--sigma", type=float, default=2.5, help="Alert threshold (std deviations)")
    main(parser.parse_args())
