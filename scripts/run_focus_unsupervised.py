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
  python scripts/run_focus_unsupervised.py --min-days 14 --min-nonzero-days 7 --min-mean-cost 0.1
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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


def _safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_").replace(",", "_")


def _plot_case_study(
    service: str,
    series: pd.Series,
    z_scores: pd.Series,
    alerts: pd.Series,
    sigma: float,
    save_path: Path,
) -> None:
    """Save a raw FOCUS cost plot with rolling-z suspicious dates highlighted."""
    fig, (ax_cost, ax_z) = plt.subplots(
        2, 1, figsize=(11, 5.8), sharex=True,
        gridspec_kw={"height_ratios": [2.3, 1.0]},
    )

    ax_cost.plot(series.index, series.values, marker="o", lw=1.4, color="#2f6f9f")
    if alerts.any():
        flagged = series[alerts]
        ax_cost.scatter(
            flagged.index, flagged.values,
            s=48, color="#d62728", zorder=4, label="flagged date",
        )
        ax_cost.legend(loc="best", fontsize=9)
    ax_cost.set_title(f"Raw FOCUS Cost Sanity Check - {service}")
    ax_cost.set_ylabel("Daily cost")
    ax_cost.grid(alpha=0.3)

    ax_z.plot(z_scores.index, z_scores.values, marker="o", lw=1.0, color="#444444")
    ax_z.axhline(sigma, color="#d62728", ls="--", lw=0.9)
    ax_z.axhline(-sigma, color="#d62728", ls="--", lw=0.9)
    ax_z.set_ylabel("Rolling z")
    ax_z.set_xlabel("Billing date")
    ax_z.grid(alpha=0.3)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    daily_dict = aggregate_daily(
        raw_df,
        group_by=group_by,
        min_days=args.min_days,
        min_nonzero_days=args.min_nonzero_days,
        min_mean_cost=args.min_mean_cost,
    )
    print(f"  Service groups (after filtering): {len(daily_dict)}")

    if not daily_dict:
        print("No groups survived filtering. Try --group-by with broader columns.")
        return

    # Detect
    alert_rows = []
    summary_rows = []
    case_cache = {}

    for label, series in daily_dict.items():
        z_scores, alerts = rolling_zscore(series, window=args.window, sigma=args.sigma)
        case_cache[label] = (series, z_scores, alerts)
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
    alerts_path = output_dir / f"{args.output_prefix}_alerts.csv"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"
    pd.DataFrame(alert_rows).to_csv(alerts_path, index=False)
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)

    print(f"\nDone.  Alerts:  {alerts_path}")
    print(f"       Summary: {summary_path}")

    if args.figure_dir:
        figure_dir = Path(args.figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        case_services = args.case_study_service or [
            "AWS / Compute",
            "Microsoft / Networking",
        ]
        for service in case_services:
            if service not in case_cache:
                print(f"       Case study skipped (missing service): {service}")
                continue
            series, z_scores, alerts = case_cache[service]
            fname = f"{args.output_prefix}_case_{_safe_label(service)}.png"
            _plot_case_study(
                service, series, z_scores, alerts,
                sigma=args.sigma,
                save_path=figure_dir / fname,
            )
            print(f"       Case study figure: {figure_dir / fname}")

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
    parser.add_argument(
        "--min-days", type=int, default=21,
        help="Min calendar days a service group must have to pass filtering",
    )
    parser.add_argument(
        "--min-nonzero-days", type=int, default=14,
        help="Min days with cost > 0 a service group must have",
    )
    parser.add_argument(
        "--min-mean-cost", type=float, default=1.0,
        help="Min mean daily cost (in billing currency) for a service group",
    )
    parser.add_argument(
        "--output-prefix", default="focus_unsupervised",
        help="Prefix for alerts/summary CSV files under outputs/results",
    )
    parser.add_argument(
        "--figure-dir", default=None,
        help="Optional directory for raw FOCUS case-study figures",
    )
    parser.add_argument(
        "--case-study-service",
        action="append",
        default=None,
        help=(
            "Service label to plot as a raw-data case study. Can be repeated. "
            "Defaults to AWS / Compute and Microsoft / Networking when --figure-dir is set."
        ),
    )
    main(parser.parse_args())
