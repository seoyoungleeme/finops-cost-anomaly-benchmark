"""CLI: run the full anomaly benchmark using FOCUS-calibrated synthetic data.

Approach A — calibrated synthetic:
  1. Download real FOCUS billing data (one month, multi-cloud).
  2. Aggregate to daily cost per service/category group.
  3. Extract per-group statistics (trend, noise, DoW pattern).
  4. Generate 730-day synthetic baselines with those real-world parameters.
  5. Inject synthetic anomalies → ground-truth labels preserved.
  6. Run EWMA / IsolationForest / Prophet / LSTM-AE across seeds & budgets.
  7. Save per-service metrics and a cross-service summary CSV.

Usage
-----
  python scripts/run_focus_benchmark.py
  python scripts/run_focus_benchmark.py --n-seeds 3 --group-by ProviderName,ServiceName
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finops_benchmark.config import (
    BUDGETS,
    EVENT_KEEP_COLS,
    FOCUS_CACHE_DIR,
    FOCUS_DATA_URL,
    FOCUS_GROUP_BY,
    RESULTS_DIR,
    SEEDS,
)
from finops_benchmark.experiment import run_one_seed_focus
from finops_benchmark.focus_calibration import calibrate_all_services
from finops_benchmark.focus_loader import aggregate_daily, download_focus_data, load_focus_data


def _safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_").replace(",", "_")


def main(args: argparse.Namespace) -> None:
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Download & load ───────────────────────────────────────────────────
    print("Step 1/4  Downloading FOCUS sample data...")
    path = download_focus_data(url=args.url, cache_dir=args.cache_dir)
    print(f"  Loaded from {path}")
    raw_df = load_focus_data(str(path))
    print(f"  Rows: {len(raw_df):,}  |  Columns: {raw_df.shape[1]}")

    # ── 2. Aggregate ─────────────────────────────────────────────────────────
    group_by = [g.strip() for g in args.group_by.split(",")]
    print(f"\nStep 2/4  Aggregating daily cost by {group_by}...")
    daily_dict = aggregate_daily(raw_df, group_by=group_by)
    print(f"  Service groups found: {len(daily_dict)}")
    for lbl, s in daily_dict.items():
        print(f"    {lbl:55s}  days={len(s):3d}  mean=${s.mean():,.2f}")

    # ── 3. Calibrate ─────────────────────────────────────────────────────────
    print("\nStep 3/4  Calibrating statistics from real data...")
    stats_all = calibrate_all_services(daily_dict)
    print(f"  Calibrated {len(stats_all)} groups.")
    for lbl, st in stats_all.items():
        print(
            f"    {lbl:55s}  "
            f"base=${st['base_level']:,.2f}  "
            f"growth={st['monthly_growth']:+.3f}  "
            f"noise={st['noise_pct']:.3f}"
        )

    # ── 4. Run benchmark ─────────────────────────────────────────────────────
    print(f"\nStep 4/4  Running benchmark ({args.n_seeds} seeds × {len(BUDGETS)} budgets × 4 models)...")
    summary_rows = []

    for label, stats in stats_all.items():
        print(f"\n  ── {label}")
        metrics_parts, events_parts = [], []

        for seed in SEEDS[: args.n_seeds]:
            m, e = run_one_seed_focus(seed=seed, stats=stats, budgets=BUDGETS, verbose=args.verbose)
            metrics_parts.append(m)
            events_parts.append(e)

        metrics_df = pd.concat(metrics_parts, ignore_index=True)
        events_df = pd.concat(events_parts, ignore_index=True)

        # Print quick per-model F1 summary
        f1_summary = (
            metrics_df[metrics_df["budget"] == 0.01]
            .groupby("model_name")["f1"]
            .mean()
            .round(3)
        )
        print(f"     F1 @ budget=1%: {f1_summary.to_dict()}")

        # Save per-service files
        safe = _safe_label(label)
        metrics_df.to_csv(output_dir / f"focus_metrics_{safe}.csv", index=False)
        events_df[EVENT_KEEP_COLS].to_csv(output_dir / f"focus_events_{safe}.csv", index=False)

        row = {"service": label, **stats}
        for model, f1 in f1_summary.items():
            row[f"f1_{model}"] = f1
        summary_rows.append(row)

    # Cross-service summary
    summary_path = output_dir / "focus_benchmark_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"\nDone.  Results written to {output_dir}")
    print(f"  Cross-service summary: {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FOCUS-calibrated anomaly benchmark (Approach A)"
    )
    parser.add_argument("--url", default=FOCUS_DATA_URL, help="FOCUS sample CSV URL")
    parser.add_argument("--cache-dir", default=FOCUS_CACHE_DIR, help="Local cache directory")
    parser.add_argument(
        "--group-by",
        default=",".join(FOCUS_GROUP_BY),
        help="Comma-separated FOCUS columns to group by",
    )
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
