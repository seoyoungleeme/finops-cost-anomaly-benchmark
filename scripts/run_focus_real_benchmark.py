"""CLI: run the anomaly benchmark on real FOCUS billing data (Approach B).

Unlike the calibrated-synthetic approach, this script uses the actual FOCUS daily
cost series as the normal baseline and injects only the anomaly events synthetically.
This preserves real-world spend patterns (holidays, weekends, service bursts) while
maintaining ground-truth labels for precision/recall/cost-aware evaluation.

Data flow
---------
  1. Download real FOCUS billing CSV.
  2. Sum ALL services into a single total daily cost series.
  3. Split: first split_ratio of days = Year 1 (training), rest = Year 2 (eval).
  4. Inject synthetic spike / contextual / gradual anomalies into Year 2 only.
  5. Run EWMA / SeasonalNaiveMAD / IsolationForest / Prophet / LSTM-AE across seeds & budgets.
  6. Save results.

Key difference from run_focus_benchmark.py (Approach A)
--------------------------------------------------------
  Approach A: FOCUS stats  → synthetic 730-day series  (baseline is fake)
  Approach B: FOCUS series → inject anomalies on top   (baseline is REAL)

Output files (all in outputs/results/)
---------------------------------------
  focus_real_data_info.csv          date range, n_days, split point, cost stats
  focus_real_all_metrics.csv        raw metrics (all seeds x budgets x models)
  focus_real_summary_metrics.csv    mean +/- std per (budget, model)
  focus_real_core_results.csv       paper-ready table at year-1 FAR 1%
  focus_real_rank_comparison.csv    rank table (F1 vs cost-weighted metrics)
  focus_real_event_results.csv      event-level detection results

Usage
-----
  python scripts/run_focus_real_benchmark.py
  python scripts/run_focus_real_benchmark.py --n-seeds 5 --split-ratio 0.6
  python scripts/run_focus_real_benchmark.py --url <custom-url>
"""

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finops_benchmark.config import (
    BUDGETS,
    FOCUS_CACHE_DIR,
    FOCUS_DATA_URL,
    FOCUS_REAL_N_EVENTS_HIGH,
    FOCUS_REAL_N_EVENTS_LOW,
    FOCUS_REAL_SPLIT_RATIO,
    RESULTS_DIR,
    SEEDS,
)
from finops_benchmark.experiment import (
    build_core_results_table,
    build_rank_comparison,
    build_summary_metrics,
    run_multi_seed_focus_real,
)
from finops_benchmark.focus_loader import (
    aggregate_total_daily,
    download_focus_data,
    load_focus_data,
)

_EVAL_FAR_TARGET = 0.01


def _print_series_info(series: pd.Series, year2_start: int) -> dict:
    """Print and return summary statistics for the FOCUS daily series."""
    n = len(series)
    year2_days = n - year2_start
    info = {
        "start_date": str(series.index[0].date()),
        "end_date": str(series.index[-1].date()),
        "n_days_total": n,
        "year1_days": year2_start,
        "year2_days": year2_days,
        "split_ratio": round(year2_start / n, 3),
        "mean_daily_cost": round(float(series.mean()), 2),
        "std_daily_cost": round(float(series.std()), 2),
        "min_daily_cost": round(float(series.min()), 2),
        "max_daily_cost": round(float(series.max()), 2),
        "total_cost": round(float(series.sum()), 2),
    }
    print(f"\n  Date range  : {info['start_date']} ~ {info['end_date']}")
    print(f"  Total days  : {n}  (Year1={year2_start}, Year2={year2_days})")
    print(f"  Daily cost  : mean=${info['mean_daily_cost']:,.2f}  "
          f"std=${info['std_daily_cost']:,.2f}  "
          f"max=${info['max_daily_cost']:,.2f}")
    return info


def main(args: argparse.Namespace) -> None:
    t_start = time.time()
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download & load
    print("Step 1/5  Downloading FOCUS sample data...")
    path = download_focus_data(url=args.url, cache_dir=args.cache_dir)
    print(f"  Loaded from {path}")
    raw_df = load_focus_data(str(path))
    print(f"  Rows: {len(raw_df):,}  |  Columns: {raw_df.shape[1]}")

    # Step 2: Aggregate to single total daily series
    print("\nStep 2/5  Aggregating all services to total daily series...")
    try:
        series = aggregate_total_daily(raw_df, min_days=args.min_days)
    except ValueError as exc:
        print(f"\n  ERROR: {exc}")
        sys.exit(1)

    year2_start = int(len(series) * args.split_ratio)
    info = _print_series_info(series, year2_start)

    # Save data info
    pd.DataFrame([info]).to_csv(
        output_dir / "focus_real_data_info.csv", index=False
    )

    if info["year2_days"] < 10:
        print(
            f"\n  ERROR: Year 2 has only {info['year2_days']} days — too short for evaluation. "
            "Use longer FOCUS data or lower --split-ratio."
        )
        sys.exit(1)
    if info["n_days_total"] < 60:
        print(
            f"\n  WARNING: Only {info['n_days_total']} days of FOCUS data available. "
            "Results will have high variance. For reliable results, use 90+ days of real billing data.\n"
            "  The FOCUS GitHub sample covers only 30 days (demo data).\n"
            "  To use your own data: export FOCUS-format billing CSV and pass it via --url or "
            "place it in .focus_cache/ ."
        )

    # Step 3: Run multi-seed benchmark
    seeds = SEEDS[: args.n_seeds]
    print(
        f"\nStep 3/5  Running real-FOCUS benchmark "
        f"({len(seeds)} seeds x {len(BUDGETS)} budgets x 5 models)..."
    )
    print(f"  Baseline  : REAL FOCUS daily cost series (not synthetic)")
    print(f"  Anomalies : synthetic injection into Year 2 ({info['year2_days']} days)")
    print(f"  Events/type-intensity : {args.n_events_low}~{args.n_events_high} per seed")

    all_metrics, all_events = run_multi_seed_focus_real(
        seeds=seeds,
        series=series,
        split_ratio=args.split_ratio,
        budgets=BUDGETS,
        n_events_low=args.n_events_low,
        n_events_high=args.n_events_high,
    )

    # Step 4: Build summary tables
    print("\nStep 4/5  Building result tables...")
    all_metrics.to_csv(output_dir / "focus_real_all_metrics.csv", index=False)
    all_events.to_csv(output_dir / "focus_real_event_results.csv", index=False)

    summary = build_summary_metrics(all_metrics)
    summary.to_csv(output_dir / "focus_real_summary_metrics.csv", index=False)

    core = build_core_results_table(summary, year1_fpr_target=_EVAL_FAR_TARGET)
    core.to_csv(output_dir / "focus_real_core_results.csv", index=False)

    rank = build_rank_comparison(all_metrics, year1_fpr_target=_EVAL_FAR_TARGET)
    rank.to_csv(output_dir / "focus_real_rank_comparison.csv", index=False)

    # Step 5: Print summary
    print("\nStep 5/5  Results (year-1 FAR target=1%, real FOCUS baseline):")
    print("\n  Core metrics (mean ± std across seeds):")
    print(core.to_string(index=False))
    print("\n  Model rank comparison:")
    print(rank[["model_name", "f1", "f1_rank", "cost_weighted_recall",
                "dollar_recall_rank", "alert_cost_efficiency", "ace_rank"]].to_string(index=False))

    elapsed = round(time.time() - t_start, 1)
    print(f"\nDone.  Results written to {output_dir}/  ({elapsed}s)")
    print(
        "\n  NOTE: Baseline is REAL FOCUS data. "
        "Anomalies injected synthetically - see focus_real_event_results.csv for event details."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-FOCUS anomaly benchmark (Approach B: real baseline + synthetic anomalies)"
    )
    parser.add_argument("--url", default=FOCUS_DATA_URL, help="FOCUS sample CSV URL")
    parser.add_argument("--cache-dir", default=FOCUS_CACHE_DIR, help="Local cache directory")
    parser.add_argument(
        "--split-ratio", type=float, default=FOCUS_REAL_SPLIT_RATIO,
        help="Fraction of days used as Year 1 (training). Default: 0.6"
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of random seeds for anomaly injection"
    )
    parser.add_argument(
        "--n-events-low", type=int, default=FOCUS_REAL_N_EVENTS_LOW,
        help="Min events per (type, intensity) per seed"
    )
    parser.add_argument(
        "--n-events-high", type=int, default=FOCUS_REAL_N_EVENTS_HIGH,
        help="Max events per (type, intensity) per seed"
    )
    parser.add_argument(
        "--min-days", type=int, default=30,
        help="Minimum total days required in FOCUS data"
    )
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
