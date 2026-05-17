"""CLI: run the full anomaly benchmark using FOCUS-calibrated synthetic data.

Approach A - calibrated synthetic:
  1. Download real FOCUS billing data (one month, multi-cloud).
  2. Aggregate to daily cost per service/category group.
  3. Extract per-group statistics (trend, noise, DoW pattern).
  4. Generate 730-day synthetic baselines with those real-world parameters.
  5. Inject synthetic anomalies, preserving ground-truth labels.
  6. Run EWMA / IsolationForest / Prophet / LSTM-AE across seeds & budgets.
  7. Save per-service metrics and cross-service summary tables.

Output files (all in outputs/results/):
  focus_calibration_stats.csv          calibration parameters per service
  focus_metrics_<service>.csv          raw metrics per service (all seeds/budgets)
  focus_events_<service>.csv           event-level results per service
  focus_benchmark_summary.csv          one-row-per-service with F1/AUPRC/etc.
  focus_core_metrics_by_service.csv    mean +/- std of key metrics per service x model
  focus_rank_reversal_by_service.csv   model rank per service (detect rank reversals)
  focus_overall_model_ranking.csv      aggregated model ranking across all services

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
from finops_benchmark.experiment import (
    build_rank_comparison,
    build_summary_metrics,
    run_one_seed_focus,
)
from finops_benchmark.focus_calibration import (
    calibrate_all_services,
    compute_global_stats,
    save_calibration_stats,
)
from finops_benchmark.focus_loader import aggregate_daily, download_focus_data, load_focus_data

_EVAL_BUDGET = 0.01   # primary budget for summary tables
_SUMMARY_METRICS = [
    "f1", "auprc", "cost_weighted_recall",
    "mean_mctd", "alert_cost_efficiency", "false_alarm_rate",
]


def _safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_").replace(",", "_")


def _build_rank_reversal_table(
    per_service_metrics: dict,
    budget: float = _EVAL_BUDGET,
) -> pd.DataFrame:
    """Build a table showing model rank per service to surface rank reversals."""
    rows = []
    for service, metrics_df in per_service_metrics.items():
        sub = metrics_df[metrics_df["budget"] == budget]
        f1_means = sub.groupby("model_name")["f1"].mean()
        ranks = f1_means.rank(ascending=False, method="min").astype(int)
        for model in f1_means.index:
            rows.append({
                "service": service,
                "model_name": model,
                "f1_mean": round(float(f1_means[model]), 4),
                "f1_rank": int(ranks[model]),
            })
    return pd.DataFrame(rows)


def main(args: argparse.Namespace) -> None:
    output_dir = Path(RESULTS_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Download & load
    print("Step 1/5  Downloading FOCUS sample data...")
    path = download_focus_data(url=args.url, cache_dir=args.cache_dir)
    print(f"  Loaded from {path}")
    raw_df = load_focus_data(str(path))
    print(f"  Rows: {len(raw_df):,}  |  Columns: {raw_df.shape[1]}")

    # 2. Aggregate
    group_by = [g.strip() for g in args.group_by.split(",")]
    print(f"\nStep 2/5  Aggregating daily cost by {group_by}...")
    daily_dict = aggregate_daily(raw_df, group_by=group_by)
    print(f"  Service groups (after filtering): {len(daily_dict)}")
    for lbl, s in daily_dict.items():
        print(
            f"    {lbl:55s}  days={len(s):3d}  "
            f"nonzero={int((s > 0).sum()):3d}  mean=${s.mean():,.2f}"
        )

    if not daily_dict:
        print("\nNo service groups survived filtering. Try lowering --min-days or --min-cost.")
        return

    # 3. Calibrate
    print("\nStep 3/5  Calibrating statistics from real data...")
    global_stats = compute_global_stats(daily_dict)
    print(
        f"  Global fallback - base=${global_stats['base_level']:,.2f}  "
        f"growth={global_stats['monthly_growth']:+.3f}  "
        f"noise={global_stats['noise_pct']:.3f}"
    )
    stats_all = calibrate_all_services(daily_dict, global_stats=global_stats)
    print(f"  Calibrated {len(stats_all)} groups.")
    for lbl, st in stats_all.items():
        fb = " [FALLBACK]" if st["used_fallback"] else ""
        print(
            f"    {lbl:55s}  "
            f"base=${st['base_level']:,.2f}  "
            f"growth={st['monthly_growth']:+.3f}  "
            f"noise={st['noise_pct']:.3f}{fb}"
        )

    save_calibration_stats(
        stats_all, str(output_dir / "focus_calibration_stats.csv")
    )

    # 4. Run benchmark
    print(
        f"\nStep 4/5  Running benchmark "
        f"({args.n_seeds} seeds x {len(BUDGETS)} budgets x 4 models)..."
    )
    per_service_metrics: dict = {}
    summary_rows = []

    for label, stats in stats_all.items():
        print(f"\n  -- {label}")
        metrics_parts, events_parts = [], []

        for seed in SEEDS[: args.n_seeds]:
            m, e = run_one_seed_focus(
                seed=seed, stats=stats, budgets=BUDGETS, verbose=args.verbose
            )
            metrics_parts.append(m)
            events_parts.append(e)

        metrics_df = pd.concat(metrics_parts, ignore_index=True)
        events_df = pd.concat(events_parts, ignore_index=True)
        per_service_metrics[label] = metrics_df

        # Per-service file output
        safe = _safe_label(label)
        metrics_df.to_csv(output_dir / f"focus_metrics_{safe}.csv", index=False)
        events_df[EVENT_KEEP_COLS].to_csv(
            output_dir / f"focus_events_{safe}.csv", index=False
        )

        # Summary row: mean of each metric at primary budget across seeds & models
        sub = metrics_df[metrics_df["budget"] == _EVAL_BUDGET]
        row = {
            "service": label,
            "days_observed": stats.get("days_observed", ""),
            "base_level": round(stats["base_level"], 2),
            "monthly_growth": round(stats["monthly_growth"], 4),
            "noise_pct": round(stats["noise_pct"], 4),
            "used_fallback": stats.get("used_fallback", False),
        }
        for metric in _SUMMARY_METRICS:
            if metric in sub.columns:
                row[f"{metric}_mean"] = round(float(sub[metric].mean()), 4)
        summary_rows.append(row)

        best_f1 = (
            sub.groupby("model_name")["f1"].mean().round(3).to_dict()
        )
        print(f"     F1 @ budget=1%: {best_f1}")

    # 5. Cross-service summary tables
    print("\nStep 5/5  Building cross-service summary tables...")

    # focus_benchmark_summary.csv
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "focus_benchmark_summary.csv", index=False
    )

    # focus_core_metrics_by_service.csv  (per-service x model mean +/- std)
    core_rows = []
    for service, metrics_df in per_service_metrics.items():
        summary = build_summary_metrics(metrics_df)
        summary.insert(0, "service", service)
        core_rows.append(summary)
    if core_rows:
        core_df = pd.concat(core_rows, ignore_index=True)
        core_df.to_csv(output_dir / "focus_core_metrics_by_service.csv", index=False)

    # focus_rank_reversal_by_service.csv
    rank_reversal_df = _build_rank_reversal_table(per_service_metrics)
    rank_reversal_df.to_csv(
        output_dir / "focus_rank_reversal_by_service.csv", index=False
    )

    # focus_overall_model_ranking.csv  (pool all services, equal weight)
    all_metrics_pooled = pd.concat(
        list(per_service_metrics.values()), ignore_index=True
    )
    overall_ranking = build_rank_comparison(all_metrics_pooled, budget=_EVAL_BUDGET)
    overall_ranking.to_csv(
        output_dir / "focus_overall_model_ranking.csv", index=False
    )
    print("\n  Overall model ranking (pooled across all services, budget=1%):")
    print(overall_ranking[["model_name", "f1", "f1_rank", "cost_weighted_recall"]].to_string(index=False))

    print(f"\nDone.  Results written to {output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FOCUS-calibrated anomaly benchmark (Approach A)"
    )
    parser.add_argument("--url", default=FOCUS_DATA_URL, help="FOCUS sample CSV URL")
    parser.add_argument("--cache-dir", default=FOCUS_CACHE_DIR, help="Local cache directory")
    parser.add_argument(
        "--group-by",
        default=",".join(FOCUS_GROUP_BY),
        help="Comma-separated FOCUS columns to group by (e.g. ProviderName,ServiceCategory)",
    )
    parser.add_argument("--n-seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
