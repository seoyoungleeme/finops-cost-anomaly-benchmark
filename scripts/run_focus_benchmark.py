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
  focus_calibration_summary.csv        cross-service summary stats of calibration parameters
  focus_metrics_<service>.csv          raw metrics per service (all seeds/budgets)
  focus_events_<service>.csv           event-level results per service
  focus_service_summary.csv            one-row-per-service averaged across models & seeds
  focus_core_metrics_by_service.csv    mean +/- std of key metrics per service x model
  focus_rank_reversal_by_service.csv   model rank per service for 4 metrics (detect reversals)
  focus_rank_reversal_summary.csv      model-pair rank reversal rates across metrics & services
  focus_overall_model_ranking.csv      aggregated model ranking across all services
  focus_run_metadata.json              run metadata for research reproducibility

Usage
-----
  python scripts/run_focus_benchmark.py
  python scripts/run_focus_benchmark.py --n-seeds 3 --group-by ProviderName,ServiceName
"""

import argparse
import importlib
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from finops_benchmark.config import (
    BUDGETS,
    EVENT_KEEP_COLS,
    FOCUS_CACHE_DIR,
    FOCUS_DATA_URL,
    FOCUS_GROUP_BY,
    N_DAYS,
    RESULTS_DIR,
    SEEDS,
    YEAR2_START,
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
from finops_benchmark.visualization import (
    configure_paper_matplotlib,
    get_canonical_model_order,
    plot_focus_detection_by_type,
    plot_focus_detection_heatmap,
    plot_focus_metric_overview,
    plot_focus_mctd_by_type,
    plot_focus_radar,
    plot_focus_rank_slope,
    plot_paper_bar,
    plot_paper_line,
)

_EVAL_FAR_TARGET = 0.01   # primary year-1 FAR target for summary tables
_SUMMARY_METRICS = [
    "f1", "auprc", "cost_weighted_recall",
    "mean_mctd", "alert_cost_efficiency", "false_alarm_rate",
]

# Rank metrics: (column_in_rank_reversal_df, ascending)
_RANK_COL_MAP = {
    "f1": "f1_rank",
    "cost_weighted_recall": "dollar_recall_rank",
    "alert_cost_efficiency": "ace_rank",
    "mean_mctd": "mctd_rank",
}


def _safe_label(label: str) -> str:
    return label.replace("/", "_").replace(" ", "_").replace(",", "_")


def _get_git_commit() -> str:
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _collect_library_versions() -> dict:
    versions: dict = {"python": platform.python_version()}
    for lib in ["numpy", "pandas", "scipy", "prophet", "sklearn", "torch"]:
        try:
            mod = importlib.import_module(lib)
            versions[lib] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pass
    return versions


def _build_calibration_summary(stats_all: dict) -> pd.DataFrame:
    """Cross-service summary statistics for calibration parameters."""
    import numpy as np

    scalar_keys = ["base_level", "monthly_growth", "noise_pct", "days_observed", "nonzero_days"]
    rows = []
    for key in scalar_keys:
        vals = [s[key] for s in stats_all.values() if key in s and s[key] != ""]
        if not vals:
            continue
        arr = pd.array(vals, dtype=float)
        rows.append({
            "parameter": key,
            "n_services": len(arr),
            "mean": round(float(np.mean(arr)), 4),
            "median": round(float(np.median(arr)), 4),
            "std": round(float(np.std(arr)), 4),
            "min": round(float(np.min(arr)), 4),
            "max": round(float(np.max(arr)), 4),
        })
    n_total = len(stats_all)
    n_fallback = sum(1 for s in stats_all.values() if s.get("used_fallback", False))
    rows.append({
        "parameter": "used_fallback",
        "n_services": n_total,
        "mean": round(n_fallback / n_total, 4) if n_total else 0.0,
        "median": float(n_fallback > n_total / 2),
        "std": "",
        "min": 0,
        "max": 1,
    })
    return pd.DataFrame(rows)


def _build_rank_reversal_summary(rank_reversal_df: pd.DataFrame) -> pd.DataFrame:
    """Model-pair rank reversal rates: fraction of (service, metric-pair) combos where ranks swap."""
    rank_cols = list(_RANK_COL_MAP.values())           # [f1_rank, dollar_recall_rank, ace_rank, mctd_rank]
    metric_names = list(_RANK_COL_MAP.keys())
    models = sorted(rank_reversal_df["model_name"].unique())
    services = rank_reversal_df["service"].unique()

    # Pivot so each row is a service, columns are (model, rank_col)
    pivot = rank_reversal_df.pivot_table(
        index="service", columns="model_name", values=rank_cols, aggfunc="first"
    )

    metric_pairs = [
        (metric_names[i], rank_cols[i], metric_names[j], rank_cols[j])
        for i in range(len(rank_cols))
        for j in range(i + 1, len(rank_cols))
    ]

    rows = []
    for i, ma in enumerate(models):
        for mb in models[i + 1:]:
            n_reversals = 0
            n_total = 0
            detail = {}
            for m1_name, c1, m2_name, c2 in metric_pairs:
                pair_key = f"{m1_name}_vs_{m2_name}"
                rev_count = 0
                svc_count = 0
                for svc in services:
                    try:
                        ra1 = int(pivot.loc[svc, (c1, ma)])
                        rb1 = int(pivot.loc[svc, (c1, mb)])
                        ra2 = int(pivot.loc[svc, (c2, ma)])
                        rb2 = int(pivot.loc[svc, (c2, mb)])
                    except (KeyError, ValueError):
                        continue
                    svc_count += 1
                    if (ra1 < rb1 and ra2 > rb2) or (ra1 > rb1 and ra2 < rb2):
                        rev_count += 1
                n_reversals += rev_count
                n_total += svc_count
                detail[pair_key] = f"{rev_count}/{svc_count}"

            rows.append({
                "model_a": ma,
                "model_b": mb,
                "n_services": len(services),
                "n_metric_pairs": len(metric_pairs),
                "n_reversal_instances": n_reversals,
                "total_instances": n_total,
                "reversal_rate": round(n_reversals / n_total, 4) if n_total else 0.0,
                **detail,
            })

    if rows:
        all_rev = sum(r["n_reversal_instances"] for r in rows)
        all_total = sum(r["total_instances"] for r in rows)
        rows.append({
            "model_a": "ALL_PAIRS",
            "model_b": "(all)",
            "n_services": len(services),
            "n_metric_pairs": len(metric_pairs),
            "n_reversal_instances": all_rev,
            "total_instances": all_total,
            "reversal_rate": round(all_rev / all_total, 4) if all_total else 0.0,
        })

    return pd.DataFrame(rows)


def _build_anomaly_type_tables(
    per_service_events: dict,
    year1_fpr_target: float = _EVAL_FAR_TARGET,
):
    """Build detection metrics broken down by anomaly_type (and type×intensity).

    Returns
    -------
    type_df      : per model × anomaly_type
    intensity_df : per model × anomaly_type × intensity_level
    """
    import numpy as np

    all_ev = pd.concat(list(per_service_events.values()), ignore_index=True)
    sub = all_ev[all_ev["year1_fpr_target"] == year1_fpr_target].copy()

    def _agg(grp):
        n = len(grp)
        det_rate = float(grp["detected"].mean())
        detected = grp[grp["detected"]]
        mean_delay = float(detected["detection_delay"].mean()) if len(detected) else float("nan")
        mean_mctd = float(grp["mctd"].mean())
        tot_cost = float(grp["total_excess_cost"].sum())
        det_cost = float(grp.loc[grp["detected"], "total_excess_cost"].sum())
        dollar_recall = det_cost / tot_cost if tot_cost > 0 else 0.0
        return pd.Series({
            "n_events": n,
            "detection_rate": round(det_rate, 4),
            "mean_detection_delay": round(mean_delay, 2),
            "mean_mctd": round(mean_mctd, 4),
            "dollar_recall": round(dollar_recall, 4),
        })

    type_df = (
        sub.groupby(["model_name", "anomaly_type"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    # Sort for readability
    type_order = {"spike": 0, "contextual": 1, "gradual": 2}
    type_df["_t"] = type_df["anomaly_type"].map(type_order)
    type_df = type_df.sort_values(["_t", "model_name"]).drop("_t", axis=1).reset_index(drop=True)

    intensity_df = (
        sub.groupby(["model_name", "anomaly_type", "intensity_level"])
        .apply(_agg, include_groups=False)
        .reset_index()
    )
    intensity_order = {"low": 0, "mid": 1, "high": 2}
    intensity_df["_t"] = intensity_df["anomaly_type"].map(type_order)
    intensity_df["_i"] = intensity_df["intensity_level"].map(intensity_order)
    intensity_df = (intensity_df
                    .sort_values(["_t", "_i", "model_name"])
                    .drop(["_t", "_i"], axis=1)
                    .reset_index(drop=True))

    return type_df, intensity_df


def _build_rank_reversal_table(
    per_service_metrics: dict,
    year1_fpr_target: float = _EVAL_FAR_TARGET,
) -> pd.DataFrame:
    """Build per-service model ranks across 4 metrics to surface rank reversals.

    Ranking directions:
      f1, cost_weighted_recall, alert_cost_efficiency : higher is better (rank 1 = highest)
      mean_mctd                                       : lower is better  (rank 1 = lowest)
    """
    _RANK_METRICS = {
        "f1": (False, "f1_rank"),                              # ascending=False → higher wins
        "cost_weighted_recall": (False, "dollar_recall_rank"),
        "alert_cost_efficiency": (False, "ace_rank"),
        "mean_mctd": (True, "mctd_rank"),                      # ascending=True  → lower wins
    }

    rows = []
    for service, metrics_df in per_service_metrics.items():
        sub = metrics_df[metrics_df["year1_fpr_target"] == year1_fpr_target]
        means = sub.groupby("model_name")[list(_RANK_METRICS)].mean()

        for model in means.index:
            row: dict = {"service": service, "model_name": model}
            for metric, (asc, rank_col) in _RANK_METRICS.items():
                val = float(means.loc[model, metric])
                rank = int(means[metric].rank(ascending=asc, method="min")[model])
                row[f"{metric}_mean"] = round(val, 4)
                row[rank_col] = rank
            rows.append(row)

    col_order = [
        "service", "model_name",
        "f1_mean", "f1_rank",
        "cost_weighted_recall_mean", "dollar_recall_rank",
        "alert_cost_efficiency_mean", "ace_rank",
        "mean_mctd_mean", "mctd_rank",
    ]
    df = pd.DataFrame(rows)
    return df[col_order] if not df.empty else df


def main(args: argparse.Namespace) -> None:
    t_start = time.time()
    run_ts = datetime.now(timezone.utc).isoformat()

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
    daily_dict = aggregate_daily(
        raw_df,
        group_by=group_by,
        min_days=args.min_days,
        min_nonzero_days=args.min_nonzero_days,
        min_mean_cost=args.min_mean_cost,
    )
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

        # Summary row: mean of each metric at primary year-1 FAR target across seeds & models
        sub = metrics_df[metrics_df["year1_fpr_target"] == _EVAL_FAR_TARGET]
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
        print(f"     F1 @ year-1 FAR target=1%: {best_f1}")

    # 5. Cross-service summary tables
    print("\nStep 5/5  Building cross-service summary tables...")

    # focus_service_summary.csv — one row per service, averaged across models & seeds
    pd.DataFrame(summary_rows).to_csv(
        output_dir / "focus_service_summary.csv", index=False
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
    overall_ranking = build_rank_comparison(all_metrics_pooled, year1_fpr_target=_EVAL_FAR_TARGET)
    overall_ranking.to_csv(
        output_dir / "focus_overall_model_ranking.csv", index=False
    )
    print("\n  Overall model ranking (pooled across all services, year-1 FAR target=1%):")
    print(overall_ranking[["model_name", "f1", "f1_rank", "cost_weighted_recall"]].to_string(index=False))

    # focus_calibration_summary.csv
    cal_summary = _build_calibration_summary(stats_all)
    cal_summary.to_csv(output_dir / "focus_calibration_summary.csv", index=False)
    print("\n  Calibration summary (cross-service):")
    print(cal_summary.to_string(index=False))

    # focus_rank_reversal_summary.csv
    rr_summary = _build_rank_reversal_summary(rank_reversal_df)
    rr_summary.to_csv(output_dir / "focus_rank_reversal_summary.csv", index=False)
    print("\n  Rank reversal rates (model pairs across metrics & services):")
    print(rr_summary[["model_a", "model_b", "reversal_rate", "n_reversal_instances", "total_instances"]].to_string(index=False))

    # focus_anomaly_type_results.csv + focus_anomaly_intensity_results.csv
    per_service_events = {}
    for label in per_service_metrics:
        safe = _safe_label(label)
        ev_path = output_dir / f"focus_events_{safe}.csv"
        if ev_path.exists():
            per_service_events[label] = pd.read_csv(ev_path)

    if per_service_events:
        type_df, intensity_df = _build_anomaly_type_tables(per_service_events)
        type_df.to_csv(output_dir / "focus_anomaly_type_results.csv", index=False)
        intensity_df.to_csv(output_dir / "focus_anomaly_intensity_results.csv", index=False)
        print("\n  Detection by anomaly type (year-1 FAR target=1%):")
        pivot_preview = (type_df
                         .pivot_table(index="model_name", columns="anomaly_type",
                                      values="detection_rate")
                         .round(3))
        print(pivot_preview.to_string())

    # Step 6: Figures
    print("\nStep 6/6  Generating figures...")
    fig_dir = Path(RESULTS_DIR).parent / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    configure_paper_matplotlib()

    # Pool events across services for figure generation
    all_events_for_figs = (
        pd.concat(list(per_service_events.values()), ignore_index=True)
        if per_service_events else pd.DataFrame()
    )

    # Build core_metrics_df pooled across services
    if core_rows:
        core_df_for_figs = pd.concat(core_rows, ignore_index=True)
    else:
        core_df_for_figs = pd.DataFrame()

    canonical_order = (
        get_canonical_model_order(core_df_for_figs)
        if not core_df_for_figs.empty else None
    )

    fig_specs = []

    # Standard metric bars
    for metric, ylabel, fname in [
        ("f1",                   "F1 Score",                 "focus_f1_bar.png"),
        ("cost_weighted_recall", "Dollar Recall",            "focus_dollar_recall_bar.png"),
        ("alert_cost_efficiency","Alert Cost Efficiency ($)", "focus_ace_bar.png"),
        ("mean_mctd",            "Mean MCTD ($)",            "focus_mctd_bar.png"),
    ]:
        if not core_df_for_figs.empty:
            plot_paper_bar(
                core_df_for_figs, metric,
                str(fig_dir / fname),
                year1_fpr_target=_EVAL_FAR_TARGET,
                ylabel=ylabel,
                model_order=canonical_order,
            )
            fig_specs.append(fname)

    # Standard metric lines (across budgets)
    for metric, ylabel, fname, lower in [
        ("f1",            "F1 Score",            "focus_f1_by_budget.png",   False),
        ("mean_mctd",     "Mean MCTD ($)",        "focus_mctd_by_budget.png", True),
        ("false_alarm_rate","False Alarm Rate",   "focus_far_by_budget.png",  True),
    ]:
        if not core_df_for_figs.empty:
            plot_paper_line(
                core_df_for_figs, metric,
                str(fig_dir / fname),
                ylabel=ylabel, lower_is_better=lower,
            )
            fig_specs.append(fname)

    # 2x2 metric overview
    if not core_df_for_figs.empty:
        plot_focus_metric_overview(
            core_df_for_figs, str(fig_dir / "focus_metric_overview.png")
        )
        fig_specs.append("focus_metric_overview.png")

    if not all_events_for_figs.empty:
        plot_focus_detection_by_type(
            all_events_for_figs, str(fig_dir / "focus_detection_by_type.png")
        )
        fig_specs.append("focus_detection_by_type.png")

        plot_focus_detection_heatmap(
            all_events_for_figs, str(fig_dir / "focus_detection_heatmap.png")
        )
        fig_specs.append("focus_detection_heatmap.png")

        plot_focus_mctd_by_type(
            all_events_for_figs, str(fig_dir / "focus_mctd_by_type.png")
        )
        fig_specs.append("focus_mctd_by_type.png")

    if not overall_ranking.empty:
        plot_focus_radar(overall_ranking, str(fig_dir / "focus_radar.png"))
        fig_specs.append("focus_radar.png")

    if not rank_reversal_df.empty:
        plot_focus_rank_slope(rank_reversal_df, str(fig_dir / "focus_rank_slope.png"))
        fig_specs.append("focus_rank_slope.png")

    print(f"  Saved {len(fig_specs)} figures to {fig_dir}/")
    for f in fig_specs:
        print(f"    {f}")

    # focus_run_metadata.json
    elapsed = round(time.time() - t_start, 1)
    metadata = {
        "run_timestamp_utc": run_ts,
        "git_commit": _get_git_commit(),
        "runtime_seconds": elapsed,
        "library_versions": _collect_library_versions(),
        "cli_args": vars(args),
        "config": {
            "N_DAYS": N_DAYS,
            "YEAR2_START": YEAR2_START,
            "SEEDS_USED": SEEDS[: args.n_seeds],
            "BUDGETS": BUDGETS,
            "FOCUS_GROUP_BY": FOCUS_GROUP_BY,
            "FOCUS_DATA_URL": FOCUS_DATA_URL,
            "eval_far_target": _EVAL_FAR_TARGET,
        },
        "n_service_groups": len(stats_all),
        "services": list(stats_all.keys()),
        "output_figures": fig_specs,
    }
    meta_path = output_dir / "focus_run_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"\n  Run metadata saved: {meta_path}")
    print(f"  Total runtime: {elapsed}s")

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
    parser.add_argument("--verbose", action="store_true")
    main(parser.parse_args())
