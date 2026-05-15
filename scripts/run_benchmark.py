"""Run the full FinOps benchmark from the command line."""

import os
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from finops_benchmark.config import (
    BUDGETS,
    FIGS_DIR_NEW,
    RESULTS_DIR,
    SEEDS,
    ensure_results_dirs,
)
from finops_benchmark.experiment import (
    build_core_results_table,
    build_rank_comparison,
    build_summary_metrics,
    run_one_seed,
)
from finops_benchmark.visualization import (
    configure_paper_matplotlib,
    get_canonical_model_order,
    plot_paper_bar,
    plot_paper_line,
)


def main():
    results_dir, figures_dir = ensure_results_dirs(RESULTS_DIR, FIGS_DIR_NEW)

    all_metrics_parts = []
    all_events_parts = []
    for seed in tqdm(SEEDS, desc="seeds"):
        metrics_s, events_s = run_one_seed(seed=seed, verbose=False)
        all_metrics_parts.append(metrics_s)
        all_events_parts.append(events_s)

    all_model_metrics_df = pd.concat(all_metrics_parts, axis=0, ignore_index=True)
    all_event_results_df_full = pd.concat(all_events_parts, axis=0, ignore_index=True)

    from finops_benchmark.config import EVENT_KEEP_COLS
    all_event_results_df = all_event_results_df_full[EVENT_KEEP_COLS].copy()
    summary_metrics_df = build_summary_metrics(all_model_metrics_df)
    core_results_df = build_core_results_table(summary_metrics_df, budget=0.01)
    rank_comparison_df = build_rank_comparison(all_model_metrics_df, budget=0.01)

    all_model_metrics_df.to_csv(os.path.join(results_dir, "all_model_metrics.csv"), index=False)
    summary_metrics_df.to_csv(os.path.join(results_dir, "summary_metrics.csv"), index=False)
    all_event_results_df.to_csv(os.path.join(results_dir, "all_event_results.csv"), index=False)
    core_results_df.to_csv(os.path.join(results_dir, "paper_core_results_budget1pct.csv"), index=False)
    rank_comparison_df.to_csv(os.path.join(results_dir, "paper_rank_comparison_budget1pct.csv"), index=False)

    configure_paper_matplotlib()
    canonical_order = get_canonical_model_order(summary_metrics_df, budget=0.01)
    plot_paper_bar(summary_metrics_df, "f1", os.path.join(figures_dir, "paper_f1_bar_budget1pct.png"),
                   ylabel="F1 score", title=f"F1 at budget=1% (mean +/- std over {len(SEEDS)} seeds)",
                   model_order=canonical_order)
    plot_paper_bar(summary_metrics_df, "cost_weighted_recall", os.path.join(figures_dir, "paper_dollar_recall_bar_budget1pct.png"),
                   ylabel="$-Recall (cost-weighted recall)", title=f"Cost-weighted recall at budget=1% (mean +/- std over {len(SEEDS)} seeds)",
                   model_order=canonical_order)
    plot_paper_bar(summary_metrics_df, "alert_cost_efficiency", os.path.join(figures_dir, "paper_ace_bar_budget1pct.png"),
                   ylabel="Alert Cost Efficiency ($ per alert)", title=f"Alert Cost Efficiency at budget=1% (mean +/- std over {len(SEEDS)} seeds)",
                   model_order=canonical_order)
    plot_paper_line(summary_metrics_df, "f1", os.path.join(figures_dir, "paper_f1_by_budget.png"),
                    ylabel="F1 score", title="F1 across alert budgets")
    plot_paper_line(summary_metrics_df, "mean_mctd", os.path.join(figures_dir, "paper_mctd_by_budget.png"),
                    ylabel="Mean MCTD ($)", title="Mean Cost-To-Detect across alert budgets", lower_is_better=True)
    plot_paper_line(summary_metrics_df, "false_alarm_rate", os.path.join(figures_dir, "paper_far_by_budget.png"),
                    ylabel="False Alarm Rate", title="False Alarm Rate across alert budgets", lower_is_better=True)

    print("saved results to", results_dir)
    print("saved figures to", figures_dir)


if __name__ == "__main__":
    main()
