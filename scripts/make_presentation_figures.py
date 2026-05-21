"""
FinOps Cost Anomaly Benchmark - Presentation-Quality Visualization
Generates 7 key figures covering the main research findings.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

# -- output directory ---------------------------------------------------------
OUT_DIR = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "presentation_figures"
)
os.makedirs(OUT_DIR, exist_ok=True)

RES_STRICT = os.path.join(
    os.path.dirname(__file__), "..", "outputs", "results_full_strict"
)
RES = os.path.join(os.path.dirname(__file__), "..", "outputs", "results")

# -- colour palette (colourblind-safe; display order follows latest F1 rank) --
PALETTE = {
    "Prophet":          "#1B6CA8",
    "LSTM_AE":          "#2E9B5E",
    "IsolationForest":  "#E07B39",
    "SeasonalNaiveMAD": "#9B59B6",
    "EWMA":             "#C0392B",
}
MODEL_ORDER = ["Prophet", "IsolationForest", "LSTM_AE", "SeasonalNaiveMAD", "EWMA"]
MODEL_LABELS = {
    "Prophet":          "Prophet",
    "LSTM_AE":          "LSTM-AE",
    "IsolationForest":  "Isolation\nForest",
    "SeasonalNaiveMAD": "Seasonal\nNaiveMAD",
    "EWMA":             "EWMA",
}

PLT_STYLE = {
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.color": "#aaaaaa",
    "figure.dpi": 150,
}


def savefig(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved -> {path}")


# -----------------------------------------------------------------------------
# Figure 1 - Four-Metric Dashboard
# -----------------------------------------------------------------------------
def fig_dashboard():
    df = pd.read_csv(os.path.join(RES_STRICT, "focus_overall_model_ranking_with_std.csv"))
    df.index = df["model_name"]

    metrics = [
        ("f1_mean",                    "f1_std",                    "F1 Score",                   "higher = better"),
        ("cost_weighted_recall_mean",  "cost_weighted_recall_std",  "Cost-Weighted Recall\n(Dollar Recall)",  "higher = better"),
        ("mean_mctd_mean",             "mean_mctd_std",             "Mean Cost-to-Detect\n(MCTD, USD)",       "lower = better"),
        ("alert_cost_efficiency_mean", "alert_cost_efficiency_std", "Alert Cost Efficiency\n(USD / alert)",   "higher = better"),
    ]

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 4, figsize=(16, 5.5))
        fig.suptitle(
            "FinOps Cost Anomaly Detection - Model Performance Dashboard\n"
            "FOCUS-Calibrated Strict Benchmark  (FA target = 1%, seeds 0-2)",
            fontsize=13, fontweight="bold", y=1.01,
        )

        for ax, (col_m, col_s, title, note) in zip(axes, metrics):
            models = MODEL_ORDER
            vals   = [df.loc[m, col_m] for m in models]
            errs   = [df.loc[m, col_s] for m in models]
            colors = [PALETTE[m] for m in models]

            bars = ax.barh(
                range(len(models)), vals, xerr=errs,
                color=colors, alpha=0.88, edgecolor="white",
                error_kw=dict(ecolor="#555", capsize=3, lw=1.2),
            )
            ax.set_yticks(range(len(models)))
            ax.set_yticklabels([MODEL_LABELS[m] for m in models], fontsize=9)
            ax.set_title(title, fontsize=10, fontweight="bold")
            ax.set_xlabel(note, fontsize=8, color="#666")

            # value labels
            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_width() + max(vals) * 0.03,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va="center", ha="left", fontsize=8,
                )

            if "mctd" in col_m:
                ax.invert_xaxis()

        fig.tight_layout()
        savefig(fig, "fig1_dashboard.png")


# -----------------------------------------------------------------------------
# Figure 2 - Rank Reversal: why metric choice changes the winner
# -----------------------------------------------------------------------------
def fig_rank_reversal():
    df = pd.read_csv(os.path.join(RES_STRICT, "focus_overall_model_ranking_with_std.csv"))
    df = df.set_index("model_name")
    rank_data = {
        "F1":                     df["f1_rank"].to_dict(),
        "Dollar\nRecall":         df["dollar_recall_rank"].to_dict(),
        "MCTD\n(low=best)":       df["mctd_rank"].to_dict(),
        "Alert Cost\nEfficiency": df["ace_rank"].to_dict(),
    }
    metrics_order = ["F1", "Dollar\nRecall", "MCTD\n(low=best)", "Alert Cost\nEfficiency"]

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(10, 5.5))
        ax.set_title(
            "Rank Reversal: Model Ranking Depends on the Chosen Metric",
            fontsize=13, fontweight="bold"
        )

        x = np.arange(len(metrics_order))
        for model in MODEL_ORDER:
            ranks = [rank_data[m][model] for m in metrics_order]
            ax.plot(x, ranks, marker="o", linewidth=2.5, markersize=9,
                    color=PALETTE[model], label=model, zorder=3)
            # annotation at the last metric
            ax.annotate(
                model if model != "LSTM_AE" else "LSTM-AE",
                xy=(x[-1], ranks[-1]),
                xytext=(x[-1] + 0.08, ranks[-1]),
                fontsize=9, color=PALETTE[model], va="center",
            )

        ax.set_xticks(x)
        ax.set_xticklabels(metrics_order, fontsize=10)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.set_yticklabels(["#1 (best)", "#2", "#3", "#4", "#5 (worst)"], fontsize=9)
        ax.set_ylabel("Rank", fontsize=10)
        ax.set_xlim(-0.3, len(metrics_order) - 0.3)
        ax.set_ylim(5.4, 0.5)
        ax.axvline(0.5, color="#ccc", lw=1, ls="--")
        ax.axvline(2.5, color="#ccc", lw=1, ls="--")

        ax.text(0.02, 0.95,
            "Key insight: F1 alone would select Prophet.\n"
            "But EWMA looks best on Alert Cost Efficiency - yet misses\n"
            "most dollar impact. FinOps needs multiple metrics.",
            transform=ax.transAxes, fontsize=8.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f0f4ff", ec="#aac", alpha=0.9),
        )

        fig.tight_layout()
        savefig(fig, "fig2_rank_reversal.png")


# -----------------------------------------------------------------------------
# Figure 3 - Dollar Recall vs F1 Scatter (core argument)
# -----------------------------------------------------------------------------
def fig_f1_vs_dollar_recall():
    df = pd.read_csv(os.path.join(RES_STRICT, "focus_overall_model_ranking_with_std.csv"))

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 6))
        ax.set_title(
            "F1 Score vs Cost-Weighted Recall\n"
            "Standard classification != operational cost value",
            fontsize=12, fontweight="bold",
        )

        for _, row in df.iterrows():
            m = row["model_name"]
            ax.scatter(
                row["f1_mean"], row["cost_weighted_recall_mean"],
                color=PALETTE[m], s=180, zorder=4, edgecolors="white", linewidths=1.5,
            )
            ax.errorbar(
                row["f1_mean"], row["cost_weighted_recall_mean"],
                xerr=row["f1_std"], yerr=row["cost_weighted_recall_std"],
                fmt="none", color=PALETTE[m], alpha=0.45, capsize=4,
            )
            offset = (-0.018, 0.015) if m == "SeasonalNaiveMAD" else (0.012, 0.008)
            ax.annotate(
                MODEL_LABELS[m].replace("\n", " "),
                xy=(row["f1_mean"], row["cost_weighted_recall_mean"]),
                xytext=(row["f1_mean"] + offset[0], row["cost_weighted_recall_mean"] + offset[1]),
                fontsize=9, color=PALETTE[m], fontweight="bold",
            )

        # ideal zone
        ax.axhspan(0.85, 1.0, color="#e8f5e9", alpha=0.6, zorder=0)
        ax.text(0.55, 0.96, "High dollar recall zone", fontsize=8, color="#388e3c")

        ax.set_xlabel("F1 Score (higher is better)", fontsize=10)
        ax.set_ylabel("Cost-Weighted Recall (higher is better)", fontsize=10)
        ax.set_xlim(0.08, 0.70)
        ax.set_ylim(0.38, 1.02)

        ax.text(0.03, 0.03,
            "EWMA/SeasonalNaiveMAD: low F1 AND low dollar recall\n"
            "LSTM-AE: lower F1 than Isolation Forest but higher dollar recall\n"
            "-> F1 rank and dollar-recall rank diverge",
            transform=ax.transAxes, fontsize=8, va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff8e1", ec="#f0c040", alpha=0.9),
        )

        fig.tight_layout()
        savefig(fig, "fig3_f1_vs_dollar_recall.png")


# -----------------------------------------------------------------------------
# Figure 4 - Anomaly-Type Heatmap (dollar recall)
# -----------------------------------------------------------------------------
def fig_type_heatmap():
    df = pd.read_csv(os.path.join(RES_STRICT, "focus_anomaly_type_results.csv"))
    pivot = df.pivot(index="model_name", columns="anomaly_type", values="dollar_recall")
    pivot = pivot.loc[MODEL_ORDER, ["spike", "contextual", "gradual"]]

    col_labels = {"spike": "Spike", "contextual": "Contextual", "gradual": "Gradual"}

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_title(
            "Dollar Recall by Model x Anomaly Type\n"
            "Gradual anomalies stress simple residual and seasonal baselines",
            fontsize=12, fontweight="bold",
        )

        im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
        plt.colorbar(im, ax=ax, label="Dollar Recall (0 -> 1)")

        ax.set_xticks(range(3))
        ax.set_xticklabels([col_labels[c] for c in pivot.columns], fontsize=11)
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels([MODEL_LABELS[m].replace("\n", " ") for m in MODEL_ORDER], fontsize=10)

        for r in range(len(MODEL_ORDER)):
            for c in range(3):
                val = pivot.values[r, c]
                color = "white" if val < 0.45 or val > 0.75 else "black"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center",
                        fontsize=11, color=color, fontweight="bold")

        ax.set_xlabel("Anomaly Type", fontsize=10)
        ax.set_ylabel("Model", fontsize=10)

        fig.tight_layout()
        savefig(fig, "fig4_type_heatmap.png")


# -----------------------------------------------------------------------------
# Figure 5 - MCTD by Anomaly Type (detection delay cost)
# -----------------------------------------------------------------------------
def fig_mctd_by_type():
    df = pd.read_csv(os.path.join(RES_STRICT, "focus_anomaly_type_results.csv"))

    types = ["spike", "contextual", "gradual"]
    type_labels = {"spike": "Spike", "contextual": "Contextual", "gradual": "Gradual"}

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(13, 5), sharey=False)
        fig.suptitle(
            "Mean Cost-to-Detect (MCTD) by Anomaly Type\n"
            "USD accrued before the first alert - lower is better",
            fontsize=12, fontweight="bold",
        )

        for ax, atype in zip(axes, types):
            sub = df[df["anomaly_type"] == atype].set_index("model_name")
            vals   = [sub.loc[m, "mean_mctd"] for m in MODEL_ORDER]
            colors = [PALETTE[m] for m in MODEL_ORDER]

            bars = ax.bar(range(len(MODEL_ORDER)), vals, color=colors,
                          alpha=0.88, edgecolor="white", width=0.65)
            ax.set_title(type_labels[atype], fontsize=11, fontweight="bold")
            ax.set_xticks(range(len(MODEL_ORDER)))
            ax.set_xticklabels(
                [MODEL_LABELS[m].replace("\n", " ") for m in MODEL_ORDER],
                fontsize=8, rotation=20, ha="right",
            )
            ax.set_ylabel("MCTD (USD)" if atype == "spike" else "")
            ax.yaxis.set_major_locator(MaxNLocator(5))

            for bar, val in zip(bars, vals):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    f"${val:.0f}", ha="center", fontsize=8,
                )

        fig.tight_layout()
        savefig(fig, "fig5_mctd_by_type.png")


# -----------------------------------------------------------------------------
# Figure 6 - Intensity Level: Dollar Recall Progression
# -----------------------------------------------------------------------------
def fig_intensity_progression():
    df = pd.read_csv(os.path.join(RES_STRICT, "focus_anomaly_intensity_results.csv"))
    # aggregate across anomaly types per model x intensity
    grp = df.groupby(["model_name", "intensity_level"])["dollar_recall"].mean().reset_index()
    intensity_order = ["low", "mid", "high"]
    intensity_labels = {
        "low": "Low\ninjected intensity",
        "mid": "Mid\ninjected intensity",
        "high": "High\ninjected intensity",
    }

    with plt.rc_context(PLT_STYLE):
        fig, ax = plt.subplots(figsize=(9, 5.5))
        ax.set_title(
            "Dollar Recall by Anomaly Intensity Level\n"
            "All models struggle at low intensity - the real operational risk",
            fontsize=12, fontweight="bold",
        )

        x = np.arange(len(intensity_order))
        width = 0.15
        offsets = np.linspace(-(len(MODEL_ORDER)-1)/2, (len(MODEL_ORDER)-1)/2, len(MODEL_ORDER)) * width

        for i, model in enumerate(MODEL_ORDER):
            sub = grp[grp["model_name"] == model].set_index("intensity_level")
            vals = [sub.loc[lvl, "dollar_recall"] if lvl in sub.index else 0
                    for lvl in intensity_order]
            bars = ax.bar(x + offsets[i], vals, width=width * 0.9,
                          color=PALETTE[model], alpha=0.88, label=model, edgecolor="white")

        ax.set_xticks(x)
        ax.set_xticklabels([intensity_labels[l] for l in intensity_order], fontsize=10)
        ax.set_ylabel("Dollar Recall (averaged over anomaly types)", fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.axhline(0.8, color="#888", ls="--", lw=1, label="0.80 threshold")

        legend_handles = [
            mpatches.Patch(color=PALETTE[m], label=MODEL_LABELS[m].replace("\n", " "))
            for m in MODEL_ORDER
        ]
        legend_handles.append(
            plt.Line2D([0], [0], color="#888", ls="--", lw=1, label="0.80 threshold")
        )
        ax.legend(handles=legend_handles, loc="upper left", fontsize=8.5,
                  framealpha=0.9, ncol=2)

        ax.text(0.98, 0.03,
            "Low-intensity anomalies are hardest to detect\n"
            "but accumulate cost silently over time.",
            transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="#fff3e0", ec="#ffb74d", alpha=0.9),
        )

        fig.tight_layout()
        savefig(fig, "fig6_intensity_progression.png")


# -----------------------------------------------------------------------------
# Figure 7 - Research Framework Overview (conceptual)
# -----------------------------------------------------------------------------
def fig_framework():
    with plt.rc_context({**PLT_STYLE, "axes.grid": False}):
        fig, ax = plt.subplots(figsize=(13, 6.5))
        ax.set_xlim(0, 13)
        ax.set_ylim(0, 6.2)
        ax.axis("off")
        ax.set_title(
            "Research Framework: FOCUS-Calibrated FinOps Anomaly Benchmark",
            fontsize=13, fontweight="bold", pad=12,
        )

        # pad is kept small so the drawn box stays close to its (x, y, w, h)
        # footprint; layout below already leaves explicit vertical gaps.
        PAD = 0.04

        def box(ax, x, y, w, h, label, sub="", color="#2E9B5E", fs=10):
            rect = mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle=f"round,pad={PAD}",
                facecolor=color + "22", edgecolor=color, linewidth=2,
            )
            ax.add_patch(rect)
            ax.text(x + w/2, y + h/2 + (0.22 if sub else 0),
                    label, ha="center", va="center",
                    fontsize=fs, fontweight="bold", color=color)
            if sub:
                ax.text(x + w/2, y + h/2 - 0.34, sub,
                        ha="center", va="center", fontsize=7.5, color="#444")

        def arrow(ax, x1, y1, x2, y2, label="", lw=1.8):
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#555", lw=lw))
            if label:
                mx, my = (x1+x2)/2, (y1+y2)/2
                ax.text(mx, my+0.20, label, ha="center", fontsize=7.5,
                        color="#555")

        # -- Row 1: data pipeline (y 4.20 - 5.70) ---------------------------
        box(ax, 0.30, 4.20, 3.30, 1.50,
            "FOCUS Sample Data",
            "5.49M rows\n(AWS / Microsoft / Oracle)\n32 billing days",
            color="#1B6CA8", fs=9)
        box(ax, 4.40, 4.35, 3.00, 1.20,
            "Calibration Stats",
            "base_level / growth\nnoise_pct / weekly_factor",
            color="#E07B39", fs=9)
        box(ax, 8.20, 4.20, 4.50, 1.50,
            "FOCUS-Calibrated Benchmark",
            "730-day labeled series (4 services)\n"
            "controlled anomaly injection\n"
            "spike / contextual / gradual x low/mid/high",
            color="#2E9B5E", fs=9)

        # -- Row 2: evaluation, wide bar spanning all detectors (y 2.25-3.25)
        box(ax, 0.30, 2.25, 12.40, 1.00,
            "FinOps-Aware Evaluation",
            "F1   /   Dollar Recall   /   MCTD   /   Alert Cost Efficiency",
            color="#9B59B6", fs=10)

        # -- Row 3: five detectors (y 0.45 - 1.40) --------------------------
        models = [
            ("Prophet",          "Prophet"),
            ("IsolationForest",  "Isolation Forest"),
            ("LSTM_AE",          "LSTM-AE"),
            ("SeasonalNaiveMAD", "Seasonal NaiveMAD"),
            ("EWMA",             "EWMA"),
        ]
        slot_w = 13.0 / 5
        box_w = 2.30
        centers = []
        for i, (m, lbl) in enumerate(models):
            cx = slot_w * (i + 0.5)
            centers.append(cx)
            box(ax, cx - box_w/2, 0.45, box_w, 0.95, lbl,
                color=PALETTE[m], fs=8.5)

        # -- arrows ----------------------------------------------------------
        # data pipeline (horizontal)
        arrow(ax, 3.65, 4.95, 4.35, 4.95, "extract\nstatistics")
        arrow(ax, 7.45, 4.95, 8.15, 4.95, "calibrate\nbaseline")
        # benchmark -> evaluation (vertical, lands on the wide bar)
        arrow(ax, 10.45, 4.15, 10.45, 3.32, "Year-1 fit\nYear-2 eval")
        # each detector -> evaluation (gap is now 0.85 units, fully clear)
        for cx in centers:
            arrow(ax, cx, 1.45, cx, 2.20, lw=1.3)

        # footnote
        ax.text(0.30, 0.06,
            "Raw FOCUS -> sanity check only (no ground-truth labels).  "
            "Quantitative metrics computed on the FOCUS-calibrated synthetic benchmark.",
            fontsize=8, color="#666", style="italic")

        fig.tight_layout()
        savefig(fig, "fig7_framework.png")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Generating presentation figures ...")
    fig_dashboard()
    fig_rank_reversal()
    fig_f1_vs_dollar_recall()
    fig_type_heatmap()
    fig_mctd_by_type()
    fig_intensity_progression()
    fig_framework()
    print(f"\nDone. All figures saved to:\n  {os.path.abspath(OUT_DIR)}")
