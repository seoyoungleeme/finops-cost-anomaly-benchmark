"""Plotting helpers for exploratory and paper-ready figures."""

import matplotlib.pyplot as plt

from .config import BUDGETS, PAPER_COLOR_MAP, PAPER_DPI, SEEDS, YEAR2_START

def plot_data_overview(df, events_df, year2_start=YEAR2_START, save_path=None):
    """
    전체 시계열과 excess_cost를 한 figure에 시각화한다.
    이상 점은 anomaly_type별 색상으로 구분한다.
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    ax0 = axes[0]
    ax0.plot(df["day"], df["y_expected_baseline"],
             color="black", lw=0.8, ls="--", alpha=0.55,
             label="y_expected_baseline")
    ax0.plot(df["day"], df["y"],
             color="steelblue", lw=0.9, label="y_actual")
    ax0.axvline(year2_start, color="gray", ls=":", label="Year 2 start")

    color_map = {"spike": "red", "contextual": "orange", "gradual": "purple"}
    for atype, c in color_map.items():
        m = (df["anomaly_type"] == atype)
        if m.any():
            ax0.scatter(df.loc[m, "day"], df.loc[m, "y"],
                        s=18, color=c, label=atype,
                        zorder=5, edgecolors="none")

    ax0.set_title("Synthetic cloud cost series with injected anomalies")
    ax0.set_ylabel("Cost")
    ax0.legend(loc="upper left", fontsize=8, ncol=3)
    ax0.grid(alpha=0.3)

    ax1 = axes[1]
    ax1.plot(df["day"], df["excess_cost"],
             color="gray", lw=0.5, alpha=0.7, label="excess_cost")
    ax1.fill_between(df["day"], 0, df["excess_cost"],
                     where=df["is_anomaly"].values,
                     color="red", alpha=0.35,
                     label="anomaly excess")
    ax1.axhline(0, color="black", lw=0.4)
    ax1.axvline(year2_start, color="gray", ls=":")
    ax1.set_xlabel("Day index")
    ax1.set_ylabel("Excess cost")
    ax1.set_title("Excess cost = y_actual - y_expected_baseline")
    ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig

def plot_anomaly_examples(df, events_df, save_path=None, window=10):
    """
    각 anomaly_type별로 한 개 예시 이벤트를 골라 zoom-in 시각화한다.
    이벤트 앞뒤 window 일을 함께 표시.
    """
    types = ["spike", "contextual", "gradual"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    for ax, atype in zip(axes, types):
        sub = events_df[events_df["anomaly_type"] == atype]
        if len(sub) == 0:
            ax.set_title(f"{atype} (no event)")
            continue

        cand = sub[sub["intensity_level"] == "mid"]
        ev = (cand.iloc[0] if len(cand) else sub.iloc[0])

        s = max(0, int(ev["start_day"]) - window)
        e = min(len(df), int(ev["end_day"]) + window + 1)
        seg = df.iloc[s:e]

        ax.plot(seg["day"], seg["y_expected_baseline"],
                color="black", ls="--", lw=0.9, label="expected")
        ax.plot(seg["day"], seg["y"],
                color="steelblue", lw=1.0, label="actual")

        m = seg["is_anomaly"]
        ax.scatter(seg.loc[m, "day"], seg.loc[m, "y"],
                   s=22, color="red", zorder=5, label="anomaly")

        ax.set_title(f"{atype} ({ev['intensity_level']}) "
                     f"event_id={int(ev['event_id'])}")
        ax.set_xlabel("Day")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Cost")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig

def plot_score_overview(df, scores_long, year2_start=YEAR2_START,
                        save_path=None):
    """
    모델별 score 시계열을 4분할 패널로 시각화.
    이상 구간을 회색 음영, 이상 점을 빨간 점으로 표시.
    """
    models = list(scores_long["model_name"].unique())
    fig, axes = plt.subplots(len(models), 1, figsize=(14, 2.6 * len(models)),
                             sharex=True)
    if len(models) == 1:
        axes = [axes]

    anomaly_days = df.loc[df["is_anomaly"], "day"].values

    for ax, name in zip(axes, models):
        sub = scores_long[scores_long["model_name"] == name]
        ax.plot(sub["day"], sub["score"], color="black", lw=0.7)
        for d in anomaly_days:
            ax.axvline(d, color="red", lw=0.4, alpha=0.25)
        ax.axvline(year2_start, color="gray", ls=":")
        ax.set_title(f"{name} score")
        ax.set_ylabel("score")
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("Day index")
    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig

def _bar_with_err(ax, names, means, stds, ylabel, title):
    """
    공통 막대 그래프 헬퍼: yerr 포함, mean 정렬은 호출자 책임.
    """
    ax.bar(names, means, yerr=stds, capsize=4,
           color="steelblue", edgecolor="black", alpha=0.85)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    for tick in ax.get_xticklabels():
        tick.set_rotation(15)

def plot_budget1_bar(summary_metrics_df, metric, save_path,
                     budget=0.01, ascending_better=False):
    """
    한 budget에서 모델별 metric 평균을 막대로, 표준편차를 errorbar로 그린다.

    Parameters
    ----------
    metric            : 'f1', 'cost_weighted_recall', 'alert_cost_efficiency' 등
    ascending_better  : True면 작을수록 좋은 metric (정렬 방향만 다름)
    """
    sub = summary_metrics_df[summary_metrics_df["budget"] == budget].copy()
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    sub = sub.sort_values(mean_col, ascending=ascending_better)

    fig, ax = plt.subplots(figsize=(7, 4))
    _bar_with_err(
        ax,
        sub["model_name"].values,
        sub[mean_col].values,
        sub[std_col].values,
        ylabel=metric,
        title=f"{metric} at budget={budget*100:.1f}% (mean ± std over {len(SEEDS)} seeds)",
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig

def plot_metric_by_budget(summary_metrics_df, metric, save_path):
    """
    x축 budget, y축 metric mean, 모델별 라인. errorbar = std.
    """
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    for name, sub in summary_metrics_df.groupby("model_name"):
        sub = sub.sort_values("budget")
        ax.errorbar(sub["budget"].values * 100,
                    sub[mean_col].values,
                    yerr=sub[std_col].values,
                    marker="o", capsize=3, lw=1.4, label=name)

    ax.set_xlabel("Budget (% of days allowed to alert)")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} across budgets (mean ± std)")
    ax.set_xticks([b * 100 for b, _ in BUDGETS])
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig

def plot_rank_comparison(rank_table, save_path):
    """
    F1 rank와 alert_cost_efficiency rank를 좌우로 배치하고 모델별로 선으로 잇는다.
    rank가 1=best (위쪽), 4=worst (아래쪽)이 되도록 y축을 반전한다.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    n = len(rank_table)

    for _, row in rank_table.iterrows():
        ax.plot([0, 1], [row["f1_rank"], row["ace_rank"]],
                marker="o", lw=1.5, label=row["model_name"])

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["F1 rank", "Alert Cost Efficiency rank"])
    ax.set_yticks(range(1, n + 1))
    ax.set_ylim(n + 0.5, 0.5)         # 1이 위쪽
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("F1 rank vs Alert Cost Efficiency rank (budget=1%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig

def get_canonical_model_order(summary_df, budget=0.01):
    """
    F1 mean 내림차순으로 모델 이름 리스트를 반환한다.
    bar chart 3종은 모두 이 순서를 따른다.
    """
    sub = (summary_df[summary_df["budget"] == budget]
           .sort_values("f1_mean", ascending=False))
    return sub["model_name"].tolist()

def _adaptive_label(v):
    """
    bar chart 위에 표시할 값 레이블을 스케일에 맞게 포맷한다.
    """
    av = abs(v)
    if av < 10:
        return f"{v:.3f}"
    if av < 1000:
        return f"{v:.1f}"
    return f"{v:,.0f}"

def plot_paper_bar(summary_df, metric, save_path,
                   budget=0.01, ylabel=None, title=None,
                   model_order=None,
                   bar_color="#3a76b8"):
    """
    budget=1% 기준 paper-ready 막대 그래프.

    - mean을 막대 높이, std를 errorbar로 표시
    - 막대 위에 mean 값 텍스트 라벨 표시 (스케일에 따라 자동 포맷)
    - 모델 순서는 model_order로 고정 (None이면 입력 순서)

    Parameters
    ----------
    summary_df  : summary_metrics_df
    metric      : 'f1' | 'cost_weighted_recall' | 'alert_cost_efficiency' 등
    save_path   : png 저장 경로
    """
    sub = summary_df[summary_df["budget"] == budget].copy()
    if model_order is not None:
        sub = (sub.set_index("model_name")
                  .loc[model_order]
                  .reset_index())

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"
    means = sub[mean_col].values.astype(float)
    stds = sub[std_col].fillna(0.0).values.astype(float)

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    bars = ax.bar(sub["model_name"], means, yerr=stds, capsize=4,
                  color=bar_color, edgecolor="black", alpha=0.85,
                  error_kw={"elinewidth": 1.2, "ecolor": "black"})

    # 값 라벨 (errorbar 위쪽으로 살짝 띄움)
    pad = 0.02 * (max(means) if max(means) > 0 else 1.0)
    for bar, mu, sd in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sd + pad,
                _adaptive_label(mu),
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel(ylabel or metric)
    ax.set_title(title or
                 f"{metric} at budget={budget*100:.1f}% (mean \u00b1 std)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    plt.xticks(rotation=12)
    plt.tight_layout()
    fig.savefig(save_path, dpi=PAPER_DPI, bbox_inches="tight")
    return fig

def plot_paper_line(summary_df, metric, save_path,
                    ylabel=None, title=None,
                    lower_is_better=False):
    """
    budget축으로 metric 변화를 보는 paper-ready 선 그래프.

    - x : budget (% 단위)
    - y : metric mean, errorbar = std
    - 모델별 라인 (PAPER_COLOR_MAP 색상 고정)
    - lower_is_better=True면 y축 라벨에 명시
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # 모델 이름은 알파벳 정렬 (legend 가독성)
    for name in sorted(summary_df["model_name"].unique()):
        sub = (summary_df[summary_df["model_name"] == name]
               .sort_values("budget"))
        ax.errorbar(sub["budget"].values * 100.0,
                    sub[mean_col].values,
                    yerr=sub[std_col].fillna(0.0).values,
                    marker="o", capsize=3, lw=1.6,
                    label=name,
                    color=PAPER_COLOR_MAP.get(name, None))

    base_ylabel = ylabel or metric
    if lower_is_better:
        base_ylabel = f"{base_ylabel} (lower is better)"
    ax.set_xlabel("Budget (% of evaluation days allowed to alert)")
    ax.set_ylabel(base_ylabel)
    ax.set_title(title or f"{metric} across budgets (mean \u00b1 std)")

    # x ticks는 BUDGETS 정의 그대로
    ax.set_xticks([b * 100.0 for b, _ in BUDGETS])
    ax.grid(alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9, loc="best")
    plt.tight_layout()
    fig.savefig(save_path, dpi=PAPER_DPI, bbox_inches="tight")
    return fig


def configure_paper_matplotlib():
    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 110,
    })
