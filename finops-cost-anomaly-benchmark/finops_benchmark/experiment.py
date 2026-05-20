"""Experiment orchestration and table builders."""

import numpy as np
import pandas as pd

try:
    from .config import (
        BUDGETS,
        EVENT_KEEP_COLS,
        METRIC_COLS,
        N_DAYS,
        START_DATE,
        YEAR2_START,
    )
    from .data import build_dataset, build_focus_calibrated_dataset
    from .evaluation import run_evaluation
    from .models import run_all_models
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        BUDGETS,
        EVENT_KEEP_COLS,
        METRIC_COLS,
        N_DAYS,
        START_DATE,
        YEAR2_START,
    )
    from data import build_dataset, build_focus_calibrated_dataset
    from evaluation import run_evaluation
    from models import run_all_models

def run_one_seed(seed, year2_start=YEAR2_START, budgets=BUDGETS, verbose=False):
    """
    한 seed에 대해 데이터 생성 -> 모델 score -> 다중 budget 평가까지 수행한다.

    Parameters
    ----------
    seed         : 실험 시드 (build_dataset, run_all_models, threshold 모두에 동일하게 사용)
    year2_start  : Year 2 시작 day index
    budgets      : [(budget_float, percentile_float), ...]
    verbose      : 진행 상황 출력 여부

    Returns
    -------
    metrics_df : 한 seed에서 모든 (budget x model) 조합의 model_metrics 행
                 추가 컬럼: seed, budget, threshold_quantile
    events_df  : 한 seed에서 모든 (budget x model x event) 조합의 event_results 행
                 추가 컬럼: seed, budget
    """
    if verbose:
        print(f"[seed={seed}] generating data and scoring all models ...")

    # 1) 데이터 생성 (Part 1 함수 재사용)
    df_s, events_table_s = build_dataset(seed=seed,
                                         n_days=N_DAYS,
                                         year2_start=year2_start,
                                         start_date=START_DATE)

    # 2) 4개 모델 score 일괄 계산 (Part 2 함수 재사용)
    scores_s, _aux = run_all_models(df_s, events_table_s,
                                    seed=seed, year2_start=year2_start,
                                    verbose=False)

    # 3) budget별 threshold 적용 + 평가 (Part 3 함수 재사용)
    metric_parts = []
    event_parts = []
    for budget, q in budgets:
        _pred_s, met_s, ev_s = run_evaluation(
            df_s, events_table_s, scores_s,
            year2_start=year2_start, percentile=q,
        )

        # 메타 정보 부착 (long-format 식별 키)
        met_s = met_s.copy()
        met_s.insert(0, "seed", seed)
        met_s.insert(1, "year1_fpr_target", budget)
        met_s.insert(2, "threshold_quantile", q)

        ev_s = ev_s.copy()
        ev_s.insert(0, "seed", seed)
        ev_s.insert(1, "year1_fpr_target", budget)

        metric_parts.append(met_s)
        event_parts.append(ev_s)

    metrics_df = pd.concat(metric_parts, axis=0, ignore_index=True)
    events_df = pd.concat(event_parts, axis=0, ignore_index=True)
    return metrics_df, events_df

def build_summary_metrics(all_model_metrics_df, metric_cols=METRIC_COLS):
    """
    seed 차원을 평균/표준편차로 축약하여 (budget, model_name) 단위 요약표를 만든다.

    반환 컬럼
        budget, model_name,
        <metric>_mean, <metric>_std (각 metric_cols 항목)
    """
    grp = (all_model_metrics_df
           .groupby(["year1_fpr_target", "model_name"])[metric_cols]
           .agg(["mean", "std"]))

    # MultiIndex 컬럼을 평탄화: ('f1','mean') -> 'f1_mean'
    grp.columns = [f"{m}_{stat}" for m, stat in grp.columns]
    grp = grp.reset_index().sort_values(["year1_fpr_target", "model_name"])
    return grp

def build_rank_table(all_model_metrics_df, year1_fpr_target=0.01):
    """
    지정 year1_fpr_target에서 모델별 F1, cost_weighted_recall, alert_cost_efficiency
    평균값과 그 순위를 한 표에 정리한다 (낮은 rank = 더 좋음).
    """
    sub = (all_model_metrics_df[all_model_metrics_df["year1_fpr_target"] == year1_fpr_target]
           .groupby("model_name")[["f1",
                                   "cost_weighted_recall",
                                   "alert_cost_efficiency",
                                   "mean_mctd"]]
           .mean()
           .round(4))

    sub["f1_rank"] = sub["f1"].rank(ascending=False, method="min").astype(int)
    sub["dollar_recall_rank"] = (
        sub["cost_weighted_recall"].rank(ascending=False, method="min").astype(int)
    )
    sub["ace_rank"] = (
        sub["alert_cost_efficiency"].rank(ascending=False, method="min").astype(int)
    )
    # MCTD는 낮을수록 좋음
    sub["mctd_rank"] = sub["mean_mctd"].rank(ascending=True, method="min").astype(int)

    return sub.reset_index().sort_values("f1_rank")

def _format_mean_std(mean, std, kind="ratio"):
    """
    metric 한 쌍을 'mean ± std' 문자열로 포맷한다.

    Parameters
    ----------
    mean : float
    std  : float (NaN이면 0으로 표시)
    kind : "ratio" | "money"
        ratio  : 0~1 또는 작은 실수 (소수점 3자리)
        money  : 비용 단위 (천 단위 콤마, 소수점 1자리)
    """
    if std is None or (isinstance(std, float) and np.isnan(std)):
        std = 0.0
    if kind == "money":
        return f"{mean:,.1f} \u00b1 {std:,.1f}"
    return f"{mean:.3f} \u00b1 {std:.3f}"

def build_core_results_table(summary_df, year1_fpr_target=0.01):
    """
    year1_fpr_target=1% 기준 paper-ready 핵심 결과표를 만든다.

    포함 metric
        f1, cost_weighted_recall, alert_cost_efficiency,
        mean_mctd, false_alarm_rate

    행 순서는 f1 mean 내림차순.
    """
    sub = summary_df[summary_df["year1_fpr_target"] == year1_fpr_target].copy()

    rows = []
    # f1 내림차순으로 순회
    sub_sorted = sub.sort_values("f1_mean", ascending=False)
    for _, r in sub_sorted.iterrows():
        rows.append({
            "model_name": r["model_name"],
            "f1": _format_mean_std(r["f1_mean"], r["f1_std"], "ratio"),
            "cost_weighted_recall": _format_mean_std(
                r["cost_weighted_recall_mean"],
                r["cost_weighted_recall_std"], "ratio"),
            "alert_cost_efficiency": _format_mean_std(
                r["alert_cost_efficiency_mean"],
                r["alert_cost_efficiency_std"], "money"),
            "mean_mctd": _format_mean_std(
                r["mean_mctd_mean"],
                r["mean_mctd_std"], "money"),
            "false_alarm_rate": _format_mean_std(
                r["false_alarm_rate_mean"],
                r["false_alarm_rate_std"], "ratio"),
        })

    return pd.DataFrame(rows)

def build_rank_comparison(all_metrics_df, year1_fpr_target=0.01):
    """
    각 모델의 metric 평균값과 그 순위를 한 표에 정리한다.

    rank 규칙
        - f1, cost_weighted_recall, alert_cost_efficiency : 큰 값이 1위
        - mean_mctd                                       : 작은 값이 1위
        - 동률 처리 method="min"

    Returns
    -------
    DataFrame
        model_name, f1, f1_rank,
        cost_weighted_recall, dollar_recall_rank,
        alert_cost_efficiency, ace_rank,
        mean_mctd, mctd_rank
    """
    sub = (all_metrics_df[all_metrics_df["year1_fpr_target"] == year1_fpr_target]
           .groupby("model_name")[["f1",
                                   "cost_weighted_recall",
                                   "alert_cost_efficiency",
                                   "mean_mctd"]]
           .mean())

    sub["f1_rank"] = sub["f1"].rank(ascending=False, method="min").astype(int)
    sub["dollar_recall_rank"] = (
        sub["cost_weighted_recall"]
        .rank(ascending=False, method="min").astype(int)
    )
    sub["ace_rank"] = (
        sub["alert_cost_efficiency"]
        .rank(ascending=False, method="min").astype(int)
    )
    sub["mctd_rank"] = (
        sub["mean_mctd"].rank(ascending=True, method="min").astype(int)
    )

    out = sub.reset_index()
    out = out[[
        "model_name",
        "f1", "f1_rank",
        "cost_weighted_recall", "dollar_recall_rank",
        "alert_cost_efficiency", "ace_rank",
        "mean_mctd", "mctd_rank",
    ]].sort_values("f1_rank").reset_index(drop=True)

    # 수치 컬럼은 보기 좋게 반올림
    out["f1"] = out["f1"].round(4)
    out["cost_weighted_recall"] = out["cost_weighted_recall"].round(4)
    out["alert_cost_efficiency"] = out["alert_cost_efficiency"].round(2)
    out["mean_mctd"] = out["mean_mctd"].round(2)
    return out


def run_one_seed_focus(seed, stats, year2_start=YEAR2_START, budgets=BUDGETS, verbose=False):
    """Like ``run_one_seed`` but uses a FOCUS-calibrated dataset.

    Parameters
    ----------
    seed : int
    stats : dict  – output of ``focus_calibration.fit_series_statistics``
    year2_start, budgets, verbose : same as ``run_one_seed``
    """
    if verbose:
        print(f"[seed={seed}] generating FOCUS-calibrated data and scoring all models ...")

    df_s, events_table_s = build_focus_calibrated_dataset(
        stats=stats, seed=seed, n_days=N_DAYS,
        year2_start=year2_start, start_date=START_DATE,
    )
    scores_s, _aux = run_all_models(
        df_s, events_table_s, seed=seed, year2_start=year2_start, verbose=False,
    )

    metric_parts = []
    event_parts = []
    for budget, q in budgets:
        _pred_s, met_s, ev_s = run_evaluation(
            df_s, events_table_s, scores_s, year2_start=year2_start, percentile=q,
        )
        met_s = met_s.copy()
        met_s.insert(0, "seed", seed)
        met_s.insert(1, "year1_fpr_target", budget)
        met_s.insert(2, "threshold_quantile", q)
        ev_s = ev_s.copy()
        ev_s.insert(0, "seed", seed)
        ev_s.insert(1, "year1_fpr_target", budget)
        metric_parts.append(met_s)
        event_parts.append(ev_s)

    return (
        pd.concat(metric_parts, axis=0, ignore_index=True),
        pd.concat(event_parts, axis=0, ignore_index=True),
    )


def run_multi_seed_focus(seeds, stats, budgets=BUDGETS, year2_start=YEAR2_START):
    """Run ``run_one_seed_focus`` over multiple seeds and concatenate results."""
    all_metrics_parts = []
    all_events_parts = []
    for seed in seeds:
        metrics_s, events_s = run_one_seed_focus(
            seed=seed, stats=stats, budgets=budgets, year2_start=year2_start,
        )
        all_metrics_parts.append(metrics_s)
        all_events_parts.append(events_s)
    all_model_metrics_df = pd.concat(all_metrics_parts, axis=0, ignore_index=True)
    all_event_results_df = pd.concat(all_events_parts, axis=0, ignore_index=True)[EVENT_KEEP_COLS]
    return all_model_metrics_df, all_event_results_df


def run_multi_seed(seeds, budgets=BUDGETS, year2_start=YEAR2_START):
    all_metrics_parts = []
    all_events_parts = []
    for seed in seeds:
        metrics_s, events_s = run_one_seed(
            seed=seed,
            year2_start=year2_start,
            budgets=budgets,
            verbose=False,
        )
        all_metrics_parts.append(metrics_s)
        all_events_parts.append(events_s)

    all_model_metrics_df = pd.concat(all_metrics_parts, axis=0, ignore_index=True)
    all_event_results_df_full = pd.concat(all_events_parts, axis=0, ignore_index=True)
    all_event_results_df = all_event_results_df_full[EVENT_KEEP_COLS].copy()
    return all_model_metrics_df, all_event_results_df
