"""Thresholding and evaluation metrics."""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .config import YEAR2_START
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import YEAR2_START

def compute_thresholds(scores_long, year2_start=YEAR2_START, percentile=99.0):
    """
    각 모델별로 Year 1 (day < year2_start) score의 percentile을 threshold로 계산한다.

    Parameters
    ----------
    scores_long : DataFrame [model_name, date, day, score]
    year2_start : Year 2 시작 day index
    percentile  : 사용할 분위수 (0~100). Primary 정책은 99.

    Returns
    -------
    thresholds : dict {model_name: threshold(float)}
    """
    thresholds = {}
    for name, sub in scores_long.groupby("model_name"):
        # Year 1 구간 score만 사용, NaN은 분위수 계산에서 제외
        year1_scores = sub.loc[sub["day"] < year2_start, "score"].dropna().values
        if len(year1_scores) == 0:
            # 모든 score가 NaN이면 절대 alert가 안 발생하도록 +inf 설정
            thresholds[name] = float("inf")
        else:
            thresholds[name] = float(np.percentile(year1_scores, percentile))
    return thresholds

def build_predictions_df(df, scores_long, thresholds):
    """
    score / threshold / alert / label을 한 long-format으로 결합한다.

    NaN 처리
        - alert 비교 시 score는 NaN -> 0.0으로 채워 사용한다 (NaN 일은 알람 불가).
        - 단, score 컬럼 자체에는 NaN을 그대로 보존하여 추후 AUPRC 등에서 다시 처리한다.

    Parameters
    ----------
    df          : 표준 데이터 DataFrame (Part 1에서 생성)
    scores_long : long-format score DataFrame (Part 2에서 생성)
    thresholds  : dict {model_name: threshold}

    Returns
    -------
    predictions_df : DataFrame with columns
        [model_name, date, day, score, threshold, alert,
         is_anomaly, anomaly_type, intensity_level, event_id, excess_cost]
    """
    label_cols = ["day", "is_anomaly", "anomaly_type", "intensity_level",
                  "event_id", "excess_cost", "cost_impact"]
    labels = df[label_cols].copy()

    sl = scores_long.copy()
    sl["threshold"] = sl["model_name"].map(thresholds).astype(float)

    # NaN score는 0으로 가정하여 비교 (양의 threshold이므로 자동 탈락)
    score_for_cmp = sl["score"].fillna(0.0)
    sl["alert"] = (score_for_cmp > sl["threshold"]).astype(bool)

    pred = sl.merge(labels, on="day", how="left")

    # date 컬럼이 scores_long에 없을 가능성 대비
    if "date" not in pred.columns:
        pred = pred.merge(df[["day", "date"]], on="day", how="left")

    cols = ["model_name", "date", "day", "score", "threshold", "alert",
            "is_anomaly", "anomaly_type", "intensity_level", "event_id",
            "excess_cost", "cost_impact"]
    pred = pred[cols].sort_values(["model_name", "day"]).reset_index(drop=True)
    return pred

def compute_pointwise_metrics(pred_m, year2_start=YEAR2_START):
    """
    Year 2 구간에서 일 단위 (point-wise) 분류 metric을 계산한다.

    AUPRC는 raw score(NaN -> 0)에 대해 계산하여 ranking 품질을 반영한다.
    Precision/Recall/F1은 alert(boolean)에 대해 계산한다.

    Returns
    -------
    dict : precision, recall, f1, auprc
    """
    sub = pred_m[pred_m["day"] >= year2_start].copy()
    y_true  = sub["is_anomaly"].astype(int).values
    y_pred  = sub["alert"].astype(int).values
    y_score = sub["score"].fillna(0.0).values

    if y_pred.sum() == 0:
        precision = 0.0
    else:
        precision = float(precision_score(y_true, y_pred, zero_division=0))
    recall = float(recall_score(y_true, y_pred, zero_division=0))
    f1     = float(f1_score(y_true, y_pred, zero_division=0))

    # AUPRC는 양성이 하나도 없으면 정의 불가 -> 0 처리
    if y_true.sum() == 0:
        auprc = 0.0
    else:
        auprc = float(average_precision_score(y_true, y_score))

    return {"precision": precision, "recall": recall,
            "f1": f1, "auprc": auprc}

def compute_far(pred_m, year2_start=YEAR2_START):
    """
    False Alarm Rate
        = (Year 2 정상일 중 alert가 발생한 일수)
          / (Year 2 정상일 수)

    Returns
    -------
    float
    """
    sub = pred_m[pred_m["day"] >= year2_start]
    normal_mask = ~sub["is_anomaly"].astype(bool)
    n_normal = int(normal_mask.sum())
    if n_normal == 0:
        return 0.0
    n_false = int(sub.loc[normal_mask, "alert"].sum())
    return float(n_false) / float(n_normal)

def compute_event_results_one(pred_m, events_df, model_name):
    """
    한 모델에 대해 이벤트 단위 결과 DataFrame을 만든다.

    각 이벤트 [start_day, end_day]에 대해
        detected         : 기간 내 alert가 한 번이라도 있으면 True
        first_alert_day  : 기간 내 첫 alert day (없으면 NaN)
        detection_delay  : first_alert_day - start_day (미탐이면 NaN)
        mctd             : 탐지 직전까지의 excess_cost 누적합
                           - detected and first_alert_day == start_day -> 0
                           - detected and first_alert_day  > start_day -> sum [s, first_alert_day - 1]
                           - undetected                                -> sum [s, e]
        total_excess_cost: events_df의 값을 그대로 사용

    Parameters
    ----------
    pred_m     : 한 모델의 predictions slice (model_name 단일)
    events_df  : 이벤트 메타 DataFrame
    model_name : 결과 행에 부여할 모델 이름

    Returns
    -------
    DataFrame
    """
    # day 기반 fast lookup
    pdy = pred_m.sort_values("day").set_index("day")

    rows = []
    for _, ev in events_df.iterrows():
        s = int(ev["start_day"])
        e = int(ev["end_day"])
        seg = pdy.loc[s:e]                    # inclusive label slicing
        alerts = seg["alert"].values.astype(bool)

        if alerts.any():
            detected = True
            first_offset = int(np.argmax(alerts))
            first_alert_day = s + first_offset
            detection_delay = float(first_alert_day - s)
            if first_alert_day == s:
                mctd = 0.0
            else:
                mctd = float(pdy.loc[s:first_alert_day - 1, "cost_impact"].sum())
        else:
            detected = False
            first_alert_day = np.nan
            detection_delay = np.nan
            mctd = float(pdy.loc[s:e, "cost_impact"].sum())

        rows.append({
            "model_name": model_name,
            "event_id": int(ev["event_id"]),
            "anomaly_type": ev["anomaly_type"],
            "intensity_level": ev["intensity_level"],
            "start_day": s,
            "end_day": e,
            "detected": detected,
            "first_alert_day": first_alert_day,
            "detection_delay": detection_delay,
            "total_excess_cost": float(ev["total_excess_cost"]),
            "mctd": mctd,
        })

    return pd.DataFrame(rows)

def compute_cost_weighted(event_res_one, total_alerts_year2):
    """
    cost-weighted metric 계산.

    Parameters
    ----------
    event_res_one      : compute_event_results_one의 출력 (단일 모델)
    total_alerts_year2 : Year 2의 총 알람 수 (int)

    Returns
    -------
    dict : cost_weighted_recall, mean_mctd, alert_cost_efficiency
    """
    total_cost = float(event_res_one["total_excess_cost"].sum())
    detected_cost = float(
        event_res_one.loc[event_res_one["detected"], "total_excess_cost"].sum()
    )

    cost_recall = (detected_cost / total_cost) if total_cost > 0 else 0.0
    mean_mctd = (float(event_res_one["mctd"].mean())
                 if len(event_res_one) > 0 else 0.0)

    if total_alerts_year2 <= 0:
        ace = 0.0
    else:
        ace = detected_cost / float(total_alerts_year2)

    return {"cost_weighted_recall": cost_recall,
            "mean_mctd": mean_mctd,
            "alert_cost_efficiency": ace}

def _dollar_recall_group(g):
    tot = g["total_excess_cost"].sum()
    det = g.loc[g["detected"], "total_excess_cost"].sum()
    return det / tot if tot > 0 else 0.0

def evaluate_one_model(pred_m, events_df, model_name, year2_start=YEAR2_START):
    """
    한 모델에 대해 모든 metric을 계산하여
        (metrics_dict, event_results_df) 튜플을 반환한다.
    """
    # event-level
    ev_res = compute_event_results_one(pred_m, events_df, model_name)
    if len(ev_res) > 0:
        event_recall = float(ev_res["detected"].mean())
        # detection_delay는 미탐 = NaN이므로 mean(skipna)이 자연 평균
        if ev_res["detected"].any():
            mean_dd = float(ev_res["detection_delay"].mean())
        else:
            mean_dd = float("nan")
    else:
        event_recall = 0.0
        mean_dd = float("nan")

    # point-wise
    pw = compute_pointwise_metrics(pred_m, year2_start=year2_start)

    # FAR
    far = compute_far(pred_m, year2_start=year2_start)

    # Year 2 전체 alert 수
    total_alerts = int(
        pred_m.loc[pred_m["day"] >= year2_start, "alert"].sum()
    )

    # cost-weighted
    cw = compute_cost_weighted(ev_res, total_alerts)

    threshold = float(pred_m["threshold"].iloc[0])

    metrics = {
        "model_name": model_name,
        "threshold": threshold,
        "total_alerts": total_alerts,
        "false_alarm_rate": far,
        "precision": pw["precision"],
        "recall": pw["recall"],
        "f1": pw["f1"],
        "auprc": pw["auprc"],
        "event_recall": event_recall,
        "mean_detection_delay": mean_dd,
        "cost_weighted_recall": cw["cost_weighted_recall"],
        "mean_mctd": cw["mean_mctd"],
        "alert_cost_efficiency": cw["alert_cost_efficiency"],
    }
    return metrics, ev_res

def run_evaluation(df, events_df, scores_long,
                   year2_start=YEAR2_START, percentile=99.0):
    """
    전체 모델에 대해 threshold 적용 + 평가를 실행하고,
    Part 3 명세의 세 DataFrame을 반환한다.

    Returns
    -------
    predictions_df    : long-format day-level
    model_metrics_df  : 모델 단위 요약
    event_results_df  : 모델 x 이벤트 단위 결과
    """
    thresholds = compute_thresholds(scores_long, year2_start=year2_start,
                                    percentile=percentile)
    predictions = build_predictions_df(df, scores_long, thresholds)

    metric_rows = []
    ev_parts = []
    for name in predictions["model_name"].unique():
        pred_m = (predictions[predictions["model_name"] == name]
                  .sort_values("day")
                  .reset_index(drop=True))
        m, ev_res = evaluate_one_model(pred_m, events_df, name,
                                       year2_start=year2_start)
        metric_rows.append(m)
        ev_parts.append(ev_res)

    metrics_cols = [
        "model_name", "threshold", "total_alerts", "false_alarm_rate",
        "precision", "recall", "f1", "auprc",
        "event_recall", "mean_detection_delay",
        "cost_weighted_recall", "mean_mctd", "alert_cost_efficiency",
    ]
    model_metrics = pd.DataFrame(metric_rows)[metrics_cols]
    event_results = pd.concat(ev_parts, axis=0, ignore_index=True)

    return predictions, model_metrics, event_results
