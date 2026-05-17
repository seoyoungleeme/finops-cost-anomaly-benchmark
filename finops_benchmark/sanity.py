"""Sanity checks for benchmark outputs."""

import numpy as np
import pandas as pd

try:
    from .config import NOISE_PCT, YEAR2_START
    from .evaluation import compute_thresholds
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import NOISE_PCT, YEAR2_START
    from evaluation import compute_thresholds

def run_sanity_checks(df, events_df, scores_long, thresholds,
                      predictions_df, model_metrics_df,
                      all_model_metrics_df=None,
                      all_event_results_df=None,
                      year2_start=YEAR2_START,
                      noise_pct=NOISE_PCT):
    """
    Part 1~4 결과에 대한 검증을 수행하고 (요약표, 부속 상세 dict)를 반환한다.

    상태 코드
        PASS : 기대대로 통과
        WARN : 설계상 허용 가능한 편차이거나 회색 영역
        FAIL : 명백한 오류
        INFO : 단순 정보 (판정 없음)
    """
    rows = []
    detail_store = {}

    # ----------------------------------------------------------
    # Check 1. Year 1 has no anomalies
    # ----------------------------------------------------------
    n_y1 = int(df.loc[df["day"] < year2_start, "is_anomaly"].sum())
    rows.append({
        "id": 1,
        "check": "Year 1 has no anomalies",
        "status": "PASS" if n_y1 == 0 else "FAIL",
        "detail": f"Year 1 anomaly days = {n_y1} (expected 0)",
    })

    # ----------------------------------------------------------
    # Check 2. Year 2 has anomalies
    # ----------------------------------------------------------
    n_y2 = int(df.loc[df["day"] >= year2_start, "is_anomaly"].sum())
    rows.append({
        "id": 2,
        "check": "Year 2 has anomalies",
        "status": "PASS" if n_y2 > 0 else "FAIL",
        "detail": f"Year 2 anomaly days = {n_y2}",
    })

    # ----------------------------------------------------------
    # Check 3. Event intervals do not overlap
    #   start_day로 정렬한 뒤 인접 쌍의 (e_curr, s_next) 비교
    # ----------------------------------------------------------
    ev_sorted = events_df.sort_values("start_day").reset_index(drop=True)
    overlap_pairs = []
    for i in range(len(ev_sorted) - 1):
        e_curr = int(ev_sorted.loc[i, "end_day"])
        s_next = int(ev_sorted.loc[i + 1, "start_day"])
        if s_next <= e_curr:
            overlap_pairs.append((int(ev_sorted.loc[i, "event_id"]),
                                  int(ev_sorted.loc[i + 1, "event_id"])))
    rows.append({
        "id": 3,
        "check": "Event intervals do not overlap",
        "status": "PASS" if not overlap_pairs else "FAIL",
        "detail": (f"overlapping pairs = {overlap_pairs[:3]}"
                   f"{' (showing first 3)' if len(overlap_pairs) > 3 else ''}"),
    })

    # ---------------------------------------------------------
    # Check 4. cost_impact >= 0 in anomaly days
    #   ground-truth 주입 비용은 항상 >= 0이어야 한다.
    #   (이전 버전은 noise 포함 residual을 검증했으므로 1건 음수 발생 가능했음)
    # ---------------------------------------------------------
    anom_mask = df["is_anomaly"].astype(bool)
    n_anom = int(anom_mask.sum())
    if n_anom > 0:
        ci_anom = df.loc[anom_mask, "cost_impact"].values
        n_nonneg = int((ci_anom >= 0).sum())
        n_neg = int((ci_anom < 0).sum())
        n_pos = int((ci_anom > 0).sum())
        status = "PASS" if n_neg == 0 else "FAIL"
        rows.append({
            "id": 4,
            "check": "cost_impact >= 0 in anomaly days (ground truth)",
            "status": status,
            "detail": (f"non-negative ratio = {n_nonneg/n_anom:.4f} "
                      f"({n_nonneg}/{n_anom}), positive = {n_pos}, "
                      f"negative = {n_neg}"),
        })
    else:
        rows.append({"id": 4,
                    "check": "cost_impact >= 0 in anomaly days (ground truth)",
                    "status": "FAIL", "detail": "no anomaly days found"})

    # ---------------------------------------------------------
    # Check 5. Normal cost_impact == 0
    #   ground-truth 정의상 정상 구간 비용 영향은 정확히 0.
    #   (observed residual인 excess_cost는 noise 때문에 0이 아니지만,
    #    cost_impact는 주입량만 담으므로 0이어야 한다.)
    # ---------------------------------------------------------
    normal_mask = ~anom_mask
    ci_norm = df.loc[normal_mask, "cost_impact"].values
    n_normal = len(ci_norm)
    n_zero = int((ci_norm == 0).sum())
    status5 = "PASS" if n_zero == n_normal else "FAIL"
    rows.append({
        "id": 5,
        "check": "Normal cost_impact == 0 (ground truth)",
        "status": status5,
        "detail": (f"zero count = {n_zero}/{n_normal}, "
                  f"sum_abs = {float(np.sum(np.abs(ci_norm))):.6f}"),
    })

    # ----------------------------------------------------------
    # Check 6. Events per seed (multi-seed가 있으면 분포, 아니면 단일 seed 정보)
    # ----------------------------------------------------------
    if (all_event_results_df is not None
            and len(all_event_results_df) > 0
            and "seed" in all_event_results_df.columns):
        # 같은 seed 안에서는 모델/budget과 관계없이 event 수가 같아야 한다.
        first_budget = sorted(all_event_results_df["budget"].unique())[0]
        per_seed_model = (
            all_event_results_df[all_event_results_df["budget"] == first_budget]
            .groupby(["seed", "model_name"])["event_id"]
            .nunique()
            .unstack("model_name")
        )
        per_seed = per_seed_model.iloc[:, 0]                # 모델 0의 카운트
        all_models_match = per_seed_model.eq(per_seed, axis=0).all().all()
        n_min, n_max = int(per_seed.min()), int(per_seed.max())
        n_mean = float(per_seed.mean())
        n_std = float(per_seed.std())

        status6 = "PASS" if all_models_match else "FAIL"
        rows.append({
            "id": 6,
            "check": "Events per seed (consistent across models)",
            "status": status6,
            "detail": (f"n_seeds={len(per_seed)}, "
                       f"min={n_min}, max={n_max}, "
                       f"mean={n_mean:.1f}, std={n_std:.2f}, "
                       f"all-models-match={all_models_match}"),
        })
        detail_store["events_per_seed"] = per_seed_model
    else:
        rows.append({
            "id": 6,
            "check": "Events per seed (current single seed only)",
            "status": "INFO",
            "detail": f"n_events = {len(events_df)}",
        })

    # ----------------------------------------------------------
    # Check 7. Thresholds use Year 1 score only
    #   single-seed: thresholds dict와 compute_thresholds(Year1, 99%) 재계산값 비교
    #   추가: Year 2 99th percentile과 비교해 두 값이 같지 않은지도 확인
    # ----------------------------------------------------------
    recomputed = compute_thresholds(scores_long,
                                    year2_start=year2_start,
                                    percentile=99.0)
    diffs_y1 = {k: abs(thresholds[k] - recomputed[k]) for k in thresholds}
    max_diff_y1 = max(diffs_y1.values())

    # Year 2 99th percentile (확인용)
    y2_99 = {}
    for name, sub in scores_long.groupby("model_name"):
        s_y2 = sub.loc[sub["day"] >= year2_start, "score"].dropna().values
        y2_99[name] = float(np.percentile(s_y2, 99)) if len(s_y2) else float("nan")

    # 만약 thresholds == Year 2 99th 이면 학습 누수
    diffs_y2 = {k: abs(thresholds[k] - y2_99[k]) for k in thresholds}
    matches_y2 = all(d < 1e-9 for d in diffs_y2.values())

    status7 = "PASS" if (max_diff_y1 < 1e-9 and not matches_y2) else "FAIL"
    rows.append({
        "id": 7,
        "check": "Thresholds computed on Year 1 scores only",
        "status": status7,
        "detail": (f"max|saved - Year1 99th| = {max_diff_y1:.2e}, "
                   f"identical-to-Year2-99th = {matches_y2}"),
    })
    detail_store["threshold_diffs_year1"] = diffs_y1
    detail_store["threshold_diffs_year2"] = diffs_y2

    # ----------------------------------------------------------
    # Check 8. AUPRC uses raw score (not binary alert)
    #   model_metrics_df의 auprc와
    #     (a) raw score 기준 average_precision_score
    #     (b) binary alert 기준 average_precision_score
    #   두 가지를 모두 재계산해 어느 쪽과 일치하는지 본다.
    # ----------------------------------------------------------
    from sklearn.metrics import average_precision_score
    auprc_rows = []
    for name in model_metrics_df["model_name"]:
        sub = predictions_df[
            (predictions_df["model_name"] == name)
            & (predictions_df["day"] >= year2_start)
        ]
        y_true = sub["is_anomaly"].astype(int).values
        if y_true.sum() == 0:
            auprc_rows.append({
                "model": name, "saved": float("nan"),
                "recomp_score": float("nan"),
                "recomp_alert": float("nan"),
                "diff_score": float("nan"),
                "diff_alert": float("nan"),
            })
            continue
        y_score = sub["score"].fillna(0.0).values
        y_alert = sub["alert"].astype(int).values
        recomp_score = float(average_precision_score(y_true, y_score))
        recomp_alert = float(average_precision_score(y_true, y_alert))
        saved = float(model_metrics_df.loc[
            model_metrics_df["model_name"] == name, "auprc"
        ].iloc[0])
        auprc_rows.append({
            "model": name, "saved": saved,
            "recomp_score": recomp_score,
            "recomp_alert": recomp_alert,
            "diff_score": abs(saved - recomp_score),
            "diff_alert": abs(saved - recomp_alert),
        })
    auprc_df = pd.DataFrame(auprc_rows)
    max_d_score = float(np.nanmax(auprc_df["diff_score"]))
    max_d_alert = float(np.nanmax(auprc_df["diff_alert"]))
    status8 = "PASS" if max_d_score < 1e-9 else (
        "FAIL" if max_d_alert < 1e-9 else "WARN"
    )
    rows.append({
        "id": 8,
        "check": "AUPRC uses raw score (not binary alert)",
        "status": status8,
        "detail": (f"max|saved - score-based| = {max_d_score:.2e}, "
                   f"max|saved - alert-based| = {max_d_alert:.2e}"),
    })
    detail_store["auprc_table"] = auprc_df

    summary = pd.DataFrame(rows)[["id", "check", "status", "detail"]]
    return summary, detail_store
