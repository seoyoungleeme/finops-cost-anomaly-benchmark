"""Synthetic data generation for the benchmark."""

import numpy as np
import pandas as pd

try:
    from .config import (
        BASE_LEVEL,
        CONTEXTUAL_LEVELS,
        DF_COLUMNS,
        GRADUAL_LEVELS,
        MONTHLY_GROWTH,
        N_DAYS,
        N_EVENTS_HIGH,
        N_EVENTS_LOW,
        NOISE_PCT,
        RANDOM_SEED,
        SPIKE_LEVELS,
        START_DATE,
        YEAR2_START,
    )
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from config import (
        BASE_LEVEL,
        CONTEXTUAL_LEVELS,
        DF_COLUMNS,
        GRADUAL_LEVELS,
        MONTHLY_GROWTH,
        N_DAYS,
        N_EVENTS_HIGH,
        N_EVENTS_LOW,
        NOISE_PCT,
        RANDOM_SEED,
        SPIKE_LEVELS,
        START_DATE,
        YEAR2_START,
    )

def generate_baseline_series(
    n_days=N_DAYS,
    base_level=BASE_LEVEL,
    monthly_growth=MONTHLY_GROWTH,
    noise_pct=NOISE_PCT,
    start_date=START_DATE,
    seed=RANDOM_SEED,
    weekly_factor=None,
):
    """
    정상 패턴 시계열을 생성한다.

    구성요소
        trend                : 월 monthly_growth 만큼 선형 증가 (일 단위 환산)
        weekly_seasonality   : 주중 높고 주말 낮은 곱셈형 패턴
        monthly_seasonality  : 월말 3일에 +5% 배치 효과
        noise                : 기대값 대비 noise_pct 비율의 Gaussian noise

    Parameters
    ----------
    n_days        : 총 일수
    base_level    : day=0 시점의 기본 비용 수준
    monthly_growth: 월별 추세 증가율 (e.g. 0.015 == 1.5%)
    noise_pct     : 기대값 대비 노이즈 표준편차 비율
    start_date    : 시계열 시작 일자 (string)
    seed          : 재현성용 난수 시드

    Returns
    -------
    y_actual    : (n_days,) 노이즈가 포함된 실제 관측 비용
    y_expected  : (n_days,) 노이즈를 제외한 기대값 (이상 주입 전 ground-truth baseline)
    dates       : pandas DatetimeIndex
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_days)
    dates = pd.date_range(start=start_date, periods=n_days, freq="D")

    # 1) 추세: 월 monthly_growth -> 일 단위로 환산
    daily_growth = monthly_growth / 30.0
    trend = base_level * (1.0 + daily_growth * t)

    # 2) 주간 계절성: 0=Mon, ..., 6=Sun, 평균 1.0이 되도록 설계
    _default_wf = np.array([1.10, 1.15, 1.15, 1.15, 1.05, 0.75, 0.65])
    if weekly_factor is None:
        _wf = _default_wf
        dow_idx = t % 7  # assumes Monday start (default start_date 2024-01-01)
    else:
        _wf = np.asarray(weekly_factor, dtype=float)
        dow_idx = dates.dayofweek.values  # calendar-correct mapping
    weekly = trend * (_wf[dow_idx] - 1.0)

    # 3) 월말 배치 효과: 실제 달력 기준으로 일 28일 이상이면 +5%
    dom = dates.day.values
    monthly = np.where(dom >= 28, trend * 0.05, 0.0)

    # 노이즈 제외 기대값
    y_expected = trend + weekly + monthly

    # 4) Gaussian noise (기대값 대비 noise_pct 비율)
    noise = rng.normal(0.0, noise_pct, n_days) * y_expected
    y_actual = y_expected + noise

    return y_actual, y_expected, dates

def inject_anomalies(
    y_actual,
    y_expected,
    year2_start=YEAR2_START,
    spike_levels=SPIKE_LEVELS,
    contextual_levels=CONTEXTUAL_LEVELS,
    gradual_levels=GRADUAL_LEVELS,
    n_events_low=N_EVENTS_LOW,
    n_events_high=N_EVENTS_HIGH,
    seed=RANDOM_SEED,
):
    """
    Year 2 구간에 spike / contextual / gradual 이상을 주입한다.

    이벤트 규칙
        - 모든 이벤트는 Year 2 (day >= year2_start) 안에만 위치한다.
        - 이벤트끼리 서로 겹치지 않는다 (양쪽 buffer 포함).
        - 각 (anomaly_type, intensity_level) 조합당 n_events_low~n_events_high 회 주입.
        - 슬롯 탐색 실패 시 해당 시도는 건너뛴다.

    Parameters
    ----------
    y_actual          : (n,) 노이즈 포함 정상 시계열
    y_expected        : (n,) 노이즈 제외 ground-truth baseline
    year2_start       : Year 2 시작 day index
    spike_levels      : {"low": 0.30, "mid": 1.00, "high": 3.00}
    contextual_levels : {"low": 0.50, "mid": 1.00, "high": 2.00}
    gradual_levels    : {"low": 0.03, "mid": 0.05, "high": 0.10}
    seed              : 재현성용 난수 시드

    Returns
    -------
    y_modified     : (n,) 이상 주입 후 시계열
    excess_cost    : (n,) y_modified - y_expected
    is_anomaly     : (n,) bool
    anomaly_type   : (n,) object, {"none","spike","contextual","gradual"}
    intensity_level: (n,) object, {"none","low","mid","high"}
    event_id_arr   : (n,) int (정상=0, 이상이면 1부터)
    events_df      : 이벤트 메타 DataFrame
                     [event_id, anomaly_type, intensity_level,
                      start_day, end_day, total_excess_cost]
    """
    # 주입 단계의 난수는 baseline 생성 단계와 분리되도록 offset을 둔다.
    rng = np.random.default_rng(seed + 10_000)
    n = len(y_actual)

    y_modified = y_actual.copy()
    is_anomaly = np.zeros(n, dtype=bool)
    anomaly_type = np.array(["none"] * n, dtype=object)
    intensity_level = np.array(["none"] * n, dtype=object)
    event_id_arr = np.zeros(n, dtype=np.int64)
    injected_cost = np.zeros(n, dtype=float)

    # 이미 점유된 day 표시. Year 1 전체는 보호.
    occupied = np.zeros(n, dtype=bool)
    occupied[:year2_start] = True

    events = []
    next_event_id = 1

    def _find_slot(length, buffer, weekend_start=False, max_tries=300):
        """
        Year 2 구간에서 길이 length짜리 비점유 슬롯을 찾는다.
        weekend_start=True면 토요일에서 시작 (dow == 5).
        """
        for _ in range(max_tries):
            start = int(rng.integers(year2_start, n - length - buffer))
            if weekend_start and (start % 7) != 5:
                continue
            lo = max(0, start - buffer)
            hi = min(n, start + length + buffer)
            if not occupied[lo:hi].any():
                return start
        return None

    def _mark_occupied(start, length, buffer):
        """슬롯을 점유 처리 (양쪽 buffer 포함)."""
        lo = max(0, start - buffer)
        hi = min(n, start + length + buffer)
        occupied[lo:hi] = True

    # ---------- 1) Spike ----------
    # 1~2일 급격 증가. baseline 대비 +mult.
    for level, mult in spike_levels.items():
        n_inject = int(rng.integers(n_events_low, n_events_high))
        for _ in range(n_inject):
            length = int(rng.integers(1, 3))     # 1 또는 2일
            start = _find_slot(length, buffer=2)
            if start is None:
                continue
            for k in range(length):
                idx = start + k
                add = y_expected[idx] * mult
                y_modified[idx] = y_actual[idx] + add
                injected_cost[idx] = add
                is_anomaly[idx] = True
                anomaly_type[idx] = "spike"
                intensity_level[idx] = level
                event_id_arr[idx] = next_event_id
            _mark_occupied(start, length, buffer=2)
            events.append({
                "event_id": next_event_id,
                "anomaly_type": "spike",
                "intensity_level": level,
                "start_day": start,
                "end_day": start + length - 1,
            })
            next_event_id += 1

    # ---------- 2) Contextual ----------
    # 주말에 주중 수준 비용. 주말 baseline 대비 +mult.
    # 토요일에서 시작, 길이 1 또는 2 (Sat 또는 Sat+Sun).
    for level, mult in contextual_levels.items():
        n_inject = int(rng.integers(n_events_low, n_events_high))
        for _ in range(n_inject):
            length = int(rng.integers(1, 3))
            start = _find_slot(length, buffer=2, weekend_start=True)
            if start is None:
                continue
            for k in range(length):
                idx = start + k
                add = y_expected[idx] * mult
                y_modified[idx] = y_actual[idx] + add
                injected_cost[idx] = add
                is_anomaly[idx] = True
                anomaly_type[idx] = "contextual"
                intensity_level[idx] = level
                event_id_arr[idx] = next_event_id
            _mark_occupied(start, length, buffer=2)
            events.append({
                "event_id": next_event_id,
                "anomaly_type": "contextual",
                "intensity_level": level,
                "start_day": start,
                "end_day": start + length - 1,
            })
            next_event_id += 1

    # ---------- 3) Gradual ----------
    # 7~14일 동안 일일 누적 +daily_pct. day k 시점 추가량 = baseline * daily_pct * (k+1).
    for level, daily_pct in gradual_levels.items():
        n_inject = int(rng.integers(n_events_low, n_events_high))
        for _ in range(n_inject):
            length = int(rng.integers(7, 15))    # 7~14일
            start = _find_slot(length, buffer=3)
            if start is None:
                continue
            for k in range(length):
                idx = start + k
                add = y_expected[idx] * daily_pct * (k + 1)
                y_modified[idx] = y_actual[idx] + add
                injected_cost[idx] = add
                is_anomaly[idx] = True
                anomaly_type[idx] = "gradual"
                intensity_level[idx] = level
                event_id_arr[idx] = next_event_id
            _mark_occupied(start, length, buffer=3)
            events.append({
                "event_id": next_event_id,
                "anomaly_type": "gradual",
                "intensity_level": level,
                "start_day": start,
                "end_day": start + length - 1,
            })
            next_event_id += 1

    # ---------- excess_cost / event total ----------
    # observed residual: 관측-기대값 (noise 포함, 호환성 위해 유지)
    excess_cost = y_modified - y_expected
    # ground-truth cost impact: 합성 시 주입한 정확한 비용. 정상 구간은 0.
    cost_impact = injected_cost.copy()

    for ev in events:
        s, e = ev["start_day"], ev["end_day"]
        ev["total_excess_cost"] = float(cost_impact[s:e + 1].sum())

    events_df = (
        pd.DataFrame(events)
        .sort_values("start_day")
        .reset_index(drop=True)
    )

    return (
    y_modified, excess_cost, cost_impact,
    is_anomaly, anomaly_type, intensity_level, event_id_arr,
    events_df,
    )

def build_dataset(seed=RANDOM_SEED, n_days=N_DAYS, year2_start=YEAR2_START,
                  start_date=START_DATE):
    """
    합성 데이터셋 한 벌(시계열 + event table)을 생성하여 표준 형태로 반환한다.

    Parameters
    ----------
    seed       : 재현성용 시드 (baseline + 이상 주입 둘 다 통제)
    n_days     : 총 일수
    year2_start: Year 2 시작 day index
    start_date : 시계열 시작 일자

    Returns
    -------
    df         : pandas.DataFrame (n_days rows, 표준 컬럼)
    events_df  : pandas.DataFrame (이벤트 단위 메타)
    """
    # 1) 정상 시계열
    y_actual, y_expected, dates = generate_baseline_series(
        n_days=n_days, start_date=start_date, seed=seed,
    )

    # 2) 이상 주입
    (y_mod, excess, cost_imp, is_anom, a_type, a_int, ev_id, events_df) = inject_anomalies(
    y_actual, y_expected,
    year2_start=year2_start,
    seed=seed,
    )

    # 3) 표준 DataFrame 구성
    df = pd.DataFrame({
    "date": dates,
    "day": np.arange(n_days),
    "y": y_mod,
    "y_expected_baseline": y_expected,
    "is_anomaly": is_anom,
    "anomaly_type": a_type,
    "intensity_level": a_int,
    "event_id": ev_id,
    "excess_cost": excess,           # observed residual (호환성용, 진단/시각화에 사용)
    "cost_impact": cost_imp,         # ground-truth 주입 비용 (평가의 진짜 기준)
    })
    df["score"] = np.nan
    df["alert"] = False
    # DF_COLUMNS는 Cell 5에 정의되어 있음. 아래 한 줄로 동적 확장.
    extended_cols = list(DF_COLUMNS)
    if "cost_impact" not in extended_cols:
        insert_at = extended_cols.index("excess_cost") + 1
        extended_cols.insert(insert_at, "cost_impact")
    df = df[extended_cols]

    return df, events_df


def build_focus_calibrated_dataset(
    stats,
    seed=RANDOM_SEED,
    n_days=N_DAYS,
    year2_start=YEAR2_START,
    start_date=START_DATE,
):
    """Build a benchmark dataset using statistics calibrated from real FOCUS data.

    Generates a 730-day synthetic baseline whose trend, noise, and DoW
    seasonality reflect the real-world parameters in *stats* (produced by
    ``focus_calibration.fit_series_statistics``), then injects synthetic
    anomalies so that ground-truth labels are available for full evaluation.

    Parameters
    ----------
    stats : dict
        Keys: ``base_level``, ``monthly_growth``, ``noise_pct``,
        ``weekly_factor`` (list of 7 DoW multipliers).
    seed, n_days, year2_start, start_date : same as ``build_dataset``.

    Returns
    -------
    df, events_df : same schema as ``build_dataset``.
    """
    y_actual, y_expected, dates = generate_baseline_series(
        n_days=n_days,
        base_level=stats["base_level"],
        monthly_growth=stats["monthly_growth"],
        noise_pct=stats["noise_pct"],
        start_date=start_date,
        seed=seed,
        weekly_factor=stats.get("weekly_factor"),
    )

    (y_mod, excess, cost_imp, is_anom, a_type, a_int, ev_id, events_df) = inject_anomalies(
        y_actual, y_expected,
        year2_start=year2_start,
        seed=seed,
    )

    df = pd.DataFrame({
        "date": dates,
        "day": np.arange(n_days),
        "y": y_mod,
        "y_expected_baseline": y_expected,
        "is_anomaly": is_anom,
        "anomaly_type": a_type,
        "intensity_level": a_int,
        "event_id": ev_id,
        "excess_cost": excess,
        "cost_impact": cost_imp,
    })
    df["score"] = np.nan
    df["alert"] = False

    extended_cols = list(DF_COLUMNS)
    if "cost_impact" not in extended_cols:
        insert_at = extended_cols.index("excess_cost") + 1
        extended_cols.insert(insert_at, "cost_impact")
    df = df[extended_cols]

    return df, events_df
