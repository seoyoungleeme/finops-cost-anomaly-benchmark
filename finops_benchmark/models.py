"""Anomaly scoring models used by the benchmark."""

import numpy as np
import pandas as pd

from .config import RANDOM_SEED, YEAR2_START

def score_with_ewma(df, events_df=None, seed=RANDOM_SEED,
                    year2_start=YEAR2_START,
                    span=14, std_window=30, eps=1e-6):
    """
    EWMA + rolling-std 기반 z-score.

    절차
        1. y에 대해 span=14 EWMA를 적용해 동적 baseline mu_t를 구한다.
        2. residual_t = y_t - mu_t.
        3. residual의 rolling std (window=std_window)을 표준편차로 사용한다.
        4. z_t = residual_t / sigma_t.
        5. score_t = |z_t|.

    Parameters
    ----------
    df          : 표준 DataFrame (Cell 11에서 생성)
    events_df   : 사용하지 않음 (인터페이스 통일을 위해 인자만 유지)
    seed        : 재현성용 (EWMA 자체는 결정론이라 사용 안 함)
    year2_start : Year 2 시작 day index
    span        : EWMA span (반응성 ↔ 평활성 트레이드오프)
    std_window  : residual rolling 표준편차 윈도우 길이
    eps         : 0 분모 방지용 작은 수

    Returns
    -------
    DataFrame [model_name, date, day, score]
    """
    y = df["y"].values.astype(float)

    # 1) EWMA baseline (adjust=False가 일반적인 EW 평활)
    s_y = pd.Series(y)
    mu = s_y.ewm(span=span, adjust=False).mean().values

    # 2) residual
    residual = y - mu

    # 3) rolling std는 Year 1의 residual로 학습한 뒤
    #    동일 규칙(rolling)을 전 구간에 적용해 동적 sigma_t를 만든다.
    res_series = pd.Series(residual)
    sigma_t = res_series.rolling(window=std_window, min_periods=std_window).std().values

    # rolling이 충족 안 되는 초반 구간은 Year 1 residual의 전체 std로 대체한다
    # (점수 계산 자체는 가능하게 두되, 워밍업 영향은 평가 시 자연 흡수됨).
    year1_residual = residual[:year2_start]
    fallback_std = float(np.nanstd(year1_residual))
    sigma_t = np.where(np.isnan(sigma_t), fallback_std, sigma_t)
    sigma_t = np.where(sigma_t < eps, eps, sigma_t)

    # 4) z-score
    z = residual / sigma_t
    score = np.abs(z)

    out = pd.DataFrame({
        "model_name": "EWMA",
        "date": df["date"].values,
        "day": df["day"].values,
        "score": score,
    })
    return out

def _build_if_features(df, year2_start=YEAR2_START):
    """
    Isolation Forest용 피처 행렬을 구성한다.

    Returns
    -------
    X         : (n, k) ndarray (NaN 가능 → 사용 시 마스크)
    valid_idx : (n,) bool, 모든 피처가 유효한 행 표시
    feat_cols : 피처 이름 리스트
    """
    y = df["y"].values.astype(float)
    dates = pd.to_datetime(df["date"].values)
    n = len(y)

    # Year 1 통계로 정규화 (Year 2 정보 누수 방지)
    mu1 = float(np.mean(y[:year2_start]))
    sd1 = float(np.std(y[:year2_start]) + 1e-6)
    y_norm = (y - mu1) / sd1

    dow = pd.Series(dates).dt.dayofweek.values
    dom = pd.Series(dates).dt.day.values
    dow_sin = np.sin(2 * np.pi * dow / 7.0)
    dow_cos = np.cos(2 * np.pi * dow / 7.0)
    dom_sin = np.sin(2 * np.pi * dom / 31.0)
    dom_cos = np.cos(2 * np.pi * dom / 31.0)

    # 안전한 lag ratio (분모가 0 근처면 NaN 처리)
    def _safe_ratio(num, den):
        out = np.full_like(num, np.nan, dtype=float)
        mask = np.abs(den) > 1e-6
        out[mask] = num[mask] / den[mask]
        return out

    lag1 = np.roll(y, 1); lag1[:1] = np.nan
    lag7 = np.roll(y, 7); lag7[:7] = np.nan
    lag1_ratio = _safe_ratio(y, lag1)
    lag7_ratio = _safe_ratio(y, lag7)

    X = np.column_stack([
        y_norm, dow_sin, dow_cos, dom_sin, dom_cos, lag1_ratio, lag7_ratio,
    ])
    feat_cols = ["y_norm", "dow_sin", "dow_cos", "dom_sin", "dom_cos",
                 "lag1_ratio", "lag7_ratio"]
    valid_idx = ~np.isnan(X).any(axis=1)
    return X, valid_idx, feat_cols

def score_with_iforest(df, events_df=None, seed=RANDOM_SEED,
                       year2_start=YEAR2_START,
                       n_estimators=200, contamination="auto"):
    """
    Isolation Forest 기반 anomaly score.

    절차
        1. 피처 구성 (Year 1 통계로 정규화).
        2. Year 1의 valid 행만으로 IsolationForest를 학습.
        3. 전체 valid 행에 대해 -decision_function을 score로 둔다 (클수록 이상).
        4. invalid 행(워밍업)은 score=NaN.

    Parameters
    ----------
    df             : 표준 DataFrame
    events_df      : 사용하지 않음
    seed           : random_state
    year2_start    : Year 2 시작 day index
    n_estimators   : 트리 수
    contamination  : sklearn 옵션 ("auto" 또는 0~0.5 부동소수)

    Returns
    -------
    DataFrame [model_name, date, day, score]
    """
    from sklearn.ensemble import IsolationForest

    n = len(df)
    X, valid_idx, _ = _build_if_features(df, year2_start=year2_start)

    # Year 1 train set: valid 행 중 day < year2_start
    days = df["day"].values
    train_mask = valid_idx & (days < year2_start)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X[train_mask])

    # 전 구간 점수: -decision_function이 클수록 이상
    score = np.full(n, np.nan, dtype=float)
    if valid_idx.any():
        score[valid_idx] = -model.decision_function(X[valid_idx])

    out = pd.DataFrame({
        "model_name": "IsolationForest",
        "date": df["date"].values,
        "day": df["day"].values,
        "score": score,
    })
    return out

def score_with_prophet(df, events_df=None, seed=RANDOM_SEED,
                       year2_start=YEAR2_START,
                       interval_width=0.99, eps=1e-6,
                       return_full=False):
    """
    Prophet 잔차 기반 anomaly score.

    절차
        1. Year 1 (day < year2_start) 데이터로 Prophet 적합.
        2. 전 구간 일자에 대해 yhat / yhat_upper 예측.
        3. residual = y - yhat.
        4. sigma_year1 = std(Year 1 residual).
        5. score = |residual| / sigma_year1.

    Parameters
    ----------
    df             : 표준 DataFrame
    events_df      : 사용하지 않음
    seed           : 재현성 (Prophet 내부 Stan 초기화에 직접 영향은 적지만 numpy 난수 통제용)
    year2_start    : Year 2 시작 day index
    interval_width : Prophet credible interval (native threshold용 보조 정보)
    eps            : 0 분모 방지
    return_full    : True면 (score_df, prophet_aux_df)를 함께 반환

    Returns
    -------
    DataFrame [model_name, date, day, score]
        return_full=True인 경우 (score_df, aux_df) 튜플.
        aux_df에는 yhat, yhat_upper가 포함되어 Part 3의 native threshold 실험에 사용된다.
    """
    # Prophet은 logging이 시끄러우므로 억제한다
    import logging
    logging.getLogger("prophet").setLevel(logging.ERROR)
    logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

    from prophet import Prophet
    np.random.seed(seed)

    # Prophet 입력 형식
    full = pd.DataFrame({
        "ds": pd.to_datetime(df["date"].values),
        "y":  df["y"].values.astype(float),
    })
    train = full.iloc[:year2_start].copy()

    m = Prophet(
        yearly_seasonality=False,   # 2년만 있어 연주기 학습은 불안정
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=interval_width,
    )
    # 월말 배치 효과를 위한 추가 계절성
    m.add_seasonality(name="monthly", period=30.5, fourier_order=5)

    m.fit(train)

    forecast = m.predict(full[["ds"]])
    yhat = forecast["yhat"].values
    yhat_upper = forecast["yhat_upper"].values

    residual = full["y"].values - yhat
    sigma1 = float(np.std(residual[:year2_start]) + eps)
    score = np.abs(residual) / sigma1

    out = pd.DataFrame({
        "model_name": "Prophet",
        "date": df["date"].values,
        "day": df["day"].values,
        "score": score,
    })

    if return_full:
        aux = pd.DataFrame({
            "date": df["date"].values,
            "day": df["day"].values,
            "yhat": yhat,
            "yhat_upper": yhat_upper,
            "residual": residual,
        })
        return out, aux
    return out

def score_with_lstm_ae(df, events_df=None, seed=RANDOM_SEED,
                       year2_start=YEAR2_START,
                       window_size=7, hidden_dim=16,
                       n_epochs=80, batch_size=32, lr=1e-2,
                       verbose=False):
    """
    LSTM Autoencoder 기반 anomaly score.

    절차
        1. y를 Year 1 평균/표준편차로 표준화한다.
        2. 길이 window_size의 sliding window를 만든다.
        3. encoder-decoder LSTM을 Year 1 윈도우(끝점 t < year2_start)로만 학습한다.
        4. 전체 윈도우에 대해 재구성 MSE를 계산하여,
           윈도우 마지막 시점 t에 score로 부여한다.
        5. 첫 (window_size - 1)일은 score=NaN.

    Parameters
    ----------
    df          : 표준 DataFrame
    events_df   : 사용하지 않음
    seed        : numpy/torch 시드
    year2_start : Year 2 시작 day index
    window_size : 슬라이딩 윈도우 길이
    hidden_dim  : LSTM hidden 크기
    n_epochs    : 학습 에폭
    batch_size  : 미니배치 크기
    lr          : Adam learning rate
    verbose     : True면 epoch별 loss 출력

    Returns
    -------
    DataFrame [model_name, date, day, score]
    """
    import torch
    import torch.nn as nn

    # 재현성
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Year 1 통계로 정규화
    y = df["y"].values.astype(np.float32)
    mu1 = float(np.mean(y[:year2_start]))
    sd1 = float(np.std(y[:year2_start]) + 1e-6)
    y_norm = (y - mu1) / sd1

    n = len(y_norm)
    if n < window_size:
        raise ValueError("series shorter than window_size")

    # 2) sliding windows: shape (n - window_size + 1, window_size, 1)
    n_win = n - window_size + 1
    end_idx = np.arange(window_size - 1, n)   # 각 윈도우의 마지막 시점
    windows = np.stack([y_norm[i:i + window_size] for i in range(n_win)], axis=0)
    # trend OOD 누수 방지: 윈도우 첫 값을 빼서 "수준"이 아니라 "모양"만 본다.
    # 이렇게 하면 Year 2의 trend 상승이 윈도우 입력 분포를 흔들지 않는다.
    windows = windows - windows[:, :1]
    windows = windows[:, :, None]             # 채널 차원 추가

    # train: 끝점이 Year 1 안에 있는 윈도우만
    train_mask = end_idx < year2_start
    train_X = torch.tensor(windows[train_mask], dtype=torch.float32, device=device)
    all_X   = torch.tensor(windows,             dtype=torch.float32, device=device)

    # 3) encoder-decoder LSTM
    class LSTMAE(nn.Module):
        def __init__(self, input_dim=1, hidden_dim=16, window=7):
            super().__init__()
            self.window = window
            self.hidden_dim = hidden_dim
            self.encoder = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.decoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.head = nn.Linear(hidden_dim, input_dim)

        def forward(self, x):
            # x: (B, T, 1)
            B = x.size(0)
            _, (h, _) = self.encoder(x)            # h: (1, B, H)
            # latent vector를 T번 반복해 decoder 입력으로 사용
            z = h.squeeze(0).unsqueeze(1).repeat(1, self.window, 1)  # (B, T, H)
            out, _ = self.decoder(z)               # (B, T, H)
            recon = self.head(out)                 # (B, T, 1)
            return recon

    model = LSTMAE(input_dim=1, hidden_dim=hidden_dim, window=window_size).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # 4) 학습 루프
    n_train = train_X.size(0)
    rng = np.random.default_rng(seed)
    model.train()
    for epoch in range(n_epochs):
        perm = rng.permutation(n_train)
        total = 0.0
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]
            xb = train_X[idx]
            recon = model(xb)
            loss = loss_fn(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  epoch {epoch + 1:3d}/{n_epochs}  train_mse={total / n_train:.6f}")

    # 5) 전체 윈도우 재구성 오차
    model.eval()
    with torch.no_grad():
        recon_all = model(all_X).cpu().numpy().squeeze(-1)   # (n_win, T)
    win_np = all_X.cpu().numpy().squeeze(-1)
    mse_win = ((recon_all - win_np) ** 2).mean(axis=1)        # (n_win,)

    # 6) day 단위 점수: 윈도우 끝점에 부여
    score = np.full(n, np.nan, dtype=float)
    score[end_idx] = mse_win

    out = pd.DataFrame({
        "model_name": "LSTM_AE",
        "date": df["date"].values,
        "day": df["day"].values,
        "score": score,
    })
    return out

def run_all_models(df, events_df, seed=RANDOM_SEED, year2_start=YEAR2_START,
                   include=("EWMA", "IsolationForest", "Prophet", "LSTM_AE"),
                   verbose=False):
    """
    4개 모델의 score를 일괄 계산하여 long-format DataFrame으로 합친다.

    Returns
    -------
    scores_long : DataFrame [model_name, date, day, score]
    aux         : dict, 모델별 보조 산출물 (예: prophet의 yhat_upper)
    """
    parts = []
    aux = {}

    if "EWMA" in include:
        if verbose:
            print("[EWMA] scoring ...")
        parts.append(score_with_ewma(df, events_df, seed=seed,
                                     year2_start=year2_start))

    if "IsolationForest" in include:
        if verbose:
            print("[IsolationForest] scoring ...")
        parts.append(score_with_iforest(df, events_df, seed=seed,
                                        year2_start=year2_start))

    if "Prophet" in include:
        if verbose:
            print("[Prophet] scoring ...")
        p_scores, p_aux = score_with_prophet(df, events_df, seed=seed,
                                             year2_start=year2_start,
                                             return_full=True)
        parts.append(p_scores)
        aux["Prophet"] = p_aux

    if "LSTM_AE" in include:
        if verbose:
            print("[LSTM_AE] scoring ...")
        parts.append(score_with_lstm_ae(df, events_df, seed=seed,
                                        year2_start=year2_start,
                                        n_epochs=60, verbose=False))

    scores_long = pd.concat(parts, axis=0, ignore_index=True)
    return scores_long, aux
