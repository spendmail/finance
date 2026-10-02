"""Признаки финального CatBoost из final_ensenble_with_sltp.ipynb.

Формулы перенесены из ноутбука без изменений (разделы 3, 4, 7.1, 7.2, 7.3), но считаются только
те 18 признаков, что прошли отбор (раздел 12.2, «мягкий отбор») и на которых обучена модель.
Порядок MODEL_COLS совпадает с порядком в ноутбуке: от него зависит rsm-сэмплирование CatBoost.
"""
import numpy as np
import pandas as pd
import pywt

OHLCV = ["open", "high", "low", "close", "volume"]

MODEL_COLS = [
    # доходности
    "ret_1", "ret_2_atr", "ret_3", "ret_6", "ret_6_atr", "ret_lag_1",
    # скользящие средние
    "close_ema5_ratio", "sma5_slope", "close_sma10_ratio", "close_sma10_atr",
    # свечные паттерны (геометрия бара)
    "lower_frac", "clv", "body_atr", "shadow_imbalance",
    # паттерны продолжения
    "donchian_pos_55",
    # вейвлет и CWT
    "wl_e_d4", "cwt_e_72p",
    # новости
    "news_count_168",
]

WARMUP = 512                  # прогрев ноутбука: самое длинное окно признаков (Фурье, 512 ч)
ANOM_WIN = 168
OUTLIER_Z = 4.0
WAVELET_WINDOW = 256
WAVELET_NAME = "db4"
WAVELET_LEVEL = 5
WAVELET_MODE = "symmetric"
CWT_WAVELET = "morl"
CWT_SCALES = np.geomspace(2, 64, 16)
CWT_CHUNK = 4000
CWT_PERIODS = 1.0 / pywt.scale2frequency(CWT_WAVELET, CWT_SCALES)
NEWS_WINDOW = 168


def wilder(s, n):
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def true_range(high, low, close):
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def atr(high, low, close, n=14):
    return wilder(true_range(high, low, close), n)


def robust_z(s, n):
    """(x - медиана) / (1.4826 * MAD) по скользящему окну — как в разделе 2 ноутбука."""
    x = s.to_numpy(dtype=np.float64)
    if len(x) < n:
        return pd.Series(np.full(len(x), np.nan), index=s.index)
    windows = np.lib.stride_tricks.sliding_window_view(x, n)
    win_med = np.median(windows, axis=-1)
    win_mad = np.median(np.abs(windows - win_med[:, None]), axis=-1)
    med = pd.Series(np.concatenate([np.full(n - 1, np.nan), win_med]), index=s.index)
    mad = pd.Series(np.concatenate([np.full(n - 1, np.nan), win_mad]), index=s.index)
    return (s - med) / (1.4826 * mad).replace(0, np.nan)


def rolling_matrix(series, w):
    a = np.nan_to_num(np.asarray(series, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.lib.stride_tricks.sliding_window_view(a, w)


def align_to_bars(values, w, index):
    return pd.Series(np.concatenate([np.full(w - 1, np.nan), np.asarray(values, float)]), index=index)


def wavelet_energy_d4(logret, index):
    """wl_e_d4: доля энергии уровня D4 в db4-разложении окна 256 лог-доходностей."""
    win = rolling_matrix(logret, WAVELET_WINDOW)
    coeffs = pywt.wavedec(win, WAVELET_NAME, level=WAVELET_LEVEL, axis=1, mode=WAVELET_MODE)
    energy = np.stack([(c ** 2).sum(axis=1) for c in coeffs], axis=1)   # a5, d5, d4, d3, d2, d1
    rel = energy / (energy.sum(axis=1, keepdims=True) + 1e-300)
    return align_to_bars(rel[:, 2], WAVELET_WINDOW, index)


def cwt_energy_72p(logret, index):
    """cwt_e_72p: доля средней энергии масштабограммы Морле с периодом >= 72 ч в окне 256."""
    w = WAVELET_WINDOW
    win = rolling_matrix(logret, w)
    mean_e = np.empty((len(win), len(CWT_SCALES)))
    for s in range(0, len(win), CWT_CHUNK):
        block = win[s:s + CWT_CHUNK]
        coef, _ = pywt.cwt(block, CWT_SCALES, CWT_WAVELET, axis=1)
        mean_e[s:s + len(block)] = (np.abs(coef) ** 2).mean(axis=2).T
    share = mean_e / (mean_e.sum(axis=1, keepdims=True) + 1e-300)
    mask = CWT_PERIODS >= 72
    vals = share[:, mask].sum(axis=1) if mask.any() else np.zeros(len(win))
    return align_to_bars(vals, w, index)


def news_count_168(dates, news_bar_counts=None):
    """log1p(число заголовков за 168 ч). Без новостной ленты (после 04.06.2025 и в live) — 0,
    ровно так этот признак и считался в walk-forward проверке ноутбука (раздел 24)."""
    dates = pd.Series(pd.to_datetime(dates))
    if news_bar_counts is None or len(news_bar_counts) == 0:
        return np.zeros(len(dates))
    start = min(news_bar_counts.index.min(), dates.min())
    grid = pd.date_range(start, dates.max(), freq="h")
    cnt = news_bar_counts.reindex(grid).fillna(0.0)
    val = np.log1p(cnt.rolling(NEWS_WINDOW, min_periods=1).sum())
    return val.reindex(dates.to_numpy()).to_numpy()


def build_features(df, news_bar_counts=None):
    """df: date, open, high, low, close, volume (часовые бары, по возрастанию, без пропусков).

    Возвращает копию df с колонками MODEL_COLS и вспомогательными anom_any, fwd_ret, target.
    """
    df = df.reset_index(drop=True)
    o_, h_, l_, c_, v_ = (df[x].astype(float) for x in OHLCV)
    X = {}

    atr_s = atr(h_, l_, c_, 14)
    natr = atr_s / c_
    ret_1 = c_.pct_change()
    logret = np.log(c_).diff()

    for k in (1, 2, 3, 6):
        X[f"ret_{k}"] = c_.pct_change(k)
        X[f"ret_{k}_atr"] = (c_ / c_.shift(k) - 1) / natr.replace(0, np.nan)
    X["ret_lag_1"] = ret_1.shift(1)

    sma5, sma10 = c_.rolling(5).mean(), c_.rolling(10).mean()
    ema5 = c_.ewm(span=5, adjust=False).mean()
    X["close_ema5_ratio"] = c_ / ema5 - 1
    X["sma5_slope"] = sma5.pct_change(3)
    X["close_sma10_ratio"] = c_ / sma10 - 1
    X["close_sma10_atr"] = (c_ - sma10) / atr_s

    rng = (h_ - l_).replace(0, np.nan)
    upper = h_ - np.maximum(o_, c_)
    lower = np.minimum(o_, c_) - l_
    X["lower_frac"] = lower / rng
    X["clv"] = ((c_ - l_) - (h_ - c_)) / rng
    X["body_atr"] = (c_ - o_) / atr_s.replace(0, np.nan)
    X["shadow_imbalance"] = (upper - lower) / rng

    dh, dl = h_.rolling(55).max(), l_.rolling(55).min()
    X["donchian_pos_55"] = (c_ - dl) / (dh - dl).replace(0, np.nan)

    X["wl_e_d4"] = wavelet_energy_d4(logret, df.index)
    X["cwt_e_72p"] = cwt_energy_72p(logret, df.index)
    X["news_count_168"] = pd.Series(news_count_168(df["date"], news_bar_counts), index=df.index)

    anom_ret = robust_z(ret_1, ANOM_WIN).abs() > OUTLIER_Z
    anom_vol = robust_z(np.log1p(v_), ANOM_WIN).abs() > OUTLIER_Z
    out = df.copy()
    for c in MODEL_COLS:
        out[c] = X[c]
    out["anom_any"] = (anom_ret | anom_vol).astype(np.int8)
    out["fwd_ret"] = c_.shift(-1) / c_ - 1
    fwd = c_.shift(-1)
    out["target"] = np.where(fwd.notna(), (fwd > c_).astype(float), np.nan)
    out[MODEL_COLS] = out[MODEL_COLS].replace([np.inf, -np.inf], np.nan)
    return out


def finalize(feat, warmup=WARMUP):
    """Как в ноутбуке: отбросить прогрев, заполнить внутренние пропуски вперёд, затем нулями."""
    feat = feat.iloc[warmup:].reset_index(drop=True)
    feat[MODEL_COLS] = feat[MODEL_COLS].ffill().fillna(0)
    return feat


def load_news_bar_counts(news_path, price_start, news_end):
    """Число заголовков по часовым барам (ceil до часа) — как NEWS_BAR["n_count"] в ноутбуке."""
    news = pd.read_csv(news_path)
    news["dt"] = pd.to_datetime(news["date_time"], utc=True, errors="coerce").dt.tz_convert(None)
    news = news.dropna(subset=["dt", "title"]).sort_values("dt").reset_index(drop=True)
    news = news.drop_duplicates().reset_index(drop=True)
    news = news[(news["dt"] >= price_start - pd.Timedelta(days=40)) & (news["dt"] <= news_end)]
    return news.groupby(news["dt"].dt.ceil("h")).size().astype(float)
