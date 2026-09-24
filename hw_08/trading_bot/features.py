"""Признаки технического анализа из ноутбука final_ensenble.ipynb (разделы 2–7).

Функции перенесены из ноутбука без изменения формул — так признаки в живой торговле
совпадают с теми, на которых обучались модели. Единственное отличие — нет новостного
блока (раздел 7.3): живой ленты новостей у бота нет.

Все признаки причинны: значение на баре t зависит только от баров <= t (проверка
раздела 7.4 ноутбука и trading_bot/selfcheck.py).
"""
import numpy as np
import pandas as pd
import pywt


OHLCV = ["open", "high", "low", "close", "volume"]


N_LAGS = 12


PIVOT_W = 5               # полуокно фрактала для свинг-точек


ANOM_WIN = 168            # окно оценки «нормы» — одна неделя часовых баров


OUTLIER_Z = 4.0           # порог робастного z-score для «экстремального» бара


def wilder(s, n):
    """Сглаживание Уайлдера (RSI/ATR/ADX): экспоненциальное среднее с alpha = 1/n."""
    return s.ewm(alpha=1.0 / n, adjust=False).mean()


def true_range(high, low, close):
    """Истинный диапазон бара с учётом разрыва от предыдущего закрытия."""
    pc = close.shift(1)
    return pd.concat([high - low, (high - pc).abs(), (low - pc).abs()], axis=1).max(axis=1)


def zscore(s, n):
    return (s - s.rolling(n).mean()) / s.rolling(n).std(ddof=0).replace(0, np.nan)


def robust_z(s, n):
    """Робастная z-оценка: (x - медиана) / (1.4826 * MAD) по скользящему окну.

    1.4826 — коэффициент, при котором MAD нормального распределения равен его std,
    поэтому пороги можно трактовать в «сигмах». В отличие от обычного z-score,
    оценка не портится самими выбросами: одна свеча в ±10% не сдвигает медиану.

    MAD считается внутри каждого окна относительно ЕДИНОЙ медианы этого же окна
    (median(|x - median(x)|)). Вариант (s - rolling_median).rolling(n).median()
    сравнивал бы каждую точку с её собственной, более длинной историей медиан —
    другая статистика с другим масштабом.
    """
    x = s.to_numpy(dtype=np.float64)
    if len(x) < n:
        return pd.Series(np.full(len(x), np.nan), index=s.index)
    windows = np.lib.stride_tricks.sliding_window_view(x, n)
    win_med = np.median(windows, axis=-1)
    win_mad = np.median(np.abs(windows - win_med[:, None]), axis=-1)
    med = pd.Series(np.concatenate([np.full(n - 1, np.nan), win_med]), index=s.index)
    mad = pd.Series(np.concatenate([np.full(n - 1, np.nan), win_mad]), index=s.index)
    return (s - med) / (1.4826 * mad).replace(0, np.nan)


def rsi(close, n=14):
    """Индекс относительной силы: отношение среднего роста к среднему падению за n баров."""
    d = close.diff()
    gain = wilder(d.clip(lower=0), n)
    loss = wilder((-d).clip(lower=0), n)
    out = 100 - 100 / (1 + gain / loss.where(loss > 0))
    # loss == 0: рост без падений -> 100; полный штиль -> нейтральные 50
    return out.where(loss > 0, np.where(gain > 0, 100.0, 50.0))


def atr(high, low, close, n=14):
    """Средний истинный диапазон — базовая мера волатильности в единицах цены."""
    return wilder(true_range(high, low, close), n)


def stochastic(high, low, close, n=14, d=3):
    """Стохастик: положение закрытия внутри диапазона последних n баров."""
    ll, hh = low.rolling(n).min(), high.rolling(n).max()
    k = 100 * (close - ll) / (hh - ll).replace(0, np.nan)
    return k, k.rolling(d).mean()


def macd(close, fast=12, slow=26, signal=9):
    """Схождение/расхождение скользящих средних: линия, сигнал и гистограмма."""
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def adx(high, low, close, n=14):
    """Индекс направленного движения: сила тренда (ADX) и его направление (+DI / -DI)."""
    up, dn = high.diff(), -low.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=high.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=high.index)
    tr_n = wilder(true_range(high, low, close), n).replace(0, np.nan)
    plus_di = 100 * wilder(plus_dm, n) / tr_n
    minus_di = 100 * wilder(minus_dm, n) / tr_n
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return wilder(dx, n), plus_di, minus_di


def cci(high, low, close, n=20):
    """Индекс товарного канала: отклонение типичной цены от среднего в единицах среднего отклонения.

    Среднее отклонение считается внутри каждого окна от ЕДИНОГО среднего этого окна
    (mean(|x - mean(x)|)), а не от скользящего среднего каждой точки — иначе разные
    элементы окна сравнивались бы с разными базами.
    """
    tp = (high + low + close) / 3
    sma = tp.rolling(n).mean()
    x = tp.to_numpy(dtype=np.float64)
    if len(x) < n:
        mad = pd.Series(np.full(len(x), np.nan), index=tp.index)
    else:
        windows = np.lib.stride_tricks.sliding_window_view(x, n)
        win_mean = np.mean(windows, axis=-1)
        win_mad = np.mean(np.abs(windows - win_mean[:, None]), axis=-1)
        mad = pd.Series(np.concatenate([np.full(n - 1, np.nan), win_mad]), index=tp.index)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def mfi(high, low, close, volume, n=14):
    """Индекс денежного потока — RSI, взвешенный объёмом."""
    tp = (high + low + close) / 3
    raw_flow = tp * volume
    up = raw_flow.where(tp > tp.shift(1), 0.0).rolling(n).sum()
    dn = raw_flow.where(tp < tp.shift(1), 0.0).rolling(n).sum()
    out = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    # dn == 0: чисто положительный поток -> 100 (как в rsi()); полный штиль (up==dn==0) -> 50
    return out.where(dn > 0, np.where(up > 0, 100.0, 50.0))


def clv(high, low, close):
    """Close Location Value: положение закрытия внутри диапазона бара, от -1 до +1."""
    return ((close - low) - (high - close)) / (high - low).replace(0, np.nan)


def cmf(high, low, close, volume, n=20):
    """Денежный поток Чайкина: объём, взвешенный положением закрытия в баре."""
    return ((clv(high, low, close).fillna(0) * volume).rolling(n).sum()
            / volume.rolling(n).sum().replace(0, np.nan))


def obv(close, volume):
    """Балансовый объём: накопленный объём со знаком движения цены."""
    return (np.sign(close.diff().fillna(0)) * volume).cumsum()


def aroon(high, low, n=25):
    """Aroon: как давно был максимум и минимум окна (100 = прямо сейчас)."""
    hi = high.rolling(n + 1).apply(lambda x: float(np.argmax(x)), raw=True)
    lo = low.rolling(n + 1).apply(lambda x: float(np.argmin(x)), raw=True)
    return 100 * hi / n, 100 * lo / n


def efficiency_ratio(close, n=20):
    """Коэффициент эффективности Кауфмана: 1 — чистый тренд, 0 — «пила» без смещения."""
    direction = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    return direction / path.replace(0, np.nan)


def candle_anatomy(o_, h_, l_, c_):
    """Геометрия свечи: тело, тени и их доли в диапазоне бара."""
    rng = (h_ - l_).replace(0, np.nan)
    body = c_ - o_
    return pd.DataFrame({
        "rng": rng, "body": body, "abs_body": body.abs(),
        "upper": h_ - np.maximum(o_, c_), "lower": np.minimum(o_, c_) - l_,
        "body_frac": body.abs() / rng,
        "upper_frac": (h_ - np.maximum(o_, c_)) / rng,
        "lower_frac": (np.minimum(o_, c_) - l_) / rng,
        "clv": clv(h_, l_, c_),
    }, index=o_.index)


CANDLE_BULL = ["cdl_hammer", "cdl_inv_hammer", "cdl_dragonfly_doji", "cdl_marubozu_bull",
               "cdl_engulfing_bull", "cdl_harami_bull", "cdl_piercing", "cdl_tweezer_bottom",
               "cdl_morning_star", "cdl_three_white_soldiers", "cdl_three_inside_up",
               "cdl_belt_hold_bull", "cdl_outside_bar_bull"]


CANDLE_BEAR = ["cdl_hanging_man", "cdl_shooting_star", "cdl_gravestone_doji", "cdl_marubozu_bear",
               "cdl_engulfing_bear", "cdl_harami_bear", "cdl_dark_cloud", "cdl_tweezer_top",
               "cdl_evening_star", "cdl_three_black_crows", "cdl_three_inside_down",
               "cdl_belt_hold_bear", "cdl_outside_bar_bear"]


CANDLE_NEUTRAL = ["cdl_doji", "cdl_long_legged_doji", "cdl_spinning_top", "cdl_high_wave",
                  "cdl_inside_bar"]


CANDLE_ALL = CANDLE_BULL + CANDLE_BEAR + CANDLE_NEUTRAL


CANDLE_RU = {
    "cdl_hammer": "молот", "cdl_inv_hammer": "перевёрнутый молот",
    "cdl_dragonfly_doji": "доджи-стрекоза", "cdl_marubozu_bull": "марубозу бычий",
    "cdl_engulfing_bull": "бычье поглощение", "cdl_harami_bull": "харами бычий",
    "cdl_piercing": "пронзающая линия", "cdl_tweezer_bottom": "пинцет снизу",
    "cdl_morning_star": "утренняя звезда", "cdl_three_white_soldiers": "три белых солдата",
    "cdl_three_inside_up": "три внутри вверх", "cdl_belt_hold_bull": "бычий пояс",
    "cdl_outside_bar_bull": "внешний бар бычий", "cdl_hanging_man": "повешенный",
    "cdl_shooting_star": "падающая звезда", "cdl_gravestone_doji": "доджи-надгробие",
    "cdl_marubozu_bear": "марубозу медвежий", "cdl_engulfing_bear": "медвежье поглощение",
    "cdl_harami_bear": "харами медвежий", "cdl_dark_cloud": "завеса из тёмных облаков",
    "cdl_tweezer_top": "пинцет сверху", "cdl_evening_star": "вечерняя звезда",
    "cdl_three_black_crows": "три чёрные вороны", "cdl_three_inside_down": "три внутри вниз",
    "cdl_belt_hold_bear": "медвежий пояс", "cdl_outside_bar_bear": "внешний бар медвежий",
    "cdl_doji": "доджи", "cdl_long_legged_doji": "длинноногий доджи",
    "cdl_spinning_top": "волчок", "cdl_high_wave": "высокая волна",
    "cdl_inside_bar": "внутренний бар",
}


def candlestick_patterns(o_, h_, l_, c_, atr_s, trend_win=10):
    """31 свечной паттерн в виде бинарных признаков (0/1)."""
    a = candle_anatomy(o_, h_, l_, c_)
    rng, body, upper, lower = a["rng"], a["abs_body"], a["upper"], a["lower"]
    bull, bear = c_ > o_, c_ < o_
    body_atr = body / atr_s.replace(0, np.nan)

    o1, h1, l1, c1 = o_.shift(1), h_.shift(1), l_.shift(1), c_.shift(1)
    o2, c2 = o_.shift(2), c_.shift(2)
    bull1, bear1 = c1 > o1, c1 < o1
    bull2, bear2 = c2 > o2, c2 < o2
    body1, body2 = (c1 - o1).abs(), (c2 - o2).abs()
    rng1 = (h1 - l1).replace(0, np.nan)

    prior = c_.shift(1) / c_.shift(1 + trend_win) - 1        # тренд ДО текущего бара
    downtrend, uptrend = prior < 0, prior > 0

    long_body = body_atr > 0.8
    small_body = a["body_frac"] < 0.30
    doji_body = a["body_frac"] < 0.10

    p = {}
    # ── однобарные ───────────────────────────────────────────────────────────
    p["cdl_doji"] = doji_body
    p["cdl_long_legged_doji"] = doji_body & (upper > 0.3 * rng) & (lower > 0.3 * rng)
    p["cdl_dragonfly_doji"] = doji_body & (lower > 0.6 * rng) & (upper < 0.1 * rng)
    p["cdl_gravestone_doji"] = doji_body & (upper > 0.6 * rng) & (lower < 0.1 * rng)
    p["cdl_spinning_top"] = small_body & (upper > body) & (lower > body) & ~doji_body
    p["cdl_high_wave"] = small_body & ((upper + lower) / rng > 0.8) & (rng / atr_s > 1.5)

    hammer_shape = (lower >= 2 * body) & (upper <= 0.25 * body.where(body > 0, rng)) & (a["body_frac"] < 0.4)
    star_shape = (upper >= 2 * body) & (lower <= 0.25 * body.where(body > 0, rng)) & (a["body_frac"] < 0.4)
    p["cdl_hammer"] = hammer_shape & downtrend            # та же форма, что и повешенный,
    p["cdl_hanging_man"] = hammer_shape & uptrend         # различает только тренд до неё
    p["cdl_inv_hammer"] = star_shape & downtrend
    p["cdl_shooting_star"] = star_shape & uptrend
    p["cdl_marubozu_bull"] = bull & (a["body_frac"] > 0.90) & long_body
    p["cdl_marubozu_bear"] = bear & (a["body_frac"] > 0.90) & long_body
    p["cdl_belt_hold_bull"] = bull & (a["lower_frac"] < 0.05) & (a["body_frac"] > 0.7) & downtrend
    p["cdl_belt_hold_bear"] = bear & (a["upper_frac"] < 0.05) & (a["body_frac"] > 0.7) & uptrend

    # ── двухбарные ───────────────────────────────────────────────────────────
    p["cdl_engulfing_bull"] = bear1 & bull & (c_ >= o1) & (o_ <= c1) & (body > body1)
    p["cdl_engulfing_bear"] = bull1 & bear & (o_ >= c1) & (c_ <= o1) & (body > body1)
    p["cdl_harami_bull"] = bear1 & bull & (c_ <= o1) & (o_ >= c1) & (body < 0.6 * body1) & (body1 / atr_s > 0.8)
    p["cdl_harami_bear"] = bull1 & bear & (o_ <= c1) & (c_ >= o1) & (body < 0.6 * body1) & (body1 / atr_s > 0.8)
    # условие гэпа ослаблено до нестрогого: на крипторынке разрывов практически не бывает
    p["cdl_piercing"] = bear1 & bull & (o_ <= c1) & (c_ > (o1 + c1) / 2) & (c_ < o1) & (body1 / atr_s > 0.6)
    p["cdl_dark_cloud"] = bull1 & bear & (o_ >= c1) & (c_ < (o1 + c1) / 2) & (c_ > o1) & (body1 / atr_s > 0.6)
    tw_tol = 0.05 * atr_s
    p["cdl_tweezer_bottom"] = bear1 & bull & ((l_ - l1).abs() < tw_tol) & downtrend
    p["cdl_tweezer_top"] = bull1 & bear & ((h_ - h1).abs() < tw_tol) & uptrend
    p["cdl_inside_bar"] = (h_ < h1) & (l_ > l1)
    p["cdl_outside_bar_bull"] = (h_ > h1) & (l_ < l1) & bull
    p["cdl_outside_bar_bear"] = (h_ > h1) & (l_ < l1) & bear

    # ── трёхбарные ───────────────────────────────────────────────────────────
    small1 = body1 < 0.5 * body2
    p["cdl_morning_star"] = (bear2 & (body2 / atr_s > 0.8) & small1 & bull &
                             (c_ > (o2 + c2) / 2) & (c1 < c2))
    p["cdl_evening_star"] = (bull2 & (body2 / atr_s > 0.8) & small1 & bear &
                             (c_ < (o2 + c2) / 2) & (c1 > c2))
    p["cdl_three_white_soldiers"] = (bull2 & bull1 & bull & (c_ > c1) & (c1 > c2) &
                                     ((c1 - o1).abs() / rng1 > 0.6) & (a["body_frac"] > 0.6))
    p["cdl_three_black_crows"] = (bear2 & bear1 & bear & (c_ < c1) & (c1 < c2) &
                                  ((c1 - o1).abs() / rng1 > 0.6) & (a["body_frac"] > 0.6))
    p["cdl_three_inside_up"] = bear2 & bull1 & bull & (c1 <= o2) & (o1 >= c2) & (c_ > o2)
    p["cdl_three_inside_down"] = bull2 & bear1 & bear & (o1 <= c2) & (c1 >= o2) & (c_ < o2)

    return pd.DataFrame(p, index=o_.index).fillna(False).astype(np.int8)[CANDLE_ALL], a


def swing_points(high, low, w=PIVOT_W):
    """Свинг-экстремумы, сдвинутые на w баров вперёд — к моменту их подтверждения."""
    is_ph = (high == high.rolling(2 * w + 1, center=True).max()) & high.notna()
    is_pl = (low == low.rolling(2 * w + 1, center=True).min()) & low.notna()
    return high.where(is_ph).shift(w), low.where(is_pl).shift(w)


def chart_patterns(high, low, close, rsi_s, atr_s, w=PIVOT_W, tol=0.02, depth=0.02):
    """Паттерны разворота по подтверждённым свинг-точкам + уровни поддержки/сопротивления."""
    ph_price, pl_price = swing_points(high, low, w)
    names = ["pat_double_top", "pat_double_bottom", "pat_triple_top", "pat_triple_bottom",
             "pat_head_shoulders", "pat_inv_head_shoulders",
             "pat_bear_divergence", "pat_bull_divergence",
             "pat_higher_high", "pat_lower_high", "pat_higher_low", "pat_lower_low"]
    out = {k: np.zeros(len(close), dtype=np.int8) for k in names}

    hi_idx, hi_val = ph_price.dropna().index.to_numpy(), ph_price.dropna().to_numpy()
    lo_idx, lo_val = pl_price.dropna().index.to_numpy(), pl_price.dropna().to_numpy()
    rsi_arr = rsi_s.to_numpy()
    hi_rsi = rsi_arr[np.maximum(hi_idx - w, 0)]        # RSI в самой точке экстремума
    lo_rsi = rsi_arr[np.maximum(lo_idx - w, 0)]

    for k in range(1, len(hi_idx)):
        i, p1, p2 = hi_idx[k], hi_val[k - 1], hi_val[k]
        out["pat_higher_high"][i] = int(p2 > p1)
        out["pat_lower_high"][i] = int(p2 < p1)
        if abs(p2 - p1) / p1 <= tol:                                  # две сопоставимые вершины
            trough = lo_val[(lo_idx > hi_idx[k - 1]) & (lo_idx < i)]  # и впадина между ними
            if len(trough) and (min(p1, p2) - trough.min()) / min(p1, p2) >= depth:
                out["pat_double_top"][i] = 1
                if k >= 2 and abs(hi_val[k - 2] - p1) / p1 <= tol:
                    out["pat_triple_top"][i] = 1
        if p2 > p1 and hi_rsi[k] < hi_rsi[k - 1] - 1:                 # цена выше, RSI ниже
            out["pat_bear_divergence"][i] = 1
        if k >= 2:
            s1, head, s2 = hi_val[k - 2], hi_val[k - 1], p2
            if head > s1 and head > s2 and abs(s1 - s2) / s1 <= 1.5 * tol:
                out["pat_head_shoulders"][i] = 1

    for k in range(1, len(lo_idx)):
        i, p1, p2 = lo_idx[k], lo_val[k - 1], lo_val[k]
        out["pat_higher_low"][i] = int(p2 > p1)
        out["pat_lower_low"][i] = int(p2 < p1)
        if abs(p2 - p1) / p1 <= tol:
            peak = hi_val[(hi_idx > lo_idx[k - 1]) & (hi_idx < i)]
            if len(peak) and (peak.max() - max(p1, p2)) / max(p1, p2) >= depth:
                out["pat_double_bottom"][i] = 1
                if k >= 2 and abs(lo_val[k - 2] - p1) / p1 <= tol:
                    out["pat_triple_bottom"][i] = 1
        if p2 < p1 and lo_rsi[k] > lo_rsi[k - 1] + 1:
            out["pat_bull_divergence"][i] = 1
        if k >= 2:
            s1, head, s2 = lo_val[k - 2], lo_val[k - 1], p2
            if head < s1 and head < s2 and abs(s1 - s2) / s1 <= 1.5 * tol:
                out["pat_inv_head_shoulders"][i] = 1

    last_ph, last_pl = ph_price.ffill(), pl_price.ffill()
    levels = pd.DataFrame({
        "dist_resistance_atr": (last_ph - close) / atr_s,
        "dist_support_atr": (close - last_pl) / atr_s,
        "pivot_range_atr": (last_ph - last_pl) / atr_s,
        "pos_in_pivot_range": (close - last_pl) / (last_ph - last_pl).replace(0, np.nan),
    }, index=close.index)
    return pd.DataFrame(out, index=close.index), levels


PATTERN_RU = {
    "key_reversal_up": "ключевой разворот вверх", "key_reversal_dn": "ключевой разворот вниз",
    "breakout_up_20": "пробой канала вверх", "breakout_dn_20": "пробой канала вниз",
    "cdl_net_score": "перевес бычьих свечей", "cdl_net_score_6": "перевес бычьих свечей за 6 ч",
    "cdl_bull_score": "число бычьих свечных сигналов", "cdl_bear_score": "число медвежьих свечных сигналов",
    "pat_double_top": "двойная вершина", "pat_double_bottom": "двойное дно",
    "pat_triple_top": "тройная вершина", "pat_triple_bottom": "тройное дно",
    "pat_head_shoulders": "голова и плечи", "pat_inv_head_shoulders": "перевёрнутые голова и плечи",
    "pat_bear_divergence": "медвежья дивергенция RSI", "pat_bull_divergence": "бычья дивергенция RSI",
    "pat_higher_high": "higher high", "pat_lower_high": "lower high",
    "pat_higher_low": "higher low", "pat_lower_low": "lower low",
    "pat_bull_flag": "бычий флаг", "pat_bear_flag": "медвежий флаг", "pat_pennant": "вымпел",
}


FFT_WINDOWS = (168, 512)       # неделя и три недели часовых баров


FFT_BANDS = ((2, 4), (4, 8), (8, 24), (24, 72), (72, None))   # полосы по периоду, в часах


WAVELET_WINDOW = 256           # степень двойки: чистое разложение на 5 уровней


WAVELET_NAME = "db4"           # Добеши-4: компромисс между гладкостью и длиной фильтра


WAVELET_LEVEL = 5              # D1..D5 покрывают периоды примерно от 2 до 64 баров


WAVELET_MODE = "symmetric"     # краевое продолжение — только барами внутри окна


def rolling_matrix(series, w):
    """Матрица окон: строка i — бары [i ... i+w-1] исходного ряда.

    Результат короче ряда на w-1: у первых баров окна ещё нет. Выравнивание к бару t
    делает `align_to_bars` — здесь возвращается «сырое» представление без копий.
    """
    a = np.nan_to_num(np.asarray(series, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return np.lib.stride_tricks.sliding_window_view(a, w)


def align_to_bars(values, w, index):
    """Признак окна, заканчивающегося на баре t, кладётся в позицию t. Первые w-1 баров — NaN."""
    return pd.Series(np.concatenate([np.full(w - 1, np.nan), np.asarray(values, float)]),
                     index=index)


def fourier_features(logret, index):
    """Форма спектра скользящего окна лог-доходностей.

    Берутся характеристики распределения энергии по частотам, а не сами гармоники: амплитуда
    отдельной гармоники на финансовом ряде неустойчива и от окна к окну скачет, а форма
    спектра — устойчивая характеристика режима.
    """
    cols = {}
    for w in FFT_WINDOWS:
        win = rolling_matrix(logret, w)
        win = win - win.mean(axis=1, keepdims=True)          # снимаем среднее окна
        # окно Ханна: без него резкий обрыв на краях «размазывает» энергию по всему спектру
        spec = np.abs(np.fft.rfft(win * np.hanning(w), axis=1)) ** 2
        spec, freqs = spec[:, 1:], np.fft.rfftfreq(w)[1:]    # нулевая частота — это среднее
        periods = 1.0 / freqs
        total = spec.sum(axis=1) + 1e-300
        p = spec / total[:, None]

        for lo, hi in FFT_BANDS:
            mask = (periods >= lo) & (periods < (hi if hi is not None else np.inf))
            name = f"fft{w}_e_{lo}_{hi}" if hi is not None else f"fft{w}_e_{lo}p"
            cols[name] = align_to_bars(p[:, mask].sum(axis=1), w, index)

        # энтропия спектра: 1 — энергия размазана ровно (белый шум), 0 — вся в одной частоте
        cols[f"fft{w}_entropy"] = align_to_bars(
            -(p * np.log(p + 1e-300)).sum(axis=1) / np.log(p.shape[1]), w, index)
        # плоскостность (мера Винера): отношение среднего геометрического к арифметическому
        cols[f"fft{w}_flatness"] = align_to_bars(
            np.exp(np.log(spec + 1e-300).mean(axis=1)) / (spec.mean(axis=1) + 1e-300), w, index)
        # наклон спектра в логарифмических координатах: показатель степени в законе 1/f^a
        xc = np.log(freqs) - np.log(freqs).mean()
        yc = np.log(spec + 1e-300)
        yc = yc - yc.mean(axis=1, keepdims=True)
        cols[f"fft{w}_slope"] = align_to_bars((yc @ xc) / (xc @ xc), w, index)
        # доминирующий период и его доля в энергии
        cols[f"fft{w}_dom_period"] = align_to_bars(np.log(periods[p.argmax(axis=1)]), w, index)
        cols[f"fft{w}_dom_share"] = align_to_bars(p.max(axis=1), w, index)
        # спектральный поток: насколько форма спектра изменилась за бар
        flux = np.concatenate([[np.nan], np.sqrt(((p[1:] - p[:-1]) ** 2).sum(axis=1))])
        cols[f"fft{w}_flux"] = align_to_bars(flux, w, index)
    return cols


def wavelet_features(logret, logprice, index):
    """Многомасштабное разложение скользящего окна: энергия по уровням и вейвлет-тренд."""
    w = WAVELET_WINDOW
    win_r = rolling_matrix(logret, w)
    coeffs = pywt.wavedec(win_r, WAVELET_NAME, level=WAVELET_LEVEL, axis=1, mode=WAVELET_MODE)
    names = ["a5", "d5", "d4", "d3", "d2", "d1"]             # порядок, в котором отдаёт wavedec
    energy = np.stack([(c ** 2).sum(axis=1) for c in coeffs], axis=1)
    rel = energy / (energy.sum(axis=1, keepdims=True) + 1e-300)

    cols = {}
    for j, nm in enumerate(names):
        cols[f"wl_e_{nm}"] = align_to_bars(rel[:, j], w, index)
    cols["wl_entropy"] = align_to_bars(
        -(rel * np.log(rel + 1e-300)).sum(axis=1) / np.log(len(names)), w, index)
    # быстрые масштабы против медленных: растёт, когда рынок «дрожит», а не движется
    cols["wl_hf_lf"] = align_to_bars(
        np.log((rel[:, 5] + rel[:, 4] + 1e-9) / (rel[:, 0] + rel[:, 1] + 1e-9)), w, index)

    # последний коэффициент каждого уровня — многомасштабный аналог импульса, со знаком
    sd = win_r.std(axis=1) + 1e-12
    for j, nm in enumerate(names[1:], start=1):
        cols[f"wl_last_{nm}"] = align_to_bars(coeffs[j][:, -1] / sd, w, index)

    # доля энергии деталей ниже универсального порога — «сколько в движении шума»
    sigma = np.median(np.abs(coeffs[-1]), axis=1) / 0.6745    # робастная оценка шума по D1
    thr = (sigma * np.sqrt(2 * np.log(w)))[:, None]
    det_energy = sum((c ** 2).sum(axis=1) for c in coeffs[1:]) + 1e-300
    kept = sum((np.where(np.abs(c) > thr, c, 0.0) ** 2).sum(axis=1) for c in coeffs[1:])
    cols["wl_noise_share"] = align_to_bars(1.0 - kept / det_energy, w, index)

    # вейвлет-сглаживание цены: тренд — реконструкция по одной аппроксимации
    win_p = rolling_matrix(logprice, w)
    cp = pywt.wavedec(win_p, WAVELET_NAME, level=WAVELET_LEVEL, axis=1, mode=WAVELET_MODE)
    trend = pywt.waverec([cp[0]] + [np.zeros_like(c) for c in cp[1:]],
                         WAVELET_NAME, axis=1, mode=WAVELET_MODE)[:, :w]
    cols["wl_trend_dev"] = align_to_bars((win_p[:, -1] - trend[:, -1]) / sd, w, index)
    cols["wl_trend_slope_24"] = align_to_bars((trend[:, -1] - trend[:, -25]) / (24 * sd), w, index)
    cols["wl_trend_curv"] = align_to_bars(
        (trend[:, -1] - 2 * trend[:, -25] + trend[:, -49]) / (24 * sd), w, index)
    return cols

# Вейвлет и Фурье преобразования: масштабограмма CWT, денойзинг мягким порогом, топ-N пиков спектра.


CWT_WAVELET = "morl"                          # вейвлет Морле — как в примере с CWT


CWT_SCALES = np.geomspace(2, 64, 16)          # 16 масштабов вместо 5 уровней DWT


CWT_CHUNK = 4000                              # окна считаются частями: вся матрица не влезет


CWT_PERIODS = 1.0 / pywt.scale2frequency(CWT_WAVELET, CWT_SCALES)   # масштаб -> период, часов


FFT_TOP_PEAKS = 3                             # сколько пиков спектра брать (в примере top_n = 8)


def cwt_features(logret, index):
    """Масштабограмма скользящего окна: усреднение по времени и профиль последней точки.

    Усреднение `np.abs(coefficients).mean(axis=время)` — приём из примера: оно описывает режим
    окна целиком. Профиль последнего столбца добавлен здесь: для прогноза следующего бара важно
    не только то, каким было окно, но и то, на каком масштабе движение идёт прямо сейчас.
    """
    w = WAVELET_WINDOW
    win = rolling_matrix(logret, w)
    n_win, n_sc = len(win), len(CWT_SCALES)
    mean_e = np.empty((n_win, n_sc))
    last_e = np.empty((n_win, n_sc))
    for s in range(0, n_win, CWT_CHUNK):
        block = win[s:s + CWT_CHUNK]
        coef, _ = pywt.cwt(block, CWT_SCALES, CWT_WAVELET, axis=1)   # (масштабы, окна, время)
        energy = np.abs(coef) ** 2          # энергия — квадрат модуля, а не сам модуль: так
                                             # cwt_e_* соответствует своему названию (энергия),
                                             # как и аналогичные fft*_e_* (codex_report.md, п.27)
        mean_e[s:s + len(block)] = energy.mean(axis=2).T
        last_e[s:s + len(block)] = energy[:, :, -1].T

    share = mean_e / (mean_e.sum(axis=1, keepdims=True) + 1e-300)
    cols = {}
    for lo, hi in FFT_BANDS:
        mask = (CWT_PERIODS >= lo) & (CWT_PERIODS < (hi if hi is not None else np.inf))
        name = f"cwt_e_{lo}_{hi}" if hi is not None else f"cwt_e_{lo}p"
        cols[name] = align_to_bars(share[:, mask].sum(axis=1) if mask.any()
                                   else np.zeros(n_win), w, index)
    cols["cwt_entropy"] = align_to_bars(
        -(share * np.log(share + 1e-300)).sum(axis=1) / np.log(n_sc), w, index)
    cols["cwt_dom_period"] = align_to_bars(np.log(CWT_PERIODS[share.argmax(axis=1)]), w, index)
    # состояние «прямо сейчас»: на каком масштабе идёт движение в последней точке окна
    cols["cwt_now_dom_period"] = align_to_bars(
        np.log(CWT_PERIODS[last_e.argmax(axis=1)]), w, index)
    cols["cwt_now_power"] = align_to_bars(
        last_e.sum(axis=1) / (mean_e.sum(axis=1) + 1e-300), w, index)
    half = n_sc // 2
    cols["cwt_now_hf_lf"] = align_to_bars(
        np.log((last_e[:, :half].sum(axis=1) + 1e-12)
               / (last_e[:, half:].sum(axis=1) + 1e-12)), w, index)
    return cols


def wavelet_denoise_features(logret, index):
    """Денойзинг мягким порогом с реконструкцией — приём из «Фильтрация_ВР_вейвлетами».

    В отличие от примера порог не фиксированный, а универсальный `σ√(2 ln N)` с робастной
    оценкой σ по коэффициентам первого уровня: на ряде с меняющейся волатильностью постоянное
    число срезало бы всё на спокойном участке и ничего на бурном.
    """
    w = WAVELET_WINDOW
    win = rolling_matrix(logret, w)
    coeffs = pywt.wavedec(win, WAVELET_NAME, level=WAVELET_LEVEL, axis=1, mode=WAVELET_MODE)
    sigma = np.median(np.abs(coeffs[-1]), axis=1) / 0.6745
    thr = (sigma * np.sqrt(2 * np.log(w)))[:, None]
    denoised = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
    rec = pywt.waverec(denoised, WAVELET_NAME, axis=1, mode=WAVELET_MODE)[:, :w]
    noise = win - rec
    sd = win.std(axis=1) + 1e-12

    zeroed = sum((np.abs(c) <= thr).sum(axis=1) for c in coeffs[1:])
    n_det = sum(c.shape[1] for c in coeffs[1:])
    return {
        # отношение сигнал/шум в окне — та же величина, что печатает пример
        "wl_snr": align_to_bars(np.log((win.std(axis=1) + 1e-12) / (noise.std(axis=1) + 1e-12)),
                                w, index),
        "wl_den_last": align_to_bars(rec[:, -1] / sd, w, index),
        "wl_noise_last": align_to_bars(noise[:, -1] / sd, w, index),
        "wl_zeroed_frac": align_to_bars(zeroed / n_det, w, index),
    }


def fourier_peak_features(logret, index):
    """Периоды и доли энергии второго и третьего пиков спектра — из «Преобразование_Фурье»."""
    cols = {}
    for w in FFT_WINDOWS:
        win = rolling_matrix(logret, w)
        win = win - win.mean(axis=1, keepdims=True)
        spec = np.abs(np.fft.rfft(win * np.hanning(w), axis=1)) ** 2
        spec, freqs = spec[:, 1:], np.fft.rfftfreq(w)[1:]
        periods = 1.0 / freqs
        p = spec / (spec.sum(axis=1, keepdims=True) + 1e-300)
        top = np.argsort(p, axis=1)[:, ::-1][:, :FFT_TOP_PEAKS]      # индексы top-N пиков
        rows = np.arange(len(p))[:, None]
        for k in range(1, FFT_TOP_PEAKS):                            # первый пик уже есть в 7.1
            cols[f"fft{w}_top{k + 1}_period"] = align_to_bars(
                np.log(periods[top[:, k]]), w, index)
            cols[f"fft{w}_top{k + 1}_share"] = align_to_bars(p[rows, top][:, k], w, index)
    return cols


MA_WINDOWS = [5, 10, 20, 50, 100, 200]


RET_HORIZONS = [1, 2, 3, 6, 12, 24, 48, 72, 168]


def running_streak(sign_series):
    """Длина текущей серии одинаковых знаков: +n — n баров подряд вверх, -n — вниз."""
    s = pd.Series(sign_series).fillna(0).to_numpy()
    out = np.zeros(len(s))
    for i in range(1, len(s)):
        out[i] = out[i - 1] + s[i] if s[i] == s[i - 1] and s[i] != 0 else s[i]
    return pd.Series(out, index=pd.Series(sign_series).index)


def bars_since(flag, cap=500):
    """Сколько баров прошло с последнего срабатывания флага."""
    f = pd.Series(flag).fillna(False).to_numpy().astype(bool)
    out, last = np.empty(len(f)), -1
    for i in range(len(f)):
        if f[i]:
            last = i
        out[i] = (i - last) if last >= 0 else np.nan
    return pd.Series(np.minimum(out, cap), index=pd.Series(flag).index)


def build_ta_features(df, n_lags=N_LAGS):
    """Полная матрица признаков технического анализа. Возвращает (признаки, группы)."""
    o_, h_, l_, c_, v_ = (df[x] for x in OHLCV)
    X, groups = {}, {}

    def add(group, cols):
        groups.setdefault(group, []).extend(cols)
        X.update(cols)

    atr_s = atr(h_, l_, c_, 14)
    natr = atr_s / c_
    ret_1 = c_.pct_change()
    logret = np.log(c_).diff()

    # ── доходности и лаги ───────────────────────────────────────────────────
    cols = {}
    for k in RET_HORIZONS:
        cols[f"ret_{k}"] = c_.pct_change(k)
        cols[f"ret_{k}_atr"] = (c_ / c_.shift(k) - 1) / natr.replace(0, np.nan)   # ход в ATR
    for k in range(n_lags):
        cols[f"ret_lag_{k}"] = ret_1.shift(k)
    cols["logret_skew_24"] = logret.rolling(24).skew()
    cols["logret_kurt_24"] = logret.rolling(24).kurt()
    cols["up_bars_frac_24"] = (ret_1 > 0).rolling(24).mean()
    cols["ret_streak"] = running_streak(np.sign(ret_1.fillna(0)))
    add("доходности", cols)

    # ── скользящие средние ──────────────────────────────────────────────────
    cols, sma, ema = {}, {}, {}
    for w in MA_WINDOWS:
        sma[w] = c_.rolling(w).mean()
        ema[w] = c_.ewm(span=w, adjust=False).mean()
        cols[f"close_sma{w}_ratio"] = c_ / sma[w] - 1
        cols[f"close_ema{w}_ratio"] = c_ / ema[w] - 1
        cols[f"sma{w}_slope"] = sma[w].pct_change(max(3, w // 4))
        cols[f"close_sma{w}_atr"] = (c_ - sma[w]) / atr_s          # удалённость от MA в ATR
    for fast, slow in [(5, 20), (10, 50), (20, 50), (50, 200)]:
        spread = sma[fast] / sma[slow] - 1
        above = (spread > 0).astype(np.int8)
        up_cross = ((above == 1) & (above.shift(1) == 0)).astype(np.int8)
        dn_cross = ((above == 0) & (above.shift(1) == 1)).astype(np.int8)
        cols[f"ma_spread_{fast}_{slow}"] = spread
        cols[f"ma_above_{fast}_{slow}"] = above
        cols[f"ma_cross_up_{fast}_{slow}"] = up_cross               # «золотой крест»
        cols[f"ma_cross_dn_{fast}_{slow}"] = dn_cross               # «мёртвый крест»
        cols[f"bars_since_cross_{fast}_{slow}"] = bars_since((up_cross + dn_cross) > 0)
    ribbon = sum((sma[a] > sma[b]).astype(int) for a, b in zip(MA_WINDOWS[:-1], MA_WINDOWS[1:]))
    cols["ma_ribbon_score"] = ribbon / (len(MA_WINDOWS) - 1)        # упорядоченность «ленты» MA
    cols["price_above_all_ma"] = np.minimum.reduce([(c_ > sma[w]).to_numpy() for w in MA_WINDOWS]).astype(np.int8)
    cols["price_below_all_ma"] = np.minimum.reduce([(c_ < sma[w]).to_numpy() for w in MA_WINDOWS]).astype(np.int8)
    add("скользящие средние", cols)

    # ── осцилляторы импульса ────────────────────────────────────────────────
    cols = {}
    for n in (7, 14, 28):
        r = rsi(c_, n)
        cols[f"rsi_{n}"], cols[f"rsi_{n}_slope"] = r, r.diff(3)
    r14 = cols["rsi_14"]
    cols["rsi_overbought"] = (r14 > 70).astype(np.int8)
    cols["rsi_oversold"] = (r14 < 30).astype(np.int8)
    cols["rsi_cross_50_up"] = ((r14 > 50) & (r14.shift(1) <= 50)).astype(np.int8)
    cols["rsi_cross_50_dn"] = ((r14 < 50) & (r14.shift(1) >= 50)).astype(np.int8)
    k_, d_ = stochastic(h_, l_, c_, 14, 3)
    cols["stoch_k"], cols["stoch_d"], cols["stoch_diff"] = k_, d_, k_ - d_
    cols["williams_r"] = k_ - 100
    line, sig, hist = macd(c_)
    cols["macd"], cols["macd_signal"], cols["macd_hist"] = line / c_, sig / c_, hist / c_
    cols["macd_hist_slope"] = (hist / c_).diff(3)
    cols["macd_cross_up"] = ((hist > 0) & (hist.shift(1) <= 0)).astype(np.int8)
    cols["macd_cross_dn"] = ((hist < 0) & (hist.shift(1) >= 0)).astype(np.int8)
    adx_s, pdi, mdi = adx(h_, l_, c_, 14)
    cols["adx"], cols["di_plus"], cols["di_minus"] = adx_s, pdi, mdi
    cols["di_diff"], cols["adx_slope"] = pdi - mdi, adx_s.diff(3)
    cols["adx_strong_trend"] = (adx_s > 25).astype(np.int8)
    cols["cci_20"] = cci(h_, l_, c_, 20)
    au, ad = aroon(h_, l_, 25)
    cols["aroon_up"], cols["aroon_down"], cols["aroon_osc"] = au, ad, au - ad
    cols["mom_roc_12"] = c_.pct_change(12)
    cols["ultimate_bias"] = (c_ - (h_.rolling(24).max() + l_.rolling(24).min()) / 2) / atr_s
    add("осцилляторы", cols)

    # ── волатильность ───────────────────────────────────────────────────────
    cols = {}
    cols["atr_14"] = natr
    cols["atr_ratio_short_long"] = atr(h_, l_, c_, 7) / atr_s.replace(0, np.nan)
    cols["tr_atr"] = true_range(h_, l_, c_) / atr_s.replace(0, np.nan)
    for n in (24, 72, 168):
        cols[f"vol_{n}"] = logret.rolling(n).std(ddof=0)
    cols["vol_regime"] = cols["vol_24"] / cols["vol_168"].replace(0, np.nan)
    cols["vol_of_vol"] = cols["vol_24"].rolling(72).std(ddof=0) / cols["vol_24"].replace(0, np.nan)
    cols["parkinson_24"] = np.sqrt((np.log(h_ / l_) ** 2).rolling(24).mean() / (4 * np.log(2)))
    cols["garman_klass_24"] = np.sqrt((0.5 * np.log(h_ / l_) ** 2
                                       - (2 * np.log(2) - 1) * np.log(c_ / o_) ** 2)
                                      .rolling(24).mean().clip(lower=0))
    bb_mid, bb_std = c_.rolling(20).mean(), c_.rolling(20).std(ddof=0)
    cols["bb_pctb"] = (c_ - (bb_mid - 2 * bb_std)) / (4 * bb_std).replace(0, np.nan)
    cols["bb_width"] = (4 * bb_std) / bb_mid
    cols["bb_squeeze"] = (cols["bb_width"] < cols["bb_width"].rolling(168).quantile(0.20)).astype(np.int8)
    kc_mid = c_.ewm(span=20, adjust=False).mean()
    cols["kc_pos"] = (c_ - kc_mid) / (2 * atr_s).replace(0, np.nan)
    cols["ttm_squeeze"] = ((bb_mid + 2 * bb_std < kc_mid + 2 * atr_s) &
                           (bb_mid - 2 * bb_std > kc_mid - 2 * atr_s)).astype(np.int8)
    add("волатильность", cols)

    # ── объём ───────────────────────────────────────────────────────────────
    cols = {}
    logv_ = np.log1p(v_)
    cols["vol_z_24"], cols["vol_z_168"] = zscore(logv_, 24), zscore(logv_, 168)
    cols["vol_ratio_sma"] = v_ / v_.rolling(24).mean().replace(0, np.nan)
    cols["vol_change"] = logv_.diff()
    obv_s = obv(c_, v_)
    cols["obv_z"] = zscore(obv_s, 168)
    cols["obv_slope"] = obv_s.diff(24) / v_.rolling(24).sum().replace(0, np.nan)
    cols["mfi_14"] = mfi(h_, l_, c_, v_, 14)
    cols["cmf_20"] = cmf(h_, l_, c_, v_, 20)
    vwap = (((h_ + l_ + c_) / 3) * v_).rolling(24).sum() / v_.rolling(24).sum().replace(0, np.nan)
    cols["vwap_dev"] = c_ / vwap - 1
    cols["force_index"] = wilder(c_.diff() * v_, 13) / (atr_s * v_.rolling(24).mean()).replace(0, np.nan)
    cols["vol_price_corr_24"] = logret.rolling(24).corr(logv_.diff())
    add("объём", cols)

    # ── свечи: геометрия и паттерны ─────────────────────────────────────────
    cdl_df, anat = candlestick_patterns(o_, h_, l_, c_, atr_s)
    cols = {
        "body_frac": anat["body_frac"], "upper_frac": anat["upper_frac"],
        "lower_frac": anat["lower_frac"], "clv": anat["clv"],
        "range_atr": anat["rng"] / atr_s.replace(0, np.nan),
        "body_atr": anat["body"] / atr_s.replace(0, np.nan),
        "gap_open": o_ / c_.shift(1) - 1,
        "shadow_imbalance": (anat["upper"] - anat["lower"]) / anat["rng"],
    }
    cols.update({name: cdl_df[name] for name in cdl_df.columns})
    bull_cnt, bear_cnt = cdl_df[CANDLE_BULL].sum(axis=1), cdl_df[CANDLE_BEAR].sum(axis=1)
    cols["cdl_bull_score"], cols["cdl_bear_score"] = bull_cnt, bear_cnt
    cols["cdl_net_score"] = bull_cnt - bear_cnt
    cols["cdl_net_score_6"] = (bull_cnt - bear_cnt).rolling(6).sum()
    add("свечные паттерны", cols)

    # ── паттерны разворота ──────────────────────────────────────────────────
    pat_df, lvl = chart_patterns(h_, l_, c_, X["rsi_14"], atr_s)
    cols = {}
    for name in pat_df.columns:
        cols[name] = pat_df[name]
        cols[f"{name}_24"] = pat_df[name].rolling(24).max()      # паттерн «действует» ещё сутки
    cols.update({name: lvl[name] for name in lvl.columns})
    cols["dist_high_168_atr"] = (h_.rolling(168).max() - c_) / atr_s
    cols["dist_low_168_atr"] = (c_ - l_.rolling(168).min()) / atr_s
    cols["new_high_168"] = (c_ >= c_.rolling(168).max()).astype(np.int8)
    cols["new_low_168"] = (c_ <= c_.rolling(168).min()).astype(np.int8)
    cols["key_reversal_up"] = ((l_ < l_.shift(1).rolling(24).min()) & (c_ > c_.shift(1))).astype(np.int8)
    cols["key_reversal_dn"] = ((h_ > h_.shift(1).rolling(24).max()) & (c_ < c_.shift(1))).astype(np.int8)
    add("паттерны разворота", cols)

    # ── паттерны продолжения ────────────────────────────────────────────────
    cols = {}
    for n in (20, 55):
        dh, dl = h_.rolling(n).max(), l_.rolling(n).min()
        cols[f"donchian_pos_{n}"] = (c_ - dl) / (dh - dl).replace(0, np.nan)
        cols[f"donchian_width_{n}"] = (dh - dl) / atr_s
        cols[f"breakout_up_{n}"] = (c_ > dh.shift(1)).astype(np.int8)
        cols[f"breakout_dn_{n}"] = (c_ < dl.shift(1)).astype(np.int8)
    cols["er_20"], cols["er_72"] = efficiency_ratio(c_, 20), efficiency_ratio(c_, 72)
    impulse_up = (c_.shift(6) / c_.shift(18) - 1) > 2 * natr.shift(6)
    impulse_dn = (c_.shift(6) / c_.shift(18) - 1) < -2 * natr.shift(6)
    consolidation = (((h_.rolling(6).max() - l_.rolling(6).min()) / atr_s < 2.0) &
                     (v_.rolling(6).mean() < v_.rolling(24).mean()))
    cols["pat_bull_flag"] = (impulse_up & consolidation).astype(np.int8)
    cols["pat_bear_flag"] = (impulse_dn & consolidation).astype(np.int8)
    cols["pat_pennant"] = (consolidation & (X["bb_squeeze"] == 1)).astype(np.int8)
    cols["range_contraction"] = ((h_.rolling(6).max() - l_.rolling(6).min()) /
                                 (h_.rolling(24).max() - l_.rolling(24).min()).replace(0, np.nan))
    cols["inside_bar_streak"] = running_streak(cdl_df["cdl_inside_bar"].astype(int))
    cols["above_ema20_streak"] = running_streak((c_ > ema[20]).astype(int))
    cols["trend_continuation"] = (((X["adx"] > 25) & (X["adx_slope"] > 0) &
                                   (X["di_diff"].abs() > 5)).astype(np.int8) * np.sign(X["di_diff"]))
    add("паттерны продолжения", cols)

    # ── аномальность бара (раздел 2) ────────────────────────────────────────
    cols = {
        "anom_ret_z": robust_z(ret_1, ANOM_WIN),
        "anom_vol_z": robust_z(np.log1p(v_), ANOM_WIN),
        "anom_tr_z": robust_z(true_range(h_, l_, c_), ANOM_WIN),
    }
    cols["anom_ret_outlier"] = (cols["anom_ret_z"].abs() > OUTLIER_Z).astype(np.int8)
    cols["anom_vol_outlier"] = (cols["anom_vol_z"].abs() > OUTLIER_Z).astype(np.int8)
    # anom_any объединяет выбросы ТОЛЬКО доходности и объёма, без истинного диапазона —
    # он строго уже, чем объединение всех трёх показателей из раздела 2.1 (см. anom_flags.any()
    # там). Именно anom_any задаёт понижающий вес в sample_weights, поэтому бары, аномальные
    # только по диапазону, обучаются с полным весом.
    cols["anom_any"] = ((cols["anom_ret_outlier"] + cols["anom_vol_outlier"]) > 0).astype(np.int8)
    cols["anom_count_24"] = cols["anom_any"].rolling(24).sum()
    cols["anom_bars_since"] = bars_since(cols["anom_any"] > 0)
    add("аномалии", cols)

    # ── спектр и вейвлет-разложение (раздел 7.1) ────────────────────────────
    add("Фурье", {**fourier_features(logret, df.index),
                  **fourier_peak_features(logret, df.index)})
    add("вейвлет", {**wavelet_features(logret, np.log(c_), df.index),
                    **wavelet_denoise_features(logret, df.index)})
    add("CWT", cwt_features(logret, df.index))

    # ── новости (раздел 7.3) — в живой торговле ленты новостей нет, блок исключён.
    #    В ноутбуке из 24 новостных признаков отбор пережил один (news_count_168),
    #    и прироста качества прогноза новости не дали (раздел 23, часть II).

    # ── календарь ───────────────────────────────────────────────────────────
    hour, dow = df["date"].dt.hour, df["date"].dt.dayofweek
    add("календарь", {
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "is_weekend": (dow >= 5).astype(np.int8),
    })

    return pd.concat([df, pd.DataFrame(X, index=df.index)], axis=1), groups


def load_ohlcv_csv(path):
    """CSV с часовыми барами (как data/btc_1h.csv) -> date, open, high, low, close, volume."""
    raw = pd.read_csv(path)
    raw.columns = [c.lower() for c in raw.columns]
    raw = raw.rename(columns={"datetime": "date"})
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_convert(None)
    raw = raw.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return raw[["date"] + OHLCV]


def feature_frame(raw):
    """Признаки + fwd_ret/target, как в ячейке 37 ноутбука (без обрезки по целевой метке).

    Возвращает (feat, feature_cols, warmup): первые warmup баров, где окна ещё не
    заполнены, отброшены; редкие внутренние пропуски переносятся с предыдущего бара.
    Последний бар остаётся (fwd_ret у него NaN) — именно по нему бот принимает решение.
    """
    feat, groups = build_ta_features(raw.reset_index(drop=True))
    feature_cols = [c for g in groups.values() for c in g]
    feat = feat.copy()
    feat[feature_cols] = feat[feature_cols].replace([np.inf, -np.inf], np.nan)
    close = feat["close"]
    fwd = close.shift(-1)
    feat["target"] = np.where(fwd.notna(), (fwd > close).astype(float), np.nan)
    feat["fwd_ret"] = fwd / close - 1
    warmup = max(feat[c].first_valid_index() for c in feature_cols)
    feat = feat.iloc[warmup:].reset_index(drop=True)
    feat[feature_cols] = feat[feature_cols].ffill().fillna(0)
    return feat, feature_cols, groups, warmup
