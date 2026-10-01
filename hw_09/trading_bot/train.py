"""Обучение финального CatBoost ровно как в final_ensenble_with_sltp.ipynb и проверка переноса.

1. Ряд цен обрезается по концу новостной ленты (04.06.2025 14:00), считаются 18 признаков,
   отбрасывается прогрев 512 баров; dev = первые 85% (train + test) без последнего бара-эмбарго.
2. Винзоризация по квантилям dev, веса: выбросы x0.25 и |fwd_ret|, CatBoost с параметрами Optuna.
3. Пороги торгового слоя — квантили сглаженных прогнозов финальной модели на dev.
4. Проверка: бэктест на валидации и на шести окнах walk-forward сверяется с числами ноутбука.

Запуск: python train.py  (артефакты — в artifacts/)
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from features import MODEL_COLS, build_features, finalize, load_news_bar_counts
from strategy import (CATBOOST_PARAMS, EXIT_PARAMS, OUTLIER_WEIGHT, RANDOM_STATE, SIGNAL_PARAMS,
                      W_RET, WINSOR_Q, hold_signal, thresholds)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
PRICE_PATH = os.path.join(DATA_DIR, "btc_1h.csv")
NEWS_PATH = os.path.join(DATA_DIR, "btc_news.csv")
ART_DIR = os.path.join(HERE, "artifacts")
TEST_END = 0.85
FEE_RATE = 0.001
PERIODS_PER_YEAR = 24 * 365

REGIME_WINDOWS = {
    "бычий": ("2025-06-04", "2025-07-19"), "боковой": ("2025-08-06", "2025-09-20"),
    "медвежий": ("2025-10-08", "2025-11-22"), "медвежий-2": ("2026-01-15", "2026-03-01"),
    "боковой-2": ("2026-06-13", "2026-07-28"), "бычий-2": ("2026-08-13", "2026-09-27"),
}
# Числа ноутбука для CatBoost (раздел 18.3 / 24.1): net по правилам «без стопов» и «подбор»
# (SL 3% / TP 15%); при «только стоп-лосс» валидация совпадает с «без стопов».
NOTEBOOK_REF = {
    "валидация": {"no_stops": 0.717, "sl3_tp15": 0.259},
    "бычий": {"no_stops": 0.094, "sl3_tp15": 0.047},
    "боковой": {"no_stops": -0.008, "sl3_tp15": -0.032},
    "медвежий": {"no_stops": -0.311, "sl3_tp15": -0.032},
    "медвежий-2": {"no_stops": -0.280, "sl3_tp15": -0.032},
    "боковой-2": {"no_stops": -0.034, "sl3_tp15": -0.032},
}


def load_prices(path=PRICE_PATH):
    raw = pd.read_csv(path)
    raw = raw.rename(columns={c: c.lower() for c in raw.columns}).rename(columns={"datetime": "date"})
    raw["date"] = pd.to_datetime(raw["date"], utc=True).dt.tz_convert(None)
    raw = raw.sort_values("date").drop_duplicates(subset="date").reset_index(drop=True)
    return raw[["date", "open", "high", "low", "close", "volume"]]


def news_end_hour(path=NEWS_PATH):
    news = pd.read_csv(path, usecols=["date_time", "title"])
    dt = pd.to_datetime(news["date_time"], utc=True, errors="coerce").dt.tz_convert(None)
    return dt[news["title"].notna()].max().floor("h")


def winsor_bounds(X):
    return X.quantile(WINSOR_Q), X.quantile(1 - WINSOR_Q)


def apply_winsor(X, bounds):
    return X.clip(lower=bounds[0], upper=bounds[1], axis=1)


def sample_weights(df):
    w = np.where(df["anom_any"].to_numpy() > 0, OUTLIER_WEIGHT, 1.0)
    if W_RET:
        a = df["fwd_ret"].abs().to_numpy()
        w = w * a / a.mean()
    return w


# ── бэктест с SL/TP: ядро _exit_kernel ноутбука (раздел 9) ───────────────────────────────
def exit_kernel(target, fwd_ret, c, o1, h1, l1, sl, tp, cooldown):
    n = len(target)
    pos, gross, turn, kind = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, np.int8)
    prev, side, entry, blocked, exit_t = 0.0, 0, 0.0, 0, -1
    use_exits = sl > 0 or tp > 0
    for t in range(n):
        tg = target[t]
        s = 1 if tg > 0 else (-1 if tg < 0 else 0)
        if blocked != 0 and (s != blocked or (cooldown >= 0 and t - exit_t >= cooldown)):
            blocked = 0
        p = 0.0 if blocked != 0 else tg
        ps = 1 if p > 0 else (-1 if p < 0 else 0)
        if ps == 0:
            side = 0
        elif ps != side:
            side, entry = ps, c[t]
        pos[t], turn[t], gross[t] = p, abs(p - prev), p * fwd_ret[t]
        prev = p
        if use_exits and side != 0:
            x, k = 0.0, 0
            if side > 0:
                stop, take = entry * (1 - sl), entry * (1 + tp)
                if sl > 0 and o1[t] <= stop: x, k = o1[t], 1
                elif tp > 0 and o1[t] >= take: x, k = o1[t], 2
                elif sl > 0 and l1[t] <= stop: x, k = stop, 1
                elif tp > 0 and h1[t] >= take: x, k = take, 2
            else:
                stop, take = entry * (1 + sl), entry * (1 - tp)
                if sl > 0 and o1[t] >= stop: x, k = o1[t], 1
                elif tp > 0 and o1[t] <= take: x, k = o1[t], 2
                elif sl > 0 and h1[t] >= stop: x, k = stop, 1
                elif tp > 0 and l1[t] <= take: x, k = take, 2
            if k:
                gross[t] = p * (x / c[t] - 1)
                turn[t] += abs(p)
                kind[t] = k
                prev, blocked, exit_t, side = 0.0, side, t, 0
    return pos, gross, turn, kind


def backtest(dates, fwd_ret, target, full, sl, tp, cooldown):
    idx = np.searchsorted(full["date"].to_numpy(), pd.to_datetime(dates).to_numpy())
    c = full["close"].to_numpy()[idx]
    o1, h1, l1 = (full[k].to_numpy()[idx + 1] for k in ("open", "high", "low"))
    pos, g, turn, kind = exit_kernel(np.asarray(target, float), np.asarray(fwd_ret, float),
                                     c, o1, h1, l1, sl, tp, cooldown)
    r = g - FEE_RATE * turn
    eq = np.cumprod(1 + r)
    dd = eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1
    return {"net": float(eq[-1] - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR)) if r.std() > 0 else 0.0,
            "max_dd": float(dd.min()), "trades": int((turn > 0.01).sum()),
            "stops": int((kind == 1).sum()), "time_in_market": float((np.abs(pos) > 0.01).mean())}


def notebook_positions(proba, hi, lo):
    """positions_from_proba ноутбука: ребалансировка — каждый 24-й бар от начала отрезка."""
    sig, _ = hold_signal(proba, hi, lo)
    keep = (np.arange(len(sig)) % SIGNAL_PARAMS["rebalance"]) == 0
    return pd.Series(np.where(keep, sig, np.nan)).ffill().fillna(0.0).to_numpy()


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    full = load_prices()
    news_end = news_end_hour()
    raw = full[full["date"] <= news_end].reset_index(drop=True)
    counts = load_news_bar_counts(NEWS_PATH, raw["date"].min(), news_end)
    print(f"Цены: {len(full)} баров до {full['date'].max()}; обучение — до конца новостей {news_end} "
          f"({len(raw)} баров)")

    feat = finalize(build_features(raw, counts))
    feat = feat.dropna(subset=["target", "fwd_ret"]).reset_index(drop=True)
    n = len(feat)
    test_end = int(n * TEST_END)
    dev, val = feat.iloc[:test_end - 1].reset_index(drop=True), feat.iloc[test_end:].reset_index(drop=True)
    print(f"Матрица {n} x {len(MODEL_COLS)} (в ноутбуке 38262); dev {len(dev)} (в ноутбуке 32521): "
          f"{dev['date'].min()} — {dev['date'].max()}; валидация {len(val)}")

    bounds = winsor_bounds(dev[MODEL_COLS])
    X_dev = apply_winsor(dev[MODEL_COLS], bounds)
    model = CatBoostClassifier(verbose=False, random_state=RANDOM_STATE, allow_writing_files=False,
                               thread_count=-1, **CATBOOST_PARAMS)
    model.fit(X_dev, dev["target"].astype(int), sample_weight=sample_weights(dev))
    dev_proba = model.predict_proba(X_dev)[:, 1]
    hi, lo = thresholds(dev_proba)
    print(f"CatBoost обучен; пороги hold-слоя по dev: вверх >= {hi:.5f}, вниз <= {lo:.5f}")

    model.save_model(os.path.join(ART_DIR, "catboost.cbm"))
    meta = {"model_cols": MODEL_COLS, "catboost_params": CATBOOST_PARAMS,
            "signal_params": SIGNAL_PARAMS, "exit_params": EXIT_PARAMS,
            "threshold_hi": hi, "threshold_lo": lo,
            "winsor_lo": bounds[0].to_dict(), "winsor_hi": bounds[1].to_dict(),
            "dev_period": [str(dev["date"].min()), str(dev["date"].max())],
            "news_end": str(news_end)}
    with open(os.path.join(ART_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"Артефакты сохранены в {ART_DIR}")

    # ── проверка переноса: те же бэктесты, что в ноутбуке ────────────────────────────────
    rows = []
    val_p = model.predict_proba(apply_winsor(val[MODEL_COLS], bounds))[:, 1]
    segments = {"валидация": (val, val_p)}
    for label, (start, end) in REGIME_WINDOWS.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        lo_i = int(full.index[full["date"] >= s - pd.Timedelta(hours=1000)][0])
        hi_i = int(full.index[full["date"] <= e][-1])
        w = finalize(build_features(full.iloc[lo_i:hi_i + 1], counts), warmup=0)
        w = w.dropna(subset=["target", "fwd_ret"])
        w = w[(w["date"] >= s) & (w["date"] <= e)].reset_index(drop=True)
        segments[label] = (w, model.predict_proba(apply_winsor(w[MODEL_COLS], bounds))[:, 1])

    for label, (seg, p) in segments.items():
        target = notebook_positions(p, hi, lo)
        bh = float((1 + seg["fwd_ret"]).prod() - 1)
        m_sl = backtest(seg["date"], seg["fwd_ret"], target, full, **EXIT_PARAMS)
        m_no = backtest(seg["date"], seg["fwd_ret"], target, full, 0.0, 0.0, -1)
        m_15 = backtest(seg["date"], seg["fwd_ret"], target, full, 0.03, 0.15, -1)
        ref = NOTEBOOK_REF.get(label, {})
        rows.append({"отрезок": label, "B&H": bh, "SL3 net": m_sl["net"], "SL3 Sharpe": m_sl["sharpe"],
                     "SL3 DD": m_sl["max_dd"], "SL3 сделок": m_sl["trades"], "стопов": m_sl["stops"],
                     "без стопов": m_no["net"], "ноутбук": ref.get("no_stops", np.nan),
                     "SL3/TP15": m_15["net"], "ноутбук ": ref.get("sl3_tp15", np.nan)})
    rep = pd.DataFrame(rows).set_index("отрезок")
    pd.set_option("display.width", 200)
    print("\nCatBoost + SL 3% без тейка (и контрольные правила для сверки с ноутбуком):")
    print(rep.round(3).to_string())
    wf = rep.drop(index="валидация")
    print(f"\nСреднее по шести окнам walk-forward: SL3 {wf['SL3 net'].mean():+.1%} "
          f"(в ноутбуке +4.3%), B&H {wf['B&H'].mean():+.1%}")
    rep.to_csv(os.path.join(ART_DIR, "validation_report.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
