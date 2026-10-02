"""Обучение ансамбля ровно как в final_ensenble_with_sltp_strategies.ipynb и проверка переноса.

1. Ряд цен обрезается по концу новостной ленты (04.06.2025 14:00), считаются 18 признаков
   (MODEL_COLS), отбрасывается прогрев 512 баров; dev = первые 85% без бара-эмбарго.
2. Винзоризация по квантилям dev, веса: выбросы x0.25 и |fwd_ret|. Обучаются LogisticRegression,
   DecisionTree и CatBoost с параметрами Optuna; ансамбль — среднее их вероятностей.
3. Проверка: бэктест «SL 3% + безубыток 5%» на валидации и шести окнах walk-forward сверяется
   с таблицей раздела 26.1 ноутбука.

Запуск: python train.py  (артефакты — в artifacts/)
"""
import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier

from features import MODEL_COLS, build_features, finalize, load_news_bar_counts
from strategy import (BE_LOCK, EXIT_PARAMS, FEE_RATE, MODEL_PARAMS, OUTLIER_WEIGHT, RANDOM_STATE,
                      SIGNAL_PARAMS, W_RET, WINSOR_Q, always_signal)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "..", "data")
PRICE_PATH = os.path.join(DATA_DIR, "btc_1h.csv")
NEWS_PATH = os.path.join(DATA_DIR, "btc_news.csv")
ART_DIR = os.path.join(HERE, "artifacts")
TEST_END = 0.85
PERIODS_PER_YEAR = 24 * 365

REGIME_WINDOWS = {
    "бычий": ("2025-06-04", "2025-07-19"), "боковой": ("2025-08-06", "2025-09-20"),
    "медвежий": ("2025-10-08", "2025-11-22"), "медвежий-2": ("2026-01-15", "2026-03-01"),
    "боковой-2": ("2026-06-13", "2026-07-28"), "бычий-2": ("2026-08-13", "2026-09-27"),
}
# Раздел 26.1 ноутбука, ансамбль: net по вариантам выхода
NOTEBOOK_REF = {
    "валидация": {"sl3_be5": 0.585, "no_stops": 0.868, "sl3": 0.392},
    "бычий": {"sl3_be5": 0.053, "no_stops": 0.101, "sl3": 0.053},
    "боковой": {"sl3_be5": -0.000, "no_stops": 0.015, "sl3": -0.032},
    "медвежий": {"sl3_be5": 0.021, "no_stops": -0.226, "sl3": -0.011},
    "медвежий-2": {"sl3_be5": -0.081, "no_stops": -0.336, "sl3": -0.081},
    "боковой-2": {"sl3_be5": -0.000, "no_stops": -0.003, "sl3": -0.032},
    "бычий-2": {"sl3_be5": 0.260, "no_stops": 0.260, "sl3": 0.260},
}


def make_models():
    lr = Pipeline([
        ("scaler", RobustScaler(quantile_range=(5, 95))),
        ("clf", LogisticRegression(max_iter=500, tol=1e-3, random_state=RANDOM_STATE,
                                   **MODEL_PARAMS["LogisticRegression"])),
    ])
    tree = DecisionTreeClassifier(random_state=RANDOM_STATE, **MODEL_PARAMS["DecisionTree"])
    cb = CatBoostClassifier(verbose=False, random_state=RANDOM_STATE, allow_writing_files=False,
                            thread_count=-1, **MODEL_PARAMS["CatBoost"])
    return {"LogisticRegression": lr, "DecisionTree": tree, "CatBoost": cb}


def ensemble_proba(models, X):
    return np.mean([np.asarray(m.predict_proba(X))[:, 1] for m in models.values()], axis=0)


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


def apply_winsor(X, bounds):
    return X.clip(lower=bounds[0], upper=bounds[1], axis=1)


def sample_weights(df):
    w = np.where(df["anom_any"].to_numpy() > 0, OUTLIER_WEIGHT, 1.0)
    if W_RET:
        a = df["fwd_ret"].abs().to_numpy()
        w = w * a / a.mean()
    return w


# ── бэктест: _exit_kernel_x ноутбука (раздел 26) — стоп, тейк, безубыток ─────────────────
def exit_kernel_x(target, fwd_ret, c, o1, h1, l1, sl, tp, cooldown, be_trig, be_lock=BE_LOCK):
    n = len(target)
    pos, gross, turn, kind = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n, np.int8)
    prev, side, entry, best, be_on, blocked, exit_t = 0.0, 0, 0.0, 0.0, False, 0, -1
    use_px = sl > 0 or tp > 0 or be_trig > 0
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
            side, entry, best, be_on = ps, c[t], c[t], False
        pos[t], turn[t], gross[t] = p, abs(p - prev), p * fwd_ret[t]
        prev = p
        if use_px and side != 0:
            x, k = 0.0, 0
            if side > 0:
                stop, has, ks = (entry * (1 - sl), True, 1) if sl > 0 else (0.0, False, 1)
                if be_on and (not has or entry * (1 + be_lock) > stop):
                    stop, has, ks = entry * (1 + be_lock), True, 5
                take = entry * (1 + tp)
                if has and o1[t] <= stop: x, k = o1[t], ks
                elif tp > 0 and o1[t] >= take: x, k = o1[t], 2
                elif has and l1[t] <= stop: x, k = stop, ks
                elif tp > 0 and h1[t] >= take: x, k = take, 2
                if k == 0:
                    best = max(best, h1[t])
                    if be_trig > 0 and best >= entry * (1 + be_trig):
                        be_on = True
            else:
                stop, has, ks = (entry * (1 + sl), True, 1) if sl > 0 else (0.0, False, 1)
                if be_on and (not has or entry * (1 - be_lock) < stop):
                    stop, has, ks = entry * (1 - be_lock), True, 5
                take = entry * (1 - tp)
                if has and o1[t] >= stop: x, k = o1[t], ks
                elif tp > 0 and o1[t] <= take: x, k = o1[t], 2
                elif has and h1[t] >= stop: x, k = stop, ks
                elif tp > 0 and l1[t] <= take: x, k = take, 2
                if k == 0:
                    best = min(best, l1[t])
                    if be_trig > 0 and best <= entry * (1 - be_trig):
                        be_on = True
            if k:
                gross[t] = p * (x / c[t] - 1)
                turn[t] += abs(p)
                kind[t] = k
                prev, blocked, exit_t, side = 0.0, side, t, 0
    return pos, gross, turn, kind


def backtest(dates, fwd_ret, target, full, sl=0.0, tp=0.0, cooldown=-1, be=0.0):
    idx = np.searchsorted(full["date"].to_numpy(), pd.to_datetime(dates).to_numpy())
    c = full["close"].to_numpy()[idx]
    o1, h1, l1 = (full[k].to_numpy()[idx + 1] for k in ("open", "high", "low"))
    pos, g, turn, kind = exit_kernel_x(np.asarray(target, float), np.asarray(fwd_ret, float),
                                       c, o1, h1, l1, sl, tp, cooldown, be)
    r = g - FEE_RATE * turn
    eq = np.cumprod(1 + r)
    dd = eq / np.maximum.accumulate(np.r_[1.0, eq])[1:] - 1
    return {"net": float(eq[-1] - 1),
            "sharpe": float(r.mean() / r.std() * np.sqrt(PERIODS_PER_YEAR)) if r.std() > 0 else 0.0,
            "max_dd": float(dd.min()), "trades": int((turn > 0.01).sum()),
            "stops": int((kind == 1).sum()), "breakevens": int((kind == 5).sum()),
            "time_in_market": float((np.abs(pos) > 0.01).mean())}


def notebook_positions(proba):
    """positions_from_proba ноутбука (mode=always): пересмотр каждый 24-й бар от начала отрезка."""
    sig, _ = always_signal(proba)
    keep = (np.arange(len(sig)) % SIGNAL_PARAMS["rebalance"]) == 0
    return pd.Series(np.where(keep, sig, np.nan)).ffill().fillna(0.0).to_numpy()


def main():
    os.makedirs(ART_DIR, exist_ok=True)
    full = load_prices()
    news_end = news_end_hour()
    raw = full[full["date"] <= news_end].reset_index(drop=True)
    counts = load_news_bar_counts(NEWS_PATH, raw["date"].min(), news_end)

    feat = finalize(build_features(raw, counts))
    feat = feat.dropna(subset=["target", "fwd_ret"]).reset_index(drop=True)
    n = len(feat)
    test_end = int(n * TEST_END)
    dev, val = feat.iloc[:test_end - 1].reset_index(drop=True), feat.iloc[test_end:].reset_index(drop=True)
    print(f"Матрица {n} x {len(MODEL_COLS)} (в ноутбуке 38262); dev {len(dev)} (в ноутбуке 32521); "
          f"валидация {len(val)}")

    bounds = (dev[MODEL_COLS].quantile(WINSOR_Q), dev[MODEL_COLS].quantile(1 - WINSOR_Q))
    X_dev, y_dev, w = apply_winsor(dev[MODEL_COLS], bounds), dev["target"].astype(int), sample_weights(dev)
    models = make_models()
    for name, m in models.items():
        if hasattr(m, "steps"):
            m.fit(X_dev, y_dev, clf__sample_weight=w)
        else:
            m.fit(X_dev, y_dev, sample_weight=w)
        p_val = m.predict_proba(apply_winsor(val[MODEL_COLS], bounds))[:, 1]
        print(f"  {name}: обучена; средняя вероятность роста на валидации {p_val.mean():.3f}")

    models["CatBoost"].save_model(os.path.join(ART_DIR, "catboost.cbm"))
    with open(os.path.join(ART_DIR, "sklearn_models.pkl"), "wb") as fh:
        pickle.dump({"LogisticRegression": models["LogisticRegression"],
                     "DecisionTree": models["DecisionTree"]}, fh)
    meta = {"model_cols": MODEL_COLS, "model_params": MODEL_PARAMS, "signal_params": SIGNAL_PARAMS,
            "exit_params": EXIT_PARAMS, "be_lock": BE_LOCK,
            "winsor_lo": bounds[0].to_dict(), "winsor_hi": bounds[1].to_dict(),
            "dev_period": [str(dev["date"].min()), str(dev["date"].max())], "news_end": str(news_end)}
    with open(os.path.join(ART_DIR, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2, ensure_ascii=False)
    print(f"Артефакты сохранены в {ART_DIR}")

    # ── проверка переноса: раздел 26.1 ноутбука ─────────────────────────────────────────
    segments = {"валидация": val}
    for label, (start, end) in REGIME_WINDOWS.items():
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        lo_i = int(full.index[full["date"] >= s - pd.Timedelta(hours=1000)][0])
        hi_i = int(full.index[full["date"] <= e][-1])
        wdf = finalize(build_features(full.iloc[lo_i:hi_i + 1], counts), warmup=0)
        wdf = wdf.dropna(subset=["target", "fwd_ret"])
        segments[label] = wdf[(wdf["date"] >= s) & (wdf["date"] <= e)].reset_index(drop=True)

    rows = []
    for label, seg in segments.items():
        p = ensemble_proba(models, apply_winsor(seg[MODEL_COLS], bounds))
        target = notebook_positions(p)
        m = backtest(seg["date"], seg["fwd_ret"], target, full, **EXIT_PARAMS)
        m_no = backtest(seg["date"], seg["fwd_ret"], target, full)
        m_sl = backtest(seg["date"], seg["fwd_ret"], target, full, sl=0.03)
        ref = NOTEBOOK_REF[label]
        rows.append({"отрезок": label, "B&H": float((1 + seg["fwd_ret"]).prod() - 1),
                     "SL3+БУ5 net": m["net"], "ноутбук": ref["sl3_be5"], "Sharpe": m["sharpe"],
                     "просадка": m["max_dd"], "сделок": m["trades"], "стопов": m["stops"],
                     "безубытков": m["breakevens"], "в рынке": m["time_in_market"],
                     "без стопов": m_no["net"], "ноутбук ": ref["no_stops"],
                     "SL3": m_sl["net"], "ноутбук  ": ref["sl3"]})
    rep = pd.DataFrame(rows).set_index("отрезок")
    pd.set_option("display.width", 250)
    print("\nАнсамбль + SL 3% + безубыток 5% (и контрольные варианты для сверки с ноутбуком):")
    print(rep.round(3).to_string())
    wf = rep.drop(index="валидация")
    print(f"\nСреднее по шести окнам walk-forward: {wf['SL3+БУ5 net'].mean():+.1%} (в ноутбуке +4.2%), "
          f"B&H {wf['B&H'].mean():+.1%}")
    rep.to_csv(os.path.join(ART_DIR, "validation_report.csv"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
