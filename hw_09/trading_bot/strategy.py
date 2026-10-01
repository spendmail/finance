"""Торговый слой CatBoost из ноутбука (positions_from_proba, режим hold) и правила выхода.

Параметры — лучшая связка из раздела 25 ноутбука: сигнал, подобранный Optuna для CatBoost
(hold, q=0.02, span=4, rebalance=24), и выход «только стоп-лосс» — SL 3%, без тейк-профита,
после стопа вне рынка до смены знака целевой позиции (cooldown = -1).
"""
import numpy as np
import pandas as pd

SIGNAL_PARAMS = {"mode": "hold", "q": 0.02, "span": 4, "rebalance": 24}
EXIT_PARAMS = {"sl": 0.03, "tp": 0.0, "cooldown": -1}

CATBOOST_PARAMS = {"depth": 7, "learning_rate": 0.013589328909885286,
                   "l2_leaf_reg": 3.879333946524696, "iterations": 178,
                   "rsm": 0.36060441187623177}
W_RET = True                   # веса наблюдений ∝ |fwd_ret| (лучшая попытка Optuna)
OUTLIER_WEIGHT = 0.25
WINSOR_Q = 0.001
RANDOM_STATE = 42


def smooth(p, span):
    p = np.asarray(p, dtype=float)
    return pd.Series(p).ewm(span=span, adjust=False).mean().to_numpy() if span > 1 else p


def thresholds(ref_proba, q=SIGNAL_PARAMS["q"], span=SIGNAL_PARAMS["span"]):
    """Квантильные пороги по прогнозам финальной модели на dev (apply_signal в ноутбуке)."""
    src = np.asarray(ref_proba, dtype=float)
    src = smooth(src[np.isfinite(src)], span)
    return float(np.quantile(src, 1 - q)), float(np.quantile(src, q))


def hold_signal(proba, hi, lo, span=SIGNAL_PARAMS["span"]):
    """+1 выше верхнего порога, -1 ниже нижнего, между ними — последний уверенный сигнал."""
    p = smooth(proba, span)
    sig = np.where(p >= hi, 1.0, np.where(p <= lo, -1.0, np.nan))
    return pd.Series(sig).ffill().fillna(0.0).to_numpy(), p


def target_positions(proba, hi, lo, rebalance_mask, span=SIGNAL_PARAMS["span"]):
    """Целевая позиция: hold-сигнал, пересматриваемый только на барах rebalance_mask.

    В ноутбуке ребалансировка — каждый 24-й бар от начала оцениваемого отрезка; в боте якорь
    фиксирован календарно (бар, закрывающийся в 00:00 UTC), чтобы решение не зависело от того,
    сколько истории скачано.
    """
    sig, p = hold_signal(proba, hi, lo, span)
    keep = np.asarray(rebalance_mask, dtype=bool)
    pos = pd.Series(np.where(keep, sig, np.nan)).ffill().fillna(0.0).to_numpy()
    return pos, sig, p
