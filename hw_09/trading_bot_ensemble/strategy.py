"""Стратегия «Ансамбль + SL 3% + безубыток 5%» из final_ensenble_with_sltp_strategies.ipynb.

Ансамбль — среднее вероятностей роста трёх моделей (LogisticRegression, DecisionTree, CatBoost)
с гиперпараметрами Optuna (раздел 15). Торговый слой ансамбля подобран в разделе 16: режим
always — всегда в рынке, LONG при сглаженной вероятности >= 0.5, иначе SHORT; EWM со span 12;
пересмотр позиции раз в 24 часа. Правило выхода — раздел 26.1: стоп-лосс 3% от цены входа,
без тейк-профита; как только цена ушла в плюс на 5%, со следующего часа стоп переносится в
безубыток (вход ± 0.2% — две комиссии). После любого выхода — вне рынка до смены знака цели.
"""
import numpy as np
import pandas as pd

SIGNAL_PARAMS = {"mode": "always", "span": 12, "rebalance": 24}
EXIT_PARAMS = {"sl": 0.03, "tp": 0.0, "cooldown": -1, "be": 0.05}
FEE_RATE = 0.001
BE_LOCK = 2 * FEE_RATE                 # безубыток = цена входа ± две комиссии

MODEL_PARAMS = {
    "LogisticRegression": {"C": 1.6924580548130095, "penalty": "l2", "class_weight": None,
                           "solver": "lbfgs"},
    "DecisionTree": {"max_depth": 3, "min_samples_split": 83, "min_samples_leaf": 49,
                     "max_features": 0.22153126556521793, "criterion": "entropy"},
    "CatBoost": {"depth": 7, "learning_rate": 0.013589328909885286,
                 "l2_leaf_reg": 3.879333946524696, "iterations": 178,
                 "rsm": 0.36060441187623177},
}
W_RET = True                           # у всех трёх моделей лучшая попытка Optuna — веса ∝ |fwd_ret|
OUTLIER_WEIGHT = 0.25
WINSOR_Q = 0.001
RANDOM_STATE = 42


def smooth(p, span=SIGNAL_PARAMS["span"]):
    p = np.asarray(p, dtype=float)
    return pd.Series(p).ewm(span=span, adjust=False).mean().to_numpy() if span > 1 else p


def always_signal(proba, span=SIGNAL_PARAMS["span"]):
    """Режим always: +1 при EWM(p) >= 0.5, иначе -1."""
    p = smooth(proba, span)
    return np.where(p >= 0.5, 1.0, -1.0), p


def target_positions(proba, rebalance_mask, span=SIGNAL_PARAMS["span"]):
    """Целевая позиция: сигнал always, пересматриваемый только на барах rebalance_mask."""
    sig, p = always_signal(proba, span)
    keep = np.asarray(rebalance_mask, dtype=bool)
    pos = pd.Series(np.where(keep, sig, np.nan)).ffill().fillna(0.0).to_numpy()
    return pos, sig, p
