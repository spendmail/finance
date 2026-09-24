"""Общие настройки бота: пути, параметры моделей из ноутбука, подключение к Bybit."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "btc_1h.csv"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
STATE_PATH = LOGS_DIR / "bot_state.json"
TRADES_LOG = LOGS_DIR / "trades.csv"
DECISIONS_LOG = LOGS_DIR / "decisions.csv"

RANDOM_STATE = 42
TRAIN_END, TEST_END = 0.70, 0.85     # доли ряда: train / train+test, остальное — валидация
EMBARGO = 1                          # разрыв между выборками = длине метки (HORIZON = 1)
N_CV_SPLITS = 5

# Гиперпараметры — лучшие попытки Optuna из раздела 15 ноутбука (все три с w_ret=True).
LOGREG_PARAMS = {"C": 7.085721663941605, "penalty": "l1", "class_weight": None, "solver": "saga"}
TREE_PARAMS = {"max_depth": 5, "min_samples_split": 122, "min_samples_leaf": 80,
               "max_features": 0.504713281344461, "criterion": "entropy"}
CATBOOST_PARAMS = {"depth": 5, "learning_rate": 0.02452113349961001,
                   "l2_leaf_reg": 5.753428332677497, "iterations": 165,
                   "rsm": 0.7995691107818175}
CB_FIXED = dict(iterations=300, depth=5, learning_rate=0.05)   # отбор признаков (раздел 12.2)
# Торговые слои, выбранные в разделе 15 (для сравнения на валидации).
SIGNAL_PARAMS = {
    "LogisticRegression": {"mode": "always", "span": 4, "rebalance": 24},
    "DecisionTree": {"mode": "hold", "span": 4, "rebalance": 1, "q": 0.05},
    "CatBoost": {"mode": "hold", "span": 12, "rebalance": 1, "q": 0.05},
}
BASE_MODELS = ("LogisticRegression", "DecisionTree", "CatBoost")

# RL (раздел 19): PPO, решение раз в сутки, мёртвая зона 10% капитала, 3 сида —
# вариант, выбранный в ноутбуке по test для обеих позиционных постановок.
RL_ALGO = "ppo"
RL_STEPS = 60_000
RL_SEEDS = (0, 1, 2)
RL_N_FEATURES = 24
RL_FREQ = {"every": 24, "deadband": 0.10}
RL_KWARGS = {"n_steps": 1024, "batch_size": 128, "ent_coef": 0.005, "learning_rate": 3e-4,
             "device": "cpu"}

# Сколько часовых баров бот загружает для расчёта признаков: 2000 хватает, чтобы все
# окна (до 512 баров) и экспоненциальные средние сошлись к значениям на полной истории
# (проверяется в trading_bot/selfcheck.py).
LIVE_HISTORY_BARS = 2000

SYMBOL = "BTCUSDT"
CATEGORY = "linear"
TESTNET_URL = "https://api-testnet.bybit.com"
MAINNET_URL = "https://api.bybit.com"


def load_env(path=ROOT / ".env"):
    """Мини-парсер .env: KEY=VALUE построчно, без перезаписи уже заданных переменных."""
    if not Path(path).exists():
        return
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
