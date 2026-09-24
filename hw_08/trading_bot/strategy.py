"""Живой инференс: свечи -> признаки -> прогнозы моделей -> целевая позиция.

Три стратегии из ноутбука (все выдают долю капитала в [-1, +1]):
  rl_ensemble — RL на вершине ансамбля (раздел 19.3): PPO видит ранги прогнозов
                LogReg / дерева / CatBoost, их согласие, ret_1, atr_14 и свою позицию;
  rl_position — RL-позиция (раздел 19.2): PPO видит отобранные признаки ТА и свою позицию;
  ensemble    — среднее вероятностей трёх моделей + торговый слой (разделы 14, 16).

Агенты RL обучены на трёх сидах; как и в ноутбуке (agent_ensemble), у каждого сида своя
позиция в наблюдении, итоговая позиция — среднее по сидам.
"""
import joblib
import numpy as np
from stable_baselines3 import PPO

from . import config as C
from .features import feature_frame
from .signals import apply_winsor, meta_obs, positions_from_proba, predict_proba1

STRATEGIES = ("rl_ensemble", "rl_position", "ensemble")
AGENT_OF = {"rl_ensemble": "RL_Ensemble", "rl_position": "RL_Position"}


class LiveModel:
    def __init__(self, models_dir=C.MODELS_DIR):
        self.b = joblib.load(models_dir / "bundle.joblib")
        self.freq = self.b["rl_freq"]
        self.agents = {name: [PPO.load(models_dir / p, device="cpu") for p in paths]
                       for name, paths in self.b["agents"].items()}

    @property
    def n_seeds(self):
        return len(next(iter(self.agents.values())))

    def features(self, bars):
        """Признаки по закрытым барам; последняя строка — бар, по которому решаем."""
        feat, _, _, _ = feature_frame(bars)
        return feat

    def base_proba(self, feat_rows):
        X = apply_winsor(feat_rows[self.b["model_cols"]], self.b["dev_bounds"])
        proba = {name: predict_proba1(m, X) for name, m in self.b["models"].items()}
        proba["Ensemble"] = np.mean([proba[n] for n in C.BASE_MODELS], axis=0)
        return proba

    def _agent_obs(self, strategy, last):
        if strategy == "rl_ensemble":
            proba = self.base_proba(last)
            return meta_obs(proba, self.b["dev_final_proba"], last, self.b["meta_norm"],
                            C.BASE_MODELS)[0]
        z = (apply_winsor(last[self.b["rl_features"]], self.b["rl_bounds"])
             - self.b["rl_mu"]) / self.b["rl_sd"]
        return np.clip(z.to_numpy(dtype=np.float32), -5.0, 5.0)[0]

    def decide(self, feat, strategy, seed_positions):
        """Новая целевая позиция и диагностика.

        seed_positions — текущие позиции агентов по сидам (из состояния бота). Мёртвая зона
        применяется к каждому сиду, как в rl_rollout: перекладка меньше deadband не исполняется.
        """
        last = feat.iloc[[-1]]
        proba = {k: float(v[-1]) for k, v in self.base_proba(feat.iloc[-400:]).items()}
        info = {"bar": str(last["date"].iloc[0]), "close": float(last["close"].iloc[0]),
                "proba": proba}
        if strategy == "ensemble":
            window = feat.iloc[-400:]            # хватает для сглаживания span <= 24
            p = self.base_proba(window)["Ensemble"]
            prm = {**self.b["signal_params"]["Ensemble"], "rebalance": 1}   # частоту задаёт бот
            pos = positions_from_proba(p, ref=self.b["dev_final_proba"]["Ensemble"], **prm)
            target = float(pos[-1])
            return target, list(seed_positions), info
        obs = self._agent_obs(strategy, last)
        new_seed_pos, raw_actions = [], []
        for agent, p_old in zip(self.agents[AGENT_OF[strategy]], seed_positions):
            action, _ = agent.predict(np.concatenate([obs, np.float32([p_old])]).astype(np.float32),
                                      deterministic=True)
            a = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
            raw_actions.append(round(a, 4))
            new_seed_pos.append(a if abs(a - p_old) >= self.freq["deadband"] else p_old)
        info["raw_actions"] = raw_actions
        info["obs"] = [round(float(x), 4) for x in obs]
        return float(np.mean(new_seed_pos)), new_seed_pos, info
