"""Самопроверка бота без биржи.

1. Признаки по «живому» окну из LIVE_HISTORY_BARS баров совпадают с признаками,
   посчитанными на всей истории (иначе модель в торговле видела бы другие числа).
2. Пошаговое принятие решений LiveModel.decide (как в боте: бар за баром, с состоянием
   позиций по сидам) даёт те же позиции, что пакетный rl_rollout на валидации при
   обучении, — то есть бот исполняет именно ту политику, которая была проверена.

Запуск: ./.venv/bin/python -m trading_bot.selfcheck
"""
import numpy as np

from . import config as C
from .features import feature_frame, load_ohlcv_csv
from .signals import apply_winsor, meta_obs, rl_rollout
from .strategy import AGENT_OF, LiveModel


def main():
    raw = load_ohlcv_csv(C.DATA_PATH)
    full, cols, _, _ = feature_frame(raw)
    full = full.set_index("date")
    worst = 0.0
    for end in (len(raw), len(raw) - 3000, len(raw) - 7000):
        live, _, _, _ = feature_frame(raw.iloc[end - C.LIVE_HISTORY_BARS:end])
        a = live.iloc[-24:].set_index("date")[cols]
        b = full.loc[a.index, cols]
        worst = max(worst, float(((a - b).abs() / (b.abs() + 1e-6)).max().max()))
    print(f"[1] признаки по окну {C.LIVE_HISTORY_BARS} баров vs вся история: "
          f"макс. относительное расхождение {worst:.2e}")
    assert worst < 1e-4

    model = LiveModel()
    b = model.b
    n_steps = 24 * 10                        # 10 суток = 10 решений при every=24
    tail = full.reset_index().iloc[-n_steps:].reset_index(drop=True)
    proba = model.base_proba(tail)
    obs_batch = {
        "rl_ensemble": meta_obs(proba, b["dev_final_proba"], tail, b["meta_norm"], C.BASE_MODELS),
        "rl_position": np.clip(((apply_winsor(tail[b["rl_features"]], b["rl_bounds"]) - b["rl_mu"])
                                / b["rl_sd"]).to_numpy(dtype=np.float32), -5, 5),
    }
    for strategy, obs in obs_batch.items():
        batch = np.mean([rl_rollout(a, obs, model.freq) for a in model.agents[AGENT_OF[strategy]]], axis=0)
        seeds = [0.0] * model.n_seeds
        live = []
        for t in range(n_steps):
            if t % model.freq["every"] == 0:
                target, seeds, _ = model.decide(full.reset_index().iloc[: len(full) - n_steps + t + 1],
                                                strategy, seeds)
            live.append(float(np.mean(seeds)))
        diff = float(np.max(np.abs(np.array(live) - batch)))
        print(f"[2] {strategy}: пошаговые решения бота vs пакетный прогон политики — "
              f"макс. расхождение позиции {diff:.2e}; позиции по суткам: "
              + " ".join(f"{p:+.2f}" for p in live[::model.freq['every']]))
        assert diff < 1e-5
    print("OK")


if __name__ == "__main__":
    main()
