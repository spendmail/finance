"""Обучение моделей бота по протоколу ноутбука final_ensenble.ipynb.

Что воспроизводится:
  * признаки ТА разделов 2–7 (без новостей — у бота нет живой ленты);
  * разбиение train / test / validation 70 / 15 / 15 с embargo;
  * отбор признаков раздела 12.2 (чистка корреляций + признаки-тени) только по train;
  * LogReg, дерево и CatBoost с гиперпараметрами Optuna из раздела 15, винзоризация
    и веса наблюдений раздела 11, OOF на dev для эталонных распределений;
  * ансамбль (среднее вероятностей) и его торговый слой (раздел 14);
  * RL-позиция (раздел 19.2) и RL на вершине ансамбля (раздел 19.3): PPO, k=24, δ=0.10,
    три сида, обучение на dev, одна проверка на валидации.

Запуск:  ./.venv/bin/python -m trading_bot.train [--csv data/btc_1h.csv]
Результат: models/bundle.joblib, models/ppo_*.zip, models/report.json.
"""
import argparse
import json
import random
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
from stable_baselines3 import PPO

from . import config as C
from .features import feature_frame, load_ohlcv_csv
from .signals import (PositionTradingEnv, apply_winsor, backtest, classification_metrics,
                      meta_norm_stats, meta_obs, oof_fold_ids, positions_from_proba, predict_proba1,
                      rl_rollout, sample_weights, tune_signal_layer, winsor_bounds)

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def make_logreg(**params):
    return Pipeline([
        ("scaler", RobustScaler(quantile_range=(5, 95))),
        ("clf", LogisticRegression(max_iter=500, tol=1e-3, random_state=C.RANDOM_STATE, **params)),
    ])


def make_tree(**params):
    return DecisionTreeClassifier(random_state=C.RANDOM_STATE, **params)


def make_catboost(**params):
    return CatBoostClassifier(verbose=False, random_state=C.RANDOM_STATE,
                              allow_writing_files=False, thread_count=-1, **params)


BUILDERS = {
    "LogisticRegression": lambda: make_logreg(**C.LOGREG_PARAMS),
    "DecisionTree": lambda: make_tree(**C.TREE_PARAMS),
    "CatBoost": lambda: make_catboost(**C.CATBOOST_PARAMS),
}


def fit_weighted(model, X, y, w):
    if hasattr(model, "steps"):
        model.fit(X, y, **{f"{model.steps[-1][0]}__sample_weight": w})
    else:
        model.fit(X, y, sample_weight=w)
    return model


def cv_oof(build_model, data, cols, splitter, w_ret=True):
    """OOF-вероятности на dev: винзоризация и веса — по обучающей части каждого фолда."""
    X, y = data[cols], data["target"]
    oof = np.full(len(data), np.nan)
    for tr_idx, te_idx in splitter.split(X):
        b = winsor_bounds(X.iloc[tr_idx])
        X_tr, X_te = apply_winsor(X.iloc[tr_idx], b), apply_winsor(X.iloc[te_idx], b)
        model = fit_weighted(build_model(), X_tr, y.iloc[tr_idx],
                             sample_weights(data, tr_idx, True, w_ret))
        oof[te_idx] = predict_proba1(model, X_te)
    return oof, ~np.isnan(oof)


def select_features(train_df, dev_df, feature_cols, splitter, log):
    """Раздел 12.2: чистка по корреляции + признаки-тени, только по train_df."""
    X_sel = train_df[feature_cols]
    y_sel = train_df["target"].to_numpy()
    w_sel = sample_weights(train_df, np.arange(len(train_df)), w_outlier=True)
    uni = pd.Series({c: abs(roc_auc_score(y_sel, X_sel[c].to_numpy()) - 0.5) for c in feature_cols})
    corr = X_sel.corr().abs().to_numpy()
    pos = {c: i for i, c in enumerate(feature_cols)}
    kept = []
    for c in uni.sort_values(ascending=False).index:
        if all(corr[pos[c], pos[k]] < 0.95 for k in kept):
            kept.append(c)
    log(f"  чистка по корреляции: {len(feature_cols)} -> {len(kept)}")
    votes = pd.Series(0, index=kept)
    votes_soft = pd.Series(0, index=kept)
    X_kept = X_sel[kept].reset_index(drop=True)
    for r in range(3):
        rng = np.random.default_rng(C.RANDOM_STATE + r)
        shadow = pd.DataFrame({f"shadow__{c}": rng.permutation(X_kept[c].to_numpy()) for c in kept})
        both = pd.concat([X_kept, shadow], axis=1)
        model = make_catboost(**C.CB_FIXED).fit(both, y_sel, sample_weight=w_sel)
        imp = pd.Series(model.get_feature_importance(), index=both.columns)
        sh = imp[[f"shadow__{c}" for c in kept]]
        votes += (imp[kept] > sh.max()).astype(int)
        votes_soft += (imp[kept] > sh.quantile(0.95)).astype(int)
    strict = [c for c in feature_cols if votes.get(c, 0) >= 2]
    soft = [c for c in feature_cols if votes_soft.get(c, 0) >= 2]
    scores = {}
    for label, cols in (("мягкий", soft), ("строгий", strict)):
        if not cols:
            scores[label] = 0.0
            continue
        oof, m = cv_oof(lambda: make_catboost(**C.CB_FIXED), dev_df, cols, splitter, w_ret=False)
        scores[label] = roc_auc_score(dev_df["target"].to_numpy()[m], oof[m])
    best = max(scores, key=scores.get)
    cols = soft if best == "мягкий" else strict
    log(f"  отбор: мягкий {len(soft)} (CV ROC-AUC {scores['мягкий']:.4f}), строгий {len(strict)} "
        f"(CV ROC-AUC {scores['строгий']:.4f}) -> «{best}»: {', '.join(cols)}")
    return cols


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_agents(obs_mat, fwd, eval_obs, tag, log):
    """PPO на позиционной среде, по агенту на сид; позиции на eval-выборках усредняются."""
    per_seed = {k: [] for k in eval_obs}
    paths = []
    for seed in C.RL_SEEDS:
        t0 = time.perf_counter()
        set_seed(seed)
        env = PositionTradingEnv(obs_mat, fwd, seed=seed, **C.RL_FREQ)
        model = PPO("MlpPolicy", env, seed=seed, verbose=0, **C.RL_KWARGS)
        model.learn(total_timesteps=C.RL_STEPS)
        path = C.MODELS_DIR / f"ppo_{tag}_seed{seed}.zip"
        model.save(path)
        paths.append(path.name)
        for k, m in eval_obs.items():
            per_seed[k].append(rl_rollout(model, m, C.RL_FREQ))
        log(f"  {tag}: сид {seed} обучен за {(time.perf_counter() - t0) / 60:.1f} мин")
    return {k: np.mean(v, axis=0) for k, v in per_seed.items()}, paths


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=str(C.DATA_PATH), help="часовые бары BTC/USDT")
    ap.add_argument("--extra-csv", default=None,
                    help="дополнительные свежие бары (например, выгрузка с Bybit), дописываются в конец")
    args = ap.parse_args()

    C.MODELS_DIR.mkdir(exist_ok=True)
    t_all = time.perf_counter()

    def log(msg):
        print(f"[{(time.perf_counter() - t_all) / 60:5.1f} мин] {msg}", flush=True)

    raw = load_ohlcv_csv(args.csv)
    if args.extra_csv:
        raw = (pd.concat([raw, load_ohlcv_csv(args.extra_csv)])
               .drop_duplicates(subset="date", keep="last").sort_values("date")
               .reset_index(drop=True))
    grid = pd.date_range(raw["date"].min(), raw["date"].max(), freq="h")
    assert len(grid) == len(raw), f"в ряду пропущено {len(grid) - len(raw)} часов"
    log(f"данные: {len(raw)} баров, {raw['date'].min()} — {raw['date'].max()}")

    feat, feature_cols, _, warmup = feature_frame(raw)
    feat = feat.dropna(subset=["target", "fwd_ret"]).reset_index(drop=True)
    feat["target"] = feat["target"].astype(int)
    log(f"признаки: {len(feature_cols)}, прогрев {warmup} баров, наблюдений {len(feat)}")

    n = len(feat)
    train_end, test_end = int(n * C.TRAIN_END), int(n * C.TEST_END)
    train_df = feat.iloc[:train_end - C.EMBARGO].reset_index(drop=True)
    dev_df = feat.iloc[:test_end - C.EMBARGO].reset_index(drop=True)
    val_df = feat.iloc[test_end:].reset_index(drop=True)
    for name, part in (("train", train_df), ("dev", dev_df), ("validation", val_df)):
        log(f"  {name:10s} {len(part):6d} | {part['date'].min()} — {part['date'].max()}")

    tscv = TimeSeriesSplit(n_splits=C.N_CV_SPLITS, gap=C.EMBARGO)
    dev_folds = oof_fold_ids(len(dev_df), tscv)

    log("отбор признаков (раздел 12.2)")
    model_cols = select_features(train_df, dev_df, feature_cols, tscv, log)

    log("базовые модели: OOF на dev + финальное обучение на dev")
    dev_bounds = winsor_bounds(dev_df[model_cols])
    X_dev = apply_winsor(dev_df[model_cols], dev_bounds)
    X_val = apply_winsor(val_df[model_cols], dev_bounds)
    w_dev = sample_weights(dev_df, np.arange(len(dev_df)), True, True)
    models, oof, dev_final, val_proba = {}, {}, {}, {}
    for name in C.BASE_MODELS:
        oof[name], _ = cv_oof(BUILDERS[name], dev_df, model_cols, tscv)
        models[name] = fit_weighted(BUILDERS[name](), X_dev, dev_df["target"], w_dev)
        dev_final[name] = predict_proba1(models[name], X_dev)
        val_proba[name] = predict_proba1(models[name], X_val)
        log(f"  {name}: val ROC-AUC {roc_auc_score(val_df['target'], val_proba[name]):.4f}")

    common = np.all([~np.isnan(oof[m]) for m in C.BASE_MODELS], axis=0)
    ens_oof = np.mean([oof[m] for m in C.BASE_MODELS], axis=0)
    ens_signal, _, ok = tune_signal_layer(ens_oof[common], dev_df["date"].to_numpy()[common],
                                          dev_df["fwd_ret"].to_numpy()[common], dev_folds[common])
    dev_final["Ensemble"] = np.mean([dev_final[m] for m in C.BASE_MODELS], axis=0)
    val_proba["Ensemble"] = np.mean([val_proba[m] for m in C.BASE_MODELS], axis=0)
    log(f"  Ensemble: торговый слой {ens_signal} (устойчивый: {ok})")

    # ── RL-позиция (19.2): наблюдение — признаки по важности CatBoost ────────
    log("RL-позиция (PPO, k=24, δ=0.10, 3 сида) на dev")
    imp = pd.Series(models["CatBoost"].get_feature_importance(), index=model_cols)
    rl_features = list(imp.sort_values(ascending=False).head(C.RL_N_FEATURES).index)
    rl_bounds = winsor_bounds(dev_df[rl_features])
    w = apply_winsor(dev_df[rl_features], rl_bounds)
    rl_mu, rl_sd = w.mean(), w.std().replace(0.0, 1.0)

    def rl_obs(part):
        z = (apply_winsor(part[rl_features], rl_bounds) - rl_mu) / rl_sd
        return np.clip(z.to_numpy(dtype=np.float32), -5.0, 5.0)

    pos_rl, pos_paths = train_agents(rl_obs(dev_df), dev_df["fwd_ret"].to_numpy(),
                                     {"val": rl_obs(val_df)}, "position", log)

    # ── RL на вершине ансамбля (19.3): наблюдение — ранги прогнозов моделей ──
    log("RL на вершине ансамбля (PPO, k=24, δ=0.10, 3 сида) на dev")
    meta_rows = np.flatnonzero(common)
    meta_dev = meta_obs(oof, oof, dev_df, dev_df, C.BASE_MODELS)
    meta_val = meta_obs(val_proba, dev_final, val_df, dev_df, C.BASE_MODELS)
    meta_rl, meta_paths = train_agents(meta_dev[meta_rows], dev_df["fwd_ret"].to_numpy()[meta_rows],
                                       {"val": meta_val}, "ensemble", log)

    # ── проверка на валидации ────────────────────────────────────────────────
    vd, vf, vy = val_df["date"].to_numpy(), val_df["fwd_ret"].to_numpy(), val_df["target"].to_numpy()
    positions = {"BuyAndHold": np.ones(len(val_df))}
    signals = {**C.SIGNAL_PARAMS, "Ensemble": ens_signal}
    for name in (*C.BASE_MODELS, "Ensemble"):
        positions[name] = positions_from_proba(val_proba[name], ref=dev_final[name], **signals[name])
    positions["RL_Position"] = pos_rl["val"]
    positions["RL_Ensemble"] = meta_rl["val"]
    report = {}
    for name, p in positions.items():
        _, m = backtest(vd, vf, p)
        cls = (classification_metrics(vy, (val_proba[name] >= 0.5).astype(int), val_proba[name])
               if name in val_proba else {})
        report[name] = {**{k: round(float(v), 4) for k, v in m.items()},
                        **{k: round(float(v), 4) for k, v in cls.items()}}
    table = pd.DataFrame(report).T[["total_return", "gross_return", "sharpe", "max_drawdown",
                                    "n_trades", "turnover_units", "avg_abs_position"]]
    log(f"валидация {val_df['date'].min()} — {val_df['date'].max()} (комиссия 0.1%):\n"
        + table.round(3).to_string())

    bundle = {
        "created": pd.Timestamp.utcnow().isoformat(),
        "data_end": str(raw["date"].max()),
        "model_cols": model_cols,
        "dev_bounds": dev_bounds,
        "models": models,
        "dev_final_proba": dev_final,           # эталон для рангов мета-агента и порогов слоя
        "signal_params": signals,
        "rl_features": rl_features, "rl_bounds": rl_bounds, "rl_mu": rl_mu, "rl_sd": rl_sd,
        "meta_norm": meta_norm_stats(dev_df),
        "rl_freq": C.RL_FREQ,
        "agents": {"RL_Position": pos_paths, "RL_Ensemble": meta_paths},
        "last_val_positions": {k: float(v[-1]) for k, v in positions.items()},
    }
    joblib.dump(bundle, C.MODELS_DIR / "bundle.joblib")
    (C.MODELS_DIR / "report.json").write_text(json.dumps(
        {"validation_period": [str(val_df["date"].min()), str(val_df["date"].max())],
         "model_cols": model_cols, "ensemble_signal": ens_signal, "metrics": report},
        ensure_ascii=False, indent=2))
    log(f"сохранено в {C.MODELS_DIR}")


if __name__ == "__main__":
    main()
