"""Торговый слой, бэктест и RL-среда из ноутбука final_ensenble.ipynb (разделы 9–11, 19).

Код перенесён из ноутбука; изменены только сигнатуры там, где ноутбук опирался на
глобальные переменные (список моделей мета-агента, сплиттер CV, стартовая позиция
прогона политики — нужна боту, чтобы продолжать с позиции, уже открытой на бирже).
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)

PERIODS_PER_YEAR = 24 * 365
FEE_RATE = 0.001          # та же комиссия, что в ноутбуке
WINSOR_Q = 0.001          # квантиль винзоризации признаков
OUTLIER_WEIGHT = 0.25     # вес аномального бара при обучении
N_BLOCKS = 5              # число подпериодов для проверки устойчивости торгового слоя
MIN_POSITIVE_BLOCKS = 3   # сколько подпериодов должны быть прибыльными
MIN_TURNOVER_PER_YEAR = 6  # минимальный оборот: 3 полных разворота позиции в год
MIN_AVG_EXPOSURE = 0.10    # минимальный средний размер позиции (доля капитала)
TRADE_EPS = 0.01           # изменение позиции меньше 1% капитала сделкой не считается
RL_EPISODE_LEN = 2048      # длина обучающего эпизода RL, баров
NO_FREQ_LIMIT = {"every": 1, "deadband": 0.0}


def backtest(dates, fwd_ret, position, fee_rate=FEE_RATE):
    """Бэктест непрерывной позиции.

    position — доля капитала в позиции от -1 (полный short) до +1 (полный long), принятая
    в момент t по информации, известной к t, и применённая к доходности бара (t -> t+1).
    Комиссия начисляется с оборота |Δposition|: переворот long <-> short стоит 2 * fee_rate.

    Доходность считается и gross (без издержек), и net (с издержками): без этого разделения
    нельзя отличить «у модели нет предсказательной силы» от «edge есть, но его съедают издержки».
    """
    dates = pd.Series(dates).reset_index(drop=True)
    fwd_ret = np.asarray(fwd_ret, dtype=float)
    position = np.asarray(position, dtype=float)
    if len(position) == 0:
        raise ValueError("backtest получил пустую выборку")

    prev = np.roll(position, 1)
    prev[0] = 0.0                                   # до первой сделки капитал вне рынка
    turnover = np.abs(position - prev)

    gross_ret = position * fwd_ret
    fees = fee_rate * turnover
    strat_ret = gross_ret - fees

    equity = (1 + strat_ret).cumprod()
    equity_gross = (1 + gross_ret).cumprod()
    # включаем стартовый капитал (1.0) в running maximum: иначе первый убыток бара 0
    # не считается просадкой от начала, хотя реально капитал уже ниже старта
    drawdown = equity / np.maximum.accumulate(np.r_[1.0, equity])[1:] - 1

    n_periods = len(strat_ret)
    total_return = float(equity[-1] - 1)
    vol = strat_ret.std(ddof=0)
    gross_vol = gross_ret.std(ddof=0)

    curve = pd.DataFrame({"date": dates, "position": position, "strategy_return": strat_ret,
                          "equity": equity, "equity_gross": equity_gross, "drawdown": drawdown})
    metrics = {
        "total_return": total_return,                                   # net, с издержками
        "gross_return": float(equity_gross[-1] - 1),                    # edge модели
        "fee_drag": float(fees.sum()),                                  # сумма ставок издержек (не п.п. доходности)
        "gross_minus_net": float(equity_gross[-1] - equity[-1]),        # реальная потеря доходности, в долях капитала
        "annualized_return": float((1 + total_return) ** (PERIODS_PER_YEAR / n_periods) - 1),
        "sharpe": float(strat_ret.mean() / vol * np.sqrt(PERIODS_PER_YEAR)) if vol > 0 else 0.0,
        "gross_sharpe": float(gross_ret.mean() / gross_vol * np.sqrt(PERIODS_PER_YEAR)) if gross_vol > 0 else 0.0,
        "max_drawdown": float(drawdown.min()),
        "n_trades": int((turnover > TRADE_EPS).sum()),
        "turnover_units": float(turnover.sum()),
        "time_in_market": float((np.abs(position) > TRADE_EPS).mean()),
        "avg_abs_position": float(np.abs(position).mean()),              # фактическая экспозиция
        "avg_position": float(position.mean()),                          # смещение в long/short
        "profitable_bars": float((strat_ret > 0).mean()),
    }
    return curve, metrics


def _ref_quantile(src, qv, fold_ids=None):
    """Квантиль qv эталонного распределения src: одно число или своё значение на каждый фолд.

    fold_ids — номер CV-фолда, в test-часть которого попал бар (-1 — бар не входит в test ни
    одного фолда). Порог для баров фолда k считается только по барам фолдов 0..k-1, то есть по
    OOF-прогнозам, которые на момент фолда k уже в прошлом и лежат внутри его обучающей части.
    Распределение самого фолда k в его пороги не попадает. У фолда 0 прошлого нет: порог NaN,
    сигнала нет, и такие бары в оценку торгового слоя не идут (см. oof_signal_backtest).
    """
    if fold_ids is None:
        return np.quantile(src, qv)
    fold_ids = np.asarray(fold_ids)
    out = np.full(len(src), np.nan)
    for k in np.unique(fold_ids[fold_ids >= 1]):
        past = src[(fold_ids >= 0) & (fold_ids < k)]
        past = past[np.isfinite(past)]
        if len(past):
            out[fold_ids == k] = np.quantile(past, qv)
    return out


def positions_from_proba(proba, mode="hold", q=0.1, scale=10.0, span=1, rebalance=1, ref=None,
                         fold_ids=None):
    """Перевод вероятности роста в позицию.

    mode:
      always    — всегда в рынке ±1 (правило из ДЗ №2);
      flat      — вне рынка в зоне неопределённости;
      hold      — в зоне неопределённости удерживается предыдущая позиция (гистерезис);
      ramp      — размер позиции растёт линейно от порога до экстремума вероятности;
      long_flat — только long или вне рынка (шорт запрещён);
      scaled    — непрерывный размер позиции, пропорциональный уверенности.

    span      — окно экспоненциального сглаживания вероятности (гасит однобаровый шум);
    rebalance — позиция пересматривается раз в k баров;
    q         — доля баров с каждой стороны, попадающая в «уверенную» зону; пороги берутся как
                квантили распределения вероятностей, что устойчиво к калибровке модели.

    Откуда берутся квантильные пороги — ровно один из двух вариантов:
    ref       — эталонное распределение (прогнозы модели на её обучающей выборке); сглаживается
                тем же span, что и proba, чтобы порог и сигнал были в одной шкале;
    fold_ids  — для OOF-вероятностей на dev: порог каждого фолда — по OOF предыдущих фолдов
                (_ref_quantile). Пороги по распределению самого оцениваемого участка (прежний
                вариант ref=None без fold_ids) больше не используются нигде в ноутбуке.
    """
    p = np.asarray(proba, dtype=float)
    if span > 1:
        p = pd.Series(p).ewm(span=span, adjust=False).mean().to_numpy()
    uses_thresholds = mode in ("hold", "flat", "long_flat", "ramp")
    if uses_thresholds and (ref is None) == (fold_ids is None):
        raise ValueError("квантильным режимам нужен ровно один источник порогов: ref или fold_ids")
    if ref is None:
        src = p
    else:
        src = np.asarray(ref, dtype=float)
        src = src[np.isfinite(src)]
        if span > 1:
            src = pd.Series(src).ewm(span=span, adjust=False).mean().to_numpy()

    def thr(qv):
        return _ref_quantile(src, qv, fold_ids)

    if mode == "always":
        pos = np.where(p >= 0.5, 1.0, -1.0)
    elif mode in ("hold", "flat"):
        hi, lo = thr(1 - q), thr(q)
        sig = np.where(p >= hi, 1.0, np.where(p <= lo, -1.0, np.nan))
        pos = pd.Series(sig).ffill().fillna(0.0).to_numpy() if mode == "hold" else np.nan_to_num(sig)
    elif mode == "long_flat":
        pos = (p >= thr(1 - q)).astype(float)
    elif mode == "ramp":
        hi, lo = thr(1 - q), thr(q)
        up = np.clip((p - hi) / np.maximum(thr(0.999) - hi, 1e-6), 0, 1)
        dn = np.clip((lo - p) / np.maximum(lo - thr(0.001), 1e-6), 0, 1)
        pos = np.nan_to_num(up - dn)
    elif mode == "scaled":
        pos = np.clip((p - 0.5) * scale, -1.0, 1.0)
    else:
        raise ValueError(f"неизвестный режим сигнала: {mode}")

    if rebalance > 1:
        keep = (np.arange(len(pos)) % rebalance) == 0
        pos = pd.Series(np.where(keep, pos, np.nan)).ffill().fillna(0.0).to_numpy()
    return pos


def oof_signal_backtest(proba, dates, fwd_ret, fold_ids, params, fee_rate=FEE_RATE):
    """Торговый слой на OOF-вероятностях dev + бэктест.

    Пороги фолда k — квантили OOF фолдов 0..k-1 (то есть данных из его обучающей части), а
    оценка идёт только по барам фолдов 1..K-1: у фолда 0 прошлого нет. Так ни подбор слоя, ни
    Optuna не видят распределение вероятностей того участка, на котором слой оценивается, —
    ровно как на валидации, где пороги зафиксированы по dev (apply_signal, раздел 18).
    """
    fold_ids = np.asarray(fold_ids)
    pos = positions_from_proba(proba, fold_ids=fold_ids, **params)
    ev = fold_ids >= 1
    return backtest(np.asarray(dates)[ev], np.asarray(fwd_ret)[ev], pos[ev], fee_rate)


# сетка торгового слоя: перебирается при подборе гиперпараметров
SIGNAL_GRID = ([{"mode": "always"}] +
               [{"mode": m, "q": q} for m in ("hold", "flat", "ramp", "long_flat")
                for q in (0.02, 0.05, 0.10, 0.20, 0.35)] +
               [{"mode": "scaled", "scale": s} for s in (4, 10, 25)])


SPAN_GRID = (1, 4, 12, 24)


REBALANCE_GRID = (1, 6, 24)


def tune_signal_layer(proba, dates, fwd_ret, fold_ids, fee_rate=FEE_RATE):
    """Перебор параметров торгового слоя по чистому (net) коэффициенту Шарпа.

    Максимизировать один только Sharpe нельзя — у этой задачи есть вырожденный оптимум.
    Стратегия, которая держит крошечную почти постоянную позицию, не платит комиссий и
    показывает прекрасный Sharpe при нулевой полезности: это замаскированный buy & hold с
    плечом 0.07, а не торговая система. Ровно в этот угол и уходит оптимизатор, если его не
    ограничить. Поэтому конфигурация обязана:

    1. давать оборот не меньше MIN_TURNOVER_PER_YEAR единиц капитала в год — то есть
       действительно менять решение, а не «пересиживать»;
    2. держать средний размер позиции не меньше MIN_AVG_EXPOSURE капитала — то есть
       действительно рисковать деньгами;
    3. быть прибыльной хотя бы в MIN_POSITIVE_BLOCKS из N_BLOCKS подпериодов — защита от
       результата, который держится на одной удачной сделке.

    Квантильные пороги каждой конфигурации считаются по прошлым фолдам (fold_ids, см.
    oof_signal_backtest), а не по оцениваемому участку, и все конфигурации оцениваются на одних
    и тех же барах фолдов 1..K-1.

    Возвращает (параметры, таблица, признак того, что допустимая конфигурация найдена).
    """
    rows = []
    for span in SPAN_GRID:
        for reb in REBALANCE_GRID:
            for prm in SIGNAL_GRID:
                curve, m = oof_signal_backtest(proba, dates, fwd_ret, fold_ids,
                                               {**prm, "span": span, "rebalance": reb}, fee_rate)
                blocks = np.array_split(curve["strategy_return"].to_numpy(), N_BLOCKS)
                block_sharpe = [float(b.mean() / b.std(ddof=0) * np.sqrt(PERIODS_PER_YEAR))
                                if b.std(ddof=0) > 0 else 0.0 for b in blocks]
                rows.append({**prm, "span": span, "rebalance": reb, "sharpe": m["sharpe"],
                             "net": m["total_return"], "gross": m["gross_return"],
                             "turnover": m["turnover_units"], "n_trades": m["n_trades"],
                             "time_in_market": m["time_in_market"],
                             "экспозиция": m["avg_abs_position"],
                             "sharpe_min_блок": float(np.min(block_sharpe)),
                             "прибыльных блоков": int(np.sum(np.array(block_sharpe) > 0))})
    tab = pd.DataFrame(rows)
    years = int((np.asarray(fold_ids) >= 1).sum()) / PERIODS_PER_YEAR   # длина оцениваемого участка
    tab["допустима"] = ((tab["turnover"] >= MIN_TURNOVER_PER_YEAR * years) &
                        (tab["экспозиция"] >= MIN_AVG_EXPOSURE))
    tab["устойчива"] = tab["допустима"] & (tab["прибыльных блоков"] >= MIN_POSITIVE_BLOCKS)
    feasible = tab[tab["допустима"]]
    ok_feasible = len(feasible) > 0
    stable = tab[tab["устойчива"]]
    ok_stable = len(stable) > 0
    # Возвращаемый ok — это ok_stable (все три требования докстринга, а не только оборот и
    # экспозиция): раньше ok=True мог означать лишь то, что нашлась хоть одна активная, но
    # неустойчивая по подпериодам конфигурация — фильтр по MIN_POSITIVE_BLOCKS был
    # необязательным в проверках вызывающего кода (codex_report.md, пункт 14). pool всё равно
    # откатывается через feasible/tab, чтобы функция всегда возвращала какую-то конфигурацию.
    pool = stable if ok_stable else (feasible if ok_feasible else tab)
    best = pool.loc[pool["sharpe"].idxmax()]
    params = {"mode": str(best["mode"]), "span": int(best["span"]),
              "rebalance": int(best["rebalance"])}
    for k in ("q", "scale"):                     # присутствуют не у всех режимов
        if k in tab.columns and pd.notna(best[k]):
            params[k] = float(best[k])
    return params, tab, ok_stable


def predict_proba1(model, X):
    """Вероятность класса 1 (рост) — нужна для порогонезависимых ROC-AUC и average precision (AP)."""
    return np.asarray(model.predict_proba(X))[:, 1]


def classification_metrics(y_true, y_pred, y_proba):
    """Шесть стандартных метрик: четыре пороговые и две порогонезависимые."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "average_precision": average_precision_score(y_true, y_proba),
    }


def winsor_bounds(X_train, q=WINSOR_Q):
    """Границы усечения по обучающей выборке."""
    return X_train.quantile(q), X_train.quantile(1 - q)


def apply_winsor(X, bounds):
    lo, hi = bounds
    return X.clip(lower=lo, upper=hi, axis=1)


def sample_weights(data, idx, w_outlier=True, w_ret=False):
    """Веса наблюдений: понижение веса выбросов и/или акцент на крупных движениях."""
    w = np.ones(len(idx), dtype=float)
    if w_outlier:
        w *= np.where(data["anom_any"].to_numpy()[idx] > 0, OUTLIER_WEIGHT, 1.0)
    if w_ret:
        a = data["fwd_ret"].abs().to_numpy()[idx]
        w *= a / a.mean()
    return w


def oof_fold_ids(n, splitter):
    """Номер фолда, в test-часть которого попал каждый бар dev (-1 — ни в какой)."""
    ids = np.full(n, -1, dtype=int)
    for k, (_, te_idx) in enumerate(splitter.split(np.arange(n))):
        ids[te_idx] = k
    return ids


# Позиционная среда: действие — сразу доля капитала от -1 до +1.
class PositionTradingEnv(gym.Env):
    """Среда в терминах доли капитала.

    Наблюдение — признаки бара плюс текущая позиция; действие — новая позиция;
    награда — log(1 + pos_t · r_t − fee · |pos_t − pos_{t-1}|).

    Логарифм, а не сама доходность, потому что сумма логарифмов побарных доходностей — это
    логарифм отношения конечного капитала к начальному (не сам капитал); SB3 дополнительно
    дисконтирует награду (gamma=0.99), так что оптимизируется величина, связанная с итогом
    бэктеста, а не буквально тождественная ему сумма. Комиссия входит в награду, поэтому
    каждая сделка сразу стоит денег. Но на практике этого мало: градиент политики оценивается
    по шуму часовых доходностей, и агент не успевает выучить, что частая перекладка убыточна.
    Поэтому частота сделок ограничена явно:
      every    — решение раз в every баров; один шаг агента проводит среду через every баров
                 с неизменной позицией, награды суммируются (комиссия — только на первом);
      deadband — изменение позиции меньше deadband капитала не исполняется.
    Ограничения MIN_TURNOVER_PER_YEAR / MIN_AVG_EXPOSURE раздела 9 (защита перебора от
    вырожденного «не торговать») здесь по-прежнему не нужны: они ограничивают снизу, а не сверху.

    Эпизод — случайный отрезок длины episode_len. Один проход от первого бара к последнему дал
    бы агенту одно начальное условие и один порядок режимов; случайный старт даёт разнообразие.
    """

    metadata = {"render_modes": []}

    def __init__(self, obs, fwd_ret, fee_rate=FEE_RATE, episode_len=RL_EPISODE_LEN,
                 random_start=True, seed=None, every=1, deadband=0.0):
        super().__init__()
        self.obs_mat = np.ascontiguousarray(obs, dtype=np.float32)
        self.fwd = np.asarray(fwd_ret, dtype=np.float64)
        self.n = len(self.fwd)
        self.fee = float(fee_rate)
        self.episode_len = int(min(episode_len or self.n, self.n))
        self.random_start = bool(random_start)
        self.every, self.deadband = max(int(every), 1), float(deadband)
        self.observation_space = spaces.Box(-np.inf, np.inf,
                                            shape=(self.obs_mat.shape[1] + 1,),
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self.t = self.t_end = 0
        self.position = 0.0

    def _obs(self):
        t = min(self.t, self.n - 1)
        return np.concatenate([self.obs_mat[t],
                               np.float32([self.position])]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        start = (int(self._rng.integers(0, self.n - self.episode_len + 1))
                 if self.random_start and self.episode_len < self.n else 0)
        self.t, self.t_end, self.position = start, start + self.episode_len, 0.0
        return self._obs(), {}

    def step(self, action):
        new_pos = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
        if abs(new_pos - self.position) < self.deadband:
            new_pos = self.position                     # мелкая перекладка не исполняется
        reward = 0.0
        for _ in range(self.every):
            turnover = abs(new_pos - self.position)     # ненулевой только на первом баре
            net = new_pos * float(self.fwd[self.t]) - self.fee * turnover
            # log1p(max(net, -0.999)) не даёт банкротству (net <= -1) обрушить награду в -inf, но
            # тем самым и скрывает его: на текущем часовом ряде это не подтверждённая причина
            # результатов (codex_report.md, пункт 12), явно завершать эпизод при net <= -1 здесь не
            # реализовано, чтобы не менять динамику награды без отдельной проверки эффекта.
            reward += float(np.log1p(max(net, -0.999)))
            self.position = new_pos
            self.t += 1
            if self.t >= self.n or self.t >= self.t_end:
                break
        # Конец РЕАЛЬНО доступных данных (terminated — следующего наблюдения не существует,
        # бутстрап недопустим) отличается от искусственного обрыва эпизода внутри доступного
        # ряда (truncated — данные есть, SB3 корректно бутстрапирует по настоящему self._obs()).
        # Раньше оба случая были truncated с одной и той же заглушкой через min(t, n-1), из-за
        # чего SB3 мог бутстрапировать по несуществующему «продолжению» после конца ряда
        # (codex_report.md, пункт 12).
        terminated = self.t >= self.n
        truncated = (not terminated) and (self.t >= self.t_end)
        return self._obs(), reward, terminated, truncated, {}


def rl_rollout(model, obs_mat, freq=NO_FREQ_LIMIT, p0=0.0):
    """Детерминированный прогон политики по всей выборке: позиция на каждом баре.

    Те же ограничения частоты, что в среде: решение раз в every баров, перекладка меньше
    deadband не исполняется. Позиция — часть наблюдения, поэтому прогон последовательный.
    """
    every, deadband = max(int(freq["every"]), 1), float(freq["deadband"])
    n = len(obs_mat)
    pos = np.empty(n, dtype=float)
    p = float(p0)
    for t in range(n):
        if t % every == 0:
            obs = np.concatenate([obs_mat[t], np.float32([p])]).astype(np.float32)
            action, _ = model.predict(obs, deterministic=True)
            target = float(np.clip(np.asarray(action).ravel()[0], -1.0, 1.0))
            if abs(target - p) >= deadband:
                p = target
        pos[t] = p
    return pos


def rank_to_ref(p, ref):
    """Квантиль значений p в эталонном распределении ref, приведённый к [-1, 1]."""
    p = np.asarray(p, dtype=float)
    ref = np.asarray(ref, dtype=float)
    ref = np.sort(ref[np.isfinite(ref)])
    q = np.searchsorted(ref, p, side="right") / max(len(ref), 1)
    return np.where(np.isfinite(p), 2.0 * q - 1.0, np.nan)


def meta_norm_stats(norm_df):
    """Статистики нормировки рыночной части наблюдения мета-агента (сохраняются для бота)."""
    return {"ret_1_std": float(norm_df["ret_1"].std()),
            "atr_14_mean": float(norm_df["atr_14"].mean()),
            "atr_14_std": float(norm_df["atr_14"].std())}


def meta_obs(proba_by_model, ref_by_model, part, norm_df, models):
    """Наблюдение мета-агента: ранги прогнозов + согласие моделей + состояние рынка.

    norm_df — DataFrame или уже посчитанный словарь meta_norm_stats (так его передаёт бот).
    market нормируется по статистикам norm_df — для честной train -> test проверки это
    train_df (не весь dev: иначе test-часть dev участвовала бы в нормировке наблюдения), для
    финала — dev_df (валидация не участвует в нормировке ни в одном из двух вариантов).
    """
    ranks = np.column_stack([rank_to_ref(proba_by_model[m], ref_by_model[m])
                             for m in models])
    ranks = np.nan_to_num(ranks, nan=0.0)
    consensus = ranks.mean(axis=1, keepdims=True)        # средний ранг — мнение ансамбля
    spread = ranks.std(axis=1, keepdims=True)            # разброс — насколько модели согласны
    agree = (np.sign(ranks) == np.sign(consensus)).mean(axis=1, keepdims=True) * 2 - 1
    norm = meta_norm_stats(norm_df) if isinstance(norm_df, pd.DataFrame) else norm_df
    market = np.column_stack([
        np.clip(part["ret_1"].to_numpy() / max(norm["ret_1_std"], 1e-9), -5, 5),
        np.clip((part["atr_14"].to_numpy() - norm["atr_14_mean"])
                / max(norm["atr_14_std"], 1e-9), -5, 5),
    ])
    return np.nan_to_num(np.hstack([ranks, consensus, spread, agree, market]),
                         nan=0.0).astype(np.float32)
