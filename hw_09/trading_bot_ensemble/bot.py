"""Торговый бот BTC/USDT perpetual на Bybit: «Ансамбль + SL 3% + безубыток 5%».

Каждый час, после закрытия часовой свечи:
  1. скачивает ~2000 последних закрытых часовых свечей BTCUSDT (linear) с Bybit;
  2. считает 18 признаков и вероятность роста трёх моделей (LogReg, DecisionTree, CatBoost);
     ансамбль — их среднее;
  3. торговый слой ансамбля (раздел 16 ноутбука): EWM(span=12), режим always — LONG при
     сглаженной вероятности >= 0.5, иначе SHORT; пересмотр позиции раз в сутки (свеча,
     закрывающаяся в 00:00 UTC);
  4. пропускает рекомендацию через риск-менеджер (risk.py);
  5. приводит позицию на бирже к целевой рыночным ордером со стоп-лоссом 3% от цены входа;
  6. если лучшая цена закрытых часов с момента входа ушла в плюс на 5%, переносит стоп на бирже
     в безубыток: вход ± 0.2% (раздел 26 ноутбука, со следующего часа после достижения порога);
  7. после выхода по стопу или безубытку — вне рынка, пока знак целевой позиции не сменится.
Между часовыми итерациями каждые RISK_CHECK_SEC секунд проверяются equity, стоп и рубильник.

Запуск:
  python bot.py                  — бесконечный цикл
  python bot.py --once           — одна итерация и выход
  python bot.py --dry-run        — считать сигналы, но не отправлять ордера
  python bot.py --test-trade     — проверка исполнения: открыть минимальную позицию по сигналу
                                    ансамбля со стопом 3%, перенести стоп в безубыток, закрыть
  python bot.py --status         — состояние бота, позиция и лимиты риска
  python bot.py --halt [причина] — рубильник: работающий бот закроет позицию и остановит торговлю
  python bot.py --resume         — снять остановку (удалить файл HALT)
"""
import argparse
import csv
import json
import logging
import math
import os
import pickle
import signal
import sys
import time
from collections import Counter
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from bybit_client import MAINNET, TESTNET, BybitClient
from features import MODEL_COLS, build_features, finalize
from market_cache import KlineCache
from risk import RiskConfig, RiskManager
from strategy import BE_LOCK, EXIT_PARAMS, target_positions

HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(HERE, "artifacts")
RUNTIME_DIR = os.environ.get("BOT_RUNTIME_DIR") or os.path.join(HERE, "runtime")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
STATE_PATH = os.path.join(RUNTIME_DIR, "state.json")
HALT_PATH = os.path.join(RUNTIME_DIR, "HALT")
HEARTBEAT_PATH = os.path.join(RUNTIME_DIR, "heartbeat")
HISTORY_BARS = 2000        # свечей на каждую итерацию: прогрев признаков + история сигнала
VALID_FROM = 600           # первые бары истории — прогрев окон (256 у вейвлетов, 55 у Дончиана)
REBALANCE_HOUR_UTC = 0     # позиция пересматривается на свече, закрывающейся в 00:00 UTC

log = logging.getLogger("bot")


# ── конфигурация ──────────────────────────────────────────────────────────────────────────
def load_env(path=os.path.join(HERE, ".env")):
    """Переменные из .env; уже заданные в окружении (docker compose env_file) не перезаписываются."""
    if not os.path.exists(path):
        return
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def config():
    load_env()
    risk = RiskConfig.from_env()
    return {
        "api_key": os.environ["BYBIT_API_KEY"],
        "api_secret": os.environ["BYBIT_API_SECRET"],
        "base_url": TESTNET if os.environ.get("BYBIT_TESTNET", "0") == "1" else MAINNET,
        "proxy": os.environ.get("BYBIT_PROXY") or None,
        "symbol": os.environ.get("SYMBOL", "BTCUSDT"),
        "leverage": min(float(os.environ.get("LEVERAGE", "2")), risk.max_leverage),
        "position_fraction": float(os.environ.get("POSITION_FRACTION", "0.95")),
        "risk": risk,
    }


def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S")
    fmt.converter = time.gmtime
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in (logging.StreamHandler(sys.stdout),
              RotatingFileHandler(os.path.join(LOG_DIR, "bot.log"), maxBytes=5_000_000, backupCount=5)):
        h.setFormatter(fmt)
        root.addHandler(h)


def append_csv(name, row):
    """Дописать строку в CSV-журнал. Если набор колонок изменился (новая версия бота), файл
    переписывается с объединённым заголовком: старые строки получают пустые новые колонки."""
    path = os.path.join(LOG_DIR, name)
    fields = list(row)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        with open(path, newline="") as fh:
            rows = list(csv.reader(fh))
        header = rows[0]
        if header != fields:
            merged = fields + [c for c in header if c not in fields]
            records = []
            for r in rows[1:]:
                cols = header if len(r) == len(header) else fields if len(r) == len(fields) else None
                if cols is None:
                    log.warning("%s: строка с %d полями не распознана и пропущена: %s", name, len(r), r[:3])
                    continue
                records.append(dict(zip(cols, r)))
            tmp = path + ".tmp"
            with open(tmp, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=merged)
                w.writeheader()
                w.writerows(records)
            os.replace(tmp, path)
            log.info("%s: формат журнала обновлён (%d -> %d колонок), %d строк перенесено",
                     name, len(header), len(merged), len(records))
            fields = merged
    new = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        if new:
            w.writeheader()
        w.writerow(row)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def current_hour():
    return pd.Timestamp.now(tz="UTC").tz_convert(None).floor("h")


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


SIDE_NAME = {1: "LONG", -1: "SHORT", 0: "FLAT"}


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as fh:
            return json.load(fh)
    return {"side": 0, "entry": 0.0, "entry_bar": None, "be_on": False, "blocked": 0,
            "last_bar": None, "risk": {}}


class Bot:
    def __init__(self, cfg, dry_run=False):
        self.cfg, self.dry_run = cfg, dry_run
        self.symbol = cfg["symbol"]
        self.client = BybitClient(cfg["api_key"], cfg["api_secret"], cfg["base_url"], cfg["proxy"])
        with open(os.path.join(ART_DIR, "meta.json")) as fh:
            self.meta = json.load(fh)
        assert self.meta["model_cols"] == MODEL_COLS, "artifacts/ обучены на другом наборе признаков"
        cb = CatBoostClassifier()
        cb.load_model(os.path.join(ART_DIR, "catboost.cbm"))
        with open(os.path.join(ART_DIR, "sklearn_models.pkl"), "rb") as fh:
            sk = pickle.load(fh)
        self.models = {"LogisticRegression": sk["LogisticRegression"],
                       "DecisionTree": sk["DecisionTree"], "CatBoost": cb}
        self.w_lo = pd.Series(self.meta["winsor_lo"])[MODEL_COLS]
        self.w_hi = pd.Series(self.meta["winsor_hi"])[MODEL_COLS]
        self.sl, self.be = EXIT_PARAMS["sl"], EXIT_PARAMS["be"]

        self.kcache = KlineCache(self.client, self.symbol,
                                 os.path.join(RUNTIME_DIR, f"klines_{self.symbol}_1h.csv"),
                                 keep=HISTORY_BARS + 200)
        inst = self.client.instrument(self.symbol)
        self.qty_step = float(inst["lotSizeFilter"]["qtyStep"])
        self.min_qty = float(inst["lotSizeFilter"]["minOrderQty"])
        self.tick = float(inst["priceFilter"]["tickSize"])
        self.state = load_state()
        for k, v in (("entry_bar", None), ("be_on", False), ("risk", {})):
            self.state.setdefault(k, v)
        self.risk = RiskManager(cfg["risk"], self.state["risk"], HALT_PATH, write_files=not dry_run)
        log.info("Bybit %s, %s: шаг лота %s, мин. лот %s, тик %s; ансамбль %s; SL %.1f%%, "
                 "безубыток при +%.1f%% (стоп -> вход ± %.1f%%)%s", cfg["base_url"], self.symbol,
                 self.qty_step, self.min_qty, self.tick, "+".join(self.models), self.sl * 100,
                 self.be * 100, BE_LOCK * 100, " [DRY-RUN]" if dry_run else "")
        log.info("Лимиты риска: %s; плечо %s", cfg["risk"], cfg["leverage"])
        if not dry_run:
            if self.client.ensure_one_way_mode(self.symbol):
                log.warning("Аккаунт был в режиме хеджирования — %s переключён в one-way режим", self.symbol)
            self.client.set_leverage(self.symbol, cfg["leverage"])

    def save_state(self):
        if self.dry_run:
            return
        tmp = STATE_PATH + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.state, fh, indent=2)
        os.replace(tmp, STATE_PATH)

    def heartbeat(self):
        if self.dry_run:
            return
        with open(HEARTBEAT_PATH, "w") as fh:
            fh.write(now_iso())

    # ── данные и рекомендация ─────────────────────────────────────────────────────────────
    def fetch_bars(self, n=HISTORY_BARS):
        """Последние n закрытых часовых свечей: из кеша + догрузка новых (market_cache.py)."""
        df = self.kcache.get(n)
        gaps = int((df["date"].diff().dropna() != pd.Timedelta(hours=1)).sum())
        if gaps:
            raise RuntimeError(f"в истории свечей {gaps} разрывов — признаки были бы смещены")
        if len(df) < VALID_FROM + 100:
            raise RuntimeError(f"биржа отдала только {len(df)} свечей — мало для прогрева признаков")
        c = self.kcache.last
        log.info("Свечи: %d из кеша + %d загружено, запросов к /v5/market/kline: %d "
                 "(без кеша было бы %d)", c["from_cache"], c["downloaded"], c["requests"],
                 n // 1000 + 1)
        return df

    def compute_signal(self):
        bars = self.fetch_bars()
        feat = finalize(build_features(bars), warmup=0)           # news_count_168 = 0, как в walk-forward
        feat = feat.iloc[VALID_FROM:].reset_index(drop=True)
        X = feat[MODEL_COLS].clip(lower=self.w_lo, upper=self.w_hi, axis=1)
        if not X.notna().all().all():
            raise RuntimeError("в признаках есть пропуски — рекомендация не формируется")
        per_model = {k: np.asarray(m.predict_proba(X))[:, 1] for k, m in self.models.items()}
        proba = np.mean(list(per_model.values()), axis=0)
        rebalance = ((feat["date"] + pd.Timedelta(hours=1)).dt.hour == REBALANCE_HOUR_UTC).to_numpy()
        target, sig, p_s = target_positions(proba, rebalance)
        last_reb = feat["date"][rebalance].iloc[-1] if rebalance.any() else None
        return {"bar": feat["date"].iloc[-1], "close": float(feat["close"].iloc[-1]),
                "proba": float(proba[-1]), "p_smooth": float(p_s[-1]), "signal": int(sig[-1]),
                "target": int(target[-1]), "last_rebalance_bar": last_reb,
                "per_model": {k: float(v[-1]) for k, v in per_model.items()}, "bars": bars}

    # ── исполнение ────────────────────────────────────────────────────────────────────────
    def round_qty(self, q):
        return math.floor(q / self.qty_step + 1e-9) * self.qty_step

    def fmt_qty(self, q):
        decimals = max(0, -int(math.floor(math.log10(self.qty_step))))
        return f"{q:.{decimals}f}"

    def fmt_price(self, p, side):
        """Цена стопа по тику: у лонга вниз, у шорта вверх (стоп не ближе расчётного)."""
        k = math.floor(p / self.tick + 1e-6) if side > 0 else math.ceil(p / self.tick - 1e-6)
        decimals = max(0, -int(math.floor(math.log10(self.tick))))
        return f"{k * self.tick:.{decimals}f}"

    def stop_price(self, entry, side):
        return self.fmt_price(entry * (1 - self.sl) if side > 0 else entry * (1 + self.sl), side)

    def be_price(self, entry, side):
        """Стоп в безубытке: вход + две комиссии (лонг) / вход − две комиссии (шорт)."""
        k = math.ceil(entry * (1 + BE_LOCK) / self.tick - 1e-6) if side > 0 else \
            math.floor(entry * (1 - BE_LOCK) / self.tick + 1e-6)
        decimals = max(0, -int(math.floor(math.log10(self.tick))))
        return f"{k * self.tick:.{decimals}f}"

    def wanted_stop(self, entry, side):
        return self.be_price(entry, side) if self.state.get("be_on") else self.stop_price(entry, side)

    def wait_position(self, want_side, timeout=15):
        for _ in range(timeout):
            size, avg, sl = self.client.position(self.symbol)
            if sign(size) == want_side:
                return size, avg, sl
            time.sleep(1)
        return self.client.position(self.symbol)

    def reset_trade(self):
        self.state.update(side=0, entry=0.0, entry_bar=None, be_on=False)

    def close_position(self, size, reason, count_pnl=True):
        side = "Sell" if size > 0 else "Buy"
        qty = self.fmt_qty(abs(size))
        log.info("Закрытие %s %s BTC рыночным ордером (%s)", SIDE_NAME[sign(size)], qty, reason)
        if self.dry_run:
            return None
        res = self.client.market_order(self.symbol, side, qty, reduce_only=True)
        self.wait_position(0)
        pnl = self.client.closed_pnl(self.symbol, limit=1)
        realized = float(pnl[0]["closedPnl"]) if pnl else float("nan")
        exit_px = float(pnl[0]["avgExitPrice"]) if pnl else float("nan")
        append_csv("trades.csv", {"time": now_iso(), "action": "close", "side": SIDE_NAME[sign(size)],
                                  "qty": qty, "price": exit_px, "stop_loss": "",
                                  "order_id": res.get("orderId"), "realized_pnl": realized,
                                  "reason": reason})
        log.info("Позиция закрыта по %.1f, реализованный PnL %.4f USDT", exit_px, realized)
        if count_pnl:
            self.risk.on_close(realized)
        self.reset_trade()
        return realized

    def open_position(self, side, reason, qty=None):
        price = self.client.last_price(self.symbol)
        equity, _ = self.client.equity_usdt()
        if qty is None:
            notional = self.risk.cap_notional(equity * self.cfg["position_fraction"])
            qty = self.round_qty(notional / price)
        if qty < self.min_qty:
            log.warning("Позиция %s не открыта: лимиты дают %.2f USDT (equity %.2f), а минимальный "
                        "лот %s BTC стоит %.2f USDT", SIDE_NAME[side], qty * price, equity,
                        self.min_qty, self.min_qty * price)
            return 0.0, 0.0
        sl_guess = self.stop_price(price, side)
        log.info("Открытие %s %s BTC (%.2f USDT, equity %.2f, цена ~%.1f), стоп-лосс ~%s (%s)",
                 SIDE_NAME[side], self.fmt_qty(qty), qty * price, equity, price, sl_guess, reason)
        if self.dry_run:
            return qty, price
        res = self.client.market_order(self.symbol, "Buy" if side > 0 else "Sell", self.fmt_qty(qty),
                                       stop_loss=sl_guess)
        size, avg, sl_set = self.wait_position(side)
        if sign(size) != side:
            raise RuntimeError(f"ордер {res.get('orderId')} отправлен, но позиции {SIDE_NAME[side]} нет")
        # первая свеча сделки — та, что открылась в момент входа: с неё считается лучшая цена
        self.state.update(side=side, entry=avg, entry_bar=str(current_hour()), be_on=False)
        sl_final = self.stop_price(avg, side)
        try:
            if abs(float(sl_final) - sl_set) >= self.tick / 2:
                self.client.set_stop_loss(self.symbol, sl_final)
            _, _, sl_set = self.client.position(self.symbol)
        except Exception as exc:
            log.error("Не удалось выставить стоп-лосс: %s", exc)
            sl_set = sl_set or 0.0
        append_csv("trades.csv", {"time": now_iso(), "action": "open", "side": SIDE_NAME[side],
                                  "qty": self.fmt_qty(abs(size)), "price": avg, "stop_loss": sl_set,
                                  "order_id": res.get("orderId"), "realized_pnl": "", "reason": reason})
        self.risk.on_open()
        if not sl_set:
            self.close_position(size, "стоп-лосс не выставлен — аварийное закрытие")
            self.risk.halt("не удалось поставить стоп-лосс на позицию")
            return 0.0, 0.0
        log.info("Позиция открыта: %s %s BTC по %.1f, стоп-лосс на бирже %.1f; безубыток включится "
                 "при %.1f", SIDE_NAME[side], self.fmt_qty(abs(size)), avg, sl_set,
                 avg * (1 + self.be) if side > 0 else avg * (1 - self.be))
        return abs(size), avg

    def update_breakeven(self, bars, size):
        """Перенос стопа в безубыток, когда лучшая цена закрытых часов сделки ушла на +be."""
        st = self.state
        if st["side"] == 0 or st.get("be_on") or not st.get("entry_bar") or not st.get("entry"):
            return
        held = bars[bars["date"] >= pd.Timestamp(st["entry_bar"])]
        if held.empty:
            return
        side, entry = st["side"], st["entry"]
        best = held["high"].max() if side > 0 else held["low"].min()
        trigger = entry * (1 + self.be) if side > 0 else entry * (1 - self.be)
        if (side > 0 and best >= trigger) or (side < 0 and best <= trigger):
            be_px = self.be_price(entry, side)
            log.info("Лучшая цена сделки %.1f достигла порога безубытка %.1f — стоп переносится "
                     "на %s (вход %.1f ± %.1f%%)", best, trigger, be_px, entry, BE_LOCK * 100)
            if self.dry_run:
                return
            self.client.set_stop_loss(self.symbol, be_px)
            st["be_on"] = True
            append_csv("trades.csv", {"time": now_iso(), "action": "breakeven", "side": SIDE_NAME[side],
                                      "qty": self.fmt_qty(abs(size)), "price": best, "stop_loss": be_px,
                                      "order_id": "", "realized_pnl": "",
                                      "reason": f"цена +{self.be:.0%} от входа: стоп в безубыток"})

    def detect_external_close(self, cur):
        """Позиция исчезла без участия бота — сработал стоп-лосс или стоп в безубытке."""
        st = self.state
        if st["side"] == 0 or cur != 0:
            return
        kind = "безубыток" if st.get("be_on") else "стоп-лосс"
        st["blocked"] = st["side"]
        pnl = self.client.closed_pnl(self.symbol, limit=1)
        realized = float(pnl[0]["closedPnl"]) if pnl else float("nan")
        log.warning("Позицию %s закрыл %s (PnL %s USDT): вне рынка до смены сигнала",
                    SIDE_NAME[st["side"]], kind, pnl[0]["closedPnl"] if pnl else "?")
        append_csv("trades.csv", {"time": now_iso(), "action": "breakeven_exit" if st.get("be_on") else "stop_loss",
                                  "side": SIDE_NAME[st["side"]], "qty": "",
                                  "price": pnl[0]["avgExitPrice"] if pnl else "", "stop_loss": "",
                                  "order_id": pnl[0].get("orderId") if pnl else "",
                                  "realized_pnl": realized, "reason": f"{kind} на бирже"})
        self.risk.on_close(realized)
        self.reset_trade()

    def enforce_risk(self, size):
        equity, _ = self.client.equity_usdt()
        status = self.risk.check_equity(equity)
        if status and size != 0:
            self.close_position(size, f"риск-менеджер: {self.state['risk'].get('halted') or 'суточная пауза'}",
                                count_pnl=False)
            self.state["blocked"] = 0
        return status, equity

    def repair_stop(self, size, avg):
        cur = sign(size)
        sl_px = self.wanted_stop(self.state.get("entry") or avg, cur)
        log.warning("На позиции нет стоп-лосса — ставлю %s", sl_px)
        try:
            self.client.set_stop_loss(self.symbol, sl_px)
        except Exception as exc:
            log.error("Стоп не восстановлен (%s) — аварийное закрытие позиции", exc)
            self.close_position(size, "стоп-лосс не восстановлен", count_pnl=False)
            self.risk.halt("не удалось восстановить стоп-лосс на позиции")

    # ── одна итерация ─────────────────────────────────────────────────────────────────────
    def step(self):
        st = self.state
        size, avg, sl_on_exchange = self.client.position(self.symbol)
        cur = sign(size)
        self.detect_external_close(cur)
        risk_status, equity = self.enforce_risk(size)
        risk_closed = bool(risk_status and cur != 0)
        if risk_status:
            size, cur = 0.0, 0

        s = self.compute_signal()
        target = s["target"]
        if st["blocked"] and target != st["blocked"]:
            log.info("Знак цели сменился (%s -> %s): блокировка после выхода снята",
                     SIDE_NAME[st["blocked"]], SIDE_NAME[target])
            st["blocked"] = 0
        recommended = 0 if st["blocked"] else target
        desired = 0 if risk_status else recommended

        pm = " ".join(f"{k[:3]}={v:.3f}" for k, v in s["per_model"].items())
        log.info("Свеча %s close=%.1f | %s -> p=%.4f EWM=%.4f | сигнал %s | цель %s (пересмотр %s) | "
                 "блок %s | риск %s | на бирже %s %s%s",
                 s["bar"], s["close"], pm, s["proba"], s["p_smooth"], SIDE_NAME[s["signal"]],
                 SIDE_NAME[target], s["last_rebalance_bar"], SIDE_NAME[st["blocked"]],
                 risk_status or "OK", SIDE_NAME[cur], abs(size),
                 " (стоп в безубытке)" if st.get("be_on") and cur else "")

        action, veto = ("risk_close" if risk_closed else "hold"), ""
        if desired != cur:
            ok, veto = self.risk.data_ok(s["bar"], s["close"], self.client.last_price(self.symbol))
            if ok and desired != 0:
                ok, veto = self.risk.can_open()
            if not ok:
                log.warning("Риск-менеджер запретил действие %s -> %s: %s",
                            SIDE_NAME[cur], SIDE_NAME[desired], veto)
                action = "vetoed"
            else:
                reason = f"цель {SIDE_NAME[desired]} (ансамбль p_ewm={s['p_smooth']:.4f})"
                if cur != 0:
                    self.close_position(size, reason)
                    action = "close"
                if desired != 0 and not self.risk.st["halted"]:
                    qty, _ = self.open_position(desired, reason)
                    if qty:
                        action = "open" if cur == 0 else "reverse"
        elif cur != 0:
            if st["side"] != cur:                       # позиция есть, а бот о ней не знал
                st.update(side=cur, entry=avg, entry_bar=st.get("entry_bar") or str(s["bar"]))
            st["entry"] = st.get("entry") or avg
            if sl_on_exchange == 0 and not self.dry_run:
                self.repair_stop(size, avg)
            self.update_breakeven(s["bars"], size)

        st["last_bar"] = str(s["bar"])
        self.save_state()
        equity, _ = self.client.equity_usdt()
        append_csv("signals.csv", {"time": now_iso(), "bar": str(s["bar"]), "close": s["close"],
                                   **{f"p_{k}": round(v, 6) for k, v in s["per_model"].items()},
                                   "proba": round(s["proba"], 6), "p_ewm": round(s["p_smooth"], 6),
                                   "signal": s["signal"], "target": target, "blocked": st["blocked"],
                                   "recommended": recommended, "desired": desired,
                                   "position_before": cur, "action": action, "veto": veto,
                                   "be_on": int(bool(st.get("be_on"))), "equity": round(equity, 4),
                                   **self.risk.metrics(equity), "dry_run": int(self.dry_run)})
        self.heartbeat()
        return s, action

    def risk_check(self):
        size, avg, sl_on_exchange = self.client.position(self.symbol)
        cur = sign(size)
        self.detect_external_close(cur)
        status, _ = self.enforce_risk(size)
        if not status and cur != 0 and sl_on_exchange == 0 and not self.dry_run:
            self.repair_stop(size, avg)
        self.save_state()
        self.heartbeat()

    # ── проверка исполнения ───────────────────────────────────────────────────────────────
    def test_trade(self):
        """Минимальная сделка по сигналу ансамбля: открыть со стопом 3%, перенести стоп в
        безубыток (как при срабатывании правила), проверить оба уровня на бирже, закрыть."""
        size, _, _ = self.client.position(self.symbol)
        if size != 0:
            raise RuntimeError("на бирже уже есть позиция — тестовая сделка не выполняется")
        s = self.compute_signal()
        side = s["signal"]
        log.info("ТЕСТ: свеча %s — модели %s, ансамбль p=%.4f, EWM=%.4f, сигнал %s, цель %s -> "
                 "тестовая сторона %s", s["bar"], s["per_model"], s["proba"], s["p_smooth"],
                 SIDE_NAME[s["signal"]], SIDE_NAME[s["target"]], SIDE_NAME[side])
        saved = dict(self.state)
        qty, _ = self.open_position(side, "тестовая сделка", qty=self.min_qty)
        if not qty:
            return False
        size, avg, sl = self.client.position(self.symbol)
        exp_sl = float(self.stop_price(avg, side))
        ok_sl = sign(size) == side and abs(sl - exp_sl) < self.tick * 1.5
        log.info("ТЕСТ: на бирже %s %s BTC, вход %.1f, стоп-лосс %.1f (ожидался %.1f, %.2f%%) — %s",
                 SIDE_NAME[sign(size)], abs(size), avg, sl, exp_sl, abs(sl / avg - 1) * 100,
                 "OK" if ok_sl else "НЕ СОВПАДАЕТ")
        be_px = self.be_price(avg, side)
        price = self.client.last_price(self.symbol)
        # безубыток выше текущей цены у лонга (ниже у шорта) биржа не примет, пока цена не ушла в плюс
        if (side > 0 and float(be_px) < price) or (side < 0 and float(be_px) > price):
            self.client.set_stop_loss(self.symbol, be_px)
            _, _, sl2 = self.client.position(self.symbol)
            ok_be = abs(sl2 - float(be_px)) < self.tick * 1.5
            log.info("ТЕСТ: стоп перенесён в безубыток %.1f (ожидался %s) — %s", sl2, be_px,
                     "OK" if ok_be else "НЕ СОВПАДАЕТ")
        else:
            # проверяем сам механизм переноса стопа: подтягиваем стоп на 1% от входа
            mid = self.fmt_price(avg * (1 - 0.01) if side > 0 else avg * (1 + 0.01), side)
            self.client.set_stop_loss(self.symbol, mid)
            _, _, sl2 = self.client.position(self.symbol)
            ok_be = abs(sl2 - float(mid)) < self.tick * 1.5
            log.info("ТЕСТ: цена %.1f ещё не прошла уровень безубытка %s, поэтому перенос стопа "
                     "проверен на уровне %s: на бирже %.1f — %s", price, be_px, mid, sl2,
                     "OK" if ok_be else "НЕ СОВПАДАЕТ")
        time.sleep(3)
        self.close_position(size, "тестовая сделка: закрытие", count_pnl=False)
        size_after, _, _ = self.client.position(self.symbol)
        log.info("ТЕСТ: позиция после закрытия %s — %s", size_after, "OK" if size_after == 0 else "ОШИБКА")
        self.state.clear()
        self.state.update(saved)
        self.risk.st["trades_today"] = max(0, self.risk.st["trades_today"] - 1)   # тест — не сделка стратегии
        self.save_state()
        return ok_sl and ok_be and size_after == 0


# ── команды управления ────────────────────────────────────────────────────────────────────
def cmd_halt(reason):
    os.makedirs(RUNTIME_DIR, exist_ok=True)
    with open(HALT_PATH, "w") as fh:
        fh.write(f"{now_iso()} {reason}\n")
    print(f"Создан {HALT_PATH}. Работающий бот закроет позицию и остановит торговлю "
          f"в течение RISK_CHECK_SEC секунд.")


def cmd_resume():
    if os.path.exists(HALT_PATH):
        os.remove(HALT_PATH)
        print(f"Удалён {HALT_PATH}. Бот возобновит торговлю при ближайшей проверке.")
    else:
        print("Файла HALT нет — торговля не остановлена.")


def cmd_status(cfg):
    st = load_state()
    client = BybitClient(cfg["api_key"], cfg["api_secret"], cfg["base_url"], cfg["proxy"])
    size, avg, sl = client.position(cfg["symbol"])
    equity, _ = client.equity_usdt()
    hb = open(HEARTBEAT_PATH).read() if os.path.exists(HEARTBEAT_PATH) else "нет"
    rm = RiskManager(cfg["risk"], st.setdefault("risk", {}), HALT_PATH, write_files=False)
    print(json.dumps({"position_btc": size, "entry": avg, "stop_loss": sl, "equity_usdt": equity,
                      "state": st, "risk_now": rm.metrics(equity),
                      "halt_file": open(HALT_PATH).read().strip() if os.path.exists(HALT_PATH) else None,
                      "heartbeat": hb, "limits": vars(cfg["risk"])}, indent=2, ensure_ascii=False))


def seconds_to_next_run(delay=20):
    now = datetime.now(timezone.utc)
    nxt = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1, seconds=delay)
    return (nxt - now).total_seconds()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-trade", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--halt", nargs="?", const="ручная остановка", default=None, metavar="ПРИЧИНА")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.halt is not None:
        cmd_halt(args.halt)
        return 0
    if args.resume:
        cmd_resume()
        return 0
    cfg = config()
    if args.status:
        cmd_status(cfg)
        return 0

    setup_logging()
    bot = Bot(cfg, dry_run=args.dry_run)
    if args.test_trade:
        return 0 if bot.test_trade() else 1
    if args.once:
        bot.step()
        return 0

    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    check_every = cfg["risk"].risk_check_sec
    log.info("Бот запущен: итерация сразу, далее раз в час; проверка риска каждые %d с", check_every)
    seen = Counter()
    while not stop["flag"]:
        try:
            bot.step()
            delta = bot.client.stats - seen               # запросы с прошлой итерации, вкл. проверки риска
            seen = Counter(bot.client.stats)
            log.info("Запросы к API за час: %d (%s)", sum(v for k, v in delta.items() if not k.startswith("cache_hit")),
                     ", ".join(f"{k.replace('/v5/', '')}={v}" for k, v in sorted(delta.items())))
            wait = seconds_to_next_run()
            log.info("Следующая итерация через %.0f с", wait)
        except Exception:
            log.exception("Итерация завершилась ошибкой — повтор через 60 с")
            wait = 60
        end = time.time() + wait
        next_check = time.time() + check_every
        while not stop["flag"] and time.time() < end:
            time.sleep(max(0.0, min(1.0, end - time.time())))
            if time.time() >= next_check and time.time() < end - 5:
                next_check = time.time() + check_every
                try:
                    bot.risk_check()
                except Exception as exc:
                    log.warning("Проверка риска не удалась: %s", exc)
    log.info("Бот остановлен")
    return 0


if __name__ == "__main__":
    sys.exit(main())
