"""Торговый бот BTC/USDT perpetual на Bybit: CatBoost + стоп-лосс 3% без тейк-профита.

Каждый час, после закрытия часовой свечи:
  1. скачивает ~2000 последних закрытых часовых свечей BTCUSDT (linear) с Bybit;
  2. считает 18 признаков финальной модели ноутбука и вероятность роста CatBoost;
  3. переводит вероятность в целевую позицию торговым слоем ноутбука: EWM(span=4), пороги —
     квантили 2%/98% прогнозов модели на dev, режим hold, пересмотр позиции раз в сутки
     (на свече, закрывающейся в 00:00 UTC);
  4. пропускает рекомендацию через риск-менеджер (risk.py): лимиты просадки, суточного убытка,
     числа сделок, размера позиции, проверки данных;
  5. приводит позицию на бирже к целевой рыночным ордером и ставит на позицию стоп-лосс 3% от
     цены входа; тейк-профита нет;
  6. если позицию закрыл стоп, остаётся вне рынка, пока знак целевой позиции не сменится.
Между часовыми итерациями каждые RISK_CHECK_SEC секунд проверяются equity, стоп-лосс и рубильник.

Запуск:
  python bot.py                 — бесконечный цикл
  python bot.py --once          — одна итерация и выход
  python bot.py --dry-run       — считать сигналы, но не отправлять ордера
  python bot.py --test-trade    — проверка исполнения: открыть минимальную позицию по сигналу
                                   модели со стопом 3%, проверить её на бирже и закрыть
  python bot.py --status        — состояние бота, позиция и лимиты риска
  python bot.py --halt [причина] — рубильник: работающий бот закроет позицию и остановит торговлю
  python bot.py --resume        — снять остановку (удалить файл HALT)
"""
import argparse
import csv
import json
import logging
import math
import os
import signal
import sys
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler

import pandas as pd
from catboost import CatBoostClassifier

from bybit_client import MAINNET, TESTNET, BybitClient
from features import MODEL_COLS, build_features, finalize
from risk import RiskConfig, RiskManager
from strategy import EXIT_PARAMS, target_positions

HERE = os.path.dirname(os.path.abspath(__file__))
ART_DIR = os.path.join(HERE, "artifacts")
# Изменяемые файлы (состояние, логи, рубильник) — в одном каталоге: в Docker это volume.
RUNTIME_DIR = os.environ.get("BOT_RUNTIME_DIR") or os.path.join(HERE, "runtime")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
STATE_PATH = os.path.join(RUNTIME_DIR, "state.json")
HALT_PATH = os.path.join(RUNTIME_DIR, "HALT")
HEARTBEAT_PATH = os.path.join(RUNTIME_DIR, "heartbeat")
HISTORY_BARS = 2000        # свечей на каждую итерацию: прогрев признаков + память hold-сигнала
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
        # доля капитала в позиции: «позиция 1» торгового слоя = POSITION_FRACTION x equity
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
    path = os.path.join(LOG_DIR, name)
    new = not os.path.exists(path)
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sign(x):
    return 1 if x > 0 else (-1 if x < 0 else 0)


SIDE_NAME = {1: "LONG", -1: "SHORT", 0: "FLAT"}


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as fh:
            return json.load(fh)
    return {"side": 0, "entry": 0.0, "blocked": 0, "last_bar": None, "risk": {}}


class Bot:
    def __init__(self, cfg, dry_run=False):
        self.cfg, self.dry_run = cfg, dry_run
        self.symbol = cfg["symbol"]
        self.client = BybitClient(cfg["api_key"], cfg["api_secret"], cfg["base_url"], cfg["proxy"])
        with open(os.path.join(ART_DIR, "meta.json")) as fh:
            self.meta = json.load(fh)
        assert self.meta["model_cols"] == MODEL_COLS, "artifacts/ обучены на другом наборе признаков"
        self.model = CatBoostClassifier()
        self.model.load_model(os.path.join(ART_DIR, "catboost.cbm"))
        self.w_lo = pd.Series(self.meta["winsor_lo"])[MODEL_COLS]
        self.w_hi = pd.Series(self.meta["winsor_hi"])[MODEL_COLS]
        self.hi, self.lo = self.meta["threshold_hi"], self.meta["threshold_lo"]
        self.sl = EXIT_PARAMS["sl"]

        inst = self.client.instrument(self.symbol)
        self.qty_step = float(inst["lotSizeFilter"]["qtyStep"])
        self.min_qty = float(inst["lotSizeFilter"]["minOrderQty"])
        self.tick = float(inst["priceFilter"]["tickSize"])
        self.state = load_state()
        self.state.setdefault("risk", {})
        self.risk = RiskManager(cfg["risk"], self.state["risk"], HALT_PATH, write_files=not dry_run)
        log.info("Bybit %s, %s: шаг лота %s, мин. лот %s, тик %s; пороги слоя %.5f / %.5f; SL %.1f%%%s",
                 cfg["base_url"], self.symbol, self.qty_step, self.min_qty, self.tick, self.hi, self.lo,
                 self.sl * 100, " [DRY-RUN]" if dry_run else "")
        log.info("Лимиты риска: %s; плечо %s", cfg["risk"], cfg["leverage"])
        if not dry_run:
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
        rows, end = [], None
        while len(rows) < n + 1:
            chunk = self.client.klines(self.symbol, "60", 1000, end=end)
            if not chunk:
                break
            rows.extend(chunk)
            end = int(chunk[-1][0]) - 1
        df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
        df = df.astype(float).drop_duplicates("ts").sort_values("ts")
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
        now_hour = pd.Timestamp.now(tz="UTC").tz_convert(None).floor("h")
        df = df[df["date"] < now_hour]                           # незакрытая свеча не нужна
        df = df.tail(n).reset_index(drop=True)[["date", "open", "high", "low", "close", "volume"]]
        gaps = int((df["date"].diff().dropna() != pd.Timedelta(hours=1)).sum())
        if gaps:
            raise RuntimeError(f"в истории свечей {gaps} разрывов — признаки были бы смещены")
        if len(df) < VALID_FROM + 100:
            raise RuntimeError(f"биржа отдала только {len(df)} свечей — мало для прогрева признаков")
        return df

    def compute_signal(self):
        bars = self.fetch_bars()
        feat = finalize(build_features(bars), warmup=0)           # news_count_168 = 0, как в walk-forward
        feat = feat.iloc[VALID_FROM:].reset_index(drop=True)
        X = feat[MODEL_COLS].clip(lower=self.w_lo, upper=self.w_hi, axis=1)
        if not X.notna().all().all():
            raise RuntimeError("в признаках есть пропуски — рекомендация не формируется")
        proba = self.model.predict_proba(X)[:, 1]
        # бар с меткой открытия T закрывается в T+1ч: пересмотр на свече, закрывающейся в 00:00 UTC
        rebalance = ((feat["date"] + pd.Timedelta(hours=1)).dt.hour == REBALANCE_HOUR_UTC).to_numpy()
        target, sig, p_s = target_positions(proba, self.hi, self.lo, rebalance)
        last_reb = feat["date"][rebalance].iloc[-1] if rebalance.any() else None
        return {"bar": feat["date"].iloc[-1], "close": float(feat["close"].iloc[-1]),
                "proba": float(proba[-1]), "p_smooth": float(p_s[-1]), "signal": int(sig[-1]),
                "target": int(target[-1]), "last_rebalance_bar": last_reb}

    # ── исполнение ────────────────────────────────────────────────────────────────────────
    def round_qty(self, q):
        return math.floor(q / self.qty_step + 1e-9) * self.qty_step

    def fmt_qty(self, q):
        decimals = max(0, -int(math.floor(math.log10(self.qty_step))))
        return f"{q:.{decimals}f}"

    def fmt_price(self, p, side):
        """Цена стопа по тику: у лонга вниз, у шорта вверх (стоп не ближе 3%)."""
        k = math.floor(p / self.tick) if side > 0 else math.ceil(p / self.tick)
        decimals = max(0, -int(math.floor(math.log10(self.tick))))
        return f"{k * self.tick:.{decimals}f}"

    def stop_price(self, entry, side):
        return self.fmt_price(entry * (1 - self.sl) if side > 0 else entry * (1 + self.sl), side)

    def wait_position(self, want_side, timeout=15):
        for _ in range(timeout):
            size, avg, sl = self.client.position(self.symbol)
            if sign(size) == want_side:
                return size, avg, sl
            time.sleep(1)
        return self.client.position(self.symbol)

    def close_position(self, size, reason, count_pnl=True):
        """Закрыть позицию рыночным reduceOnly-ордером; вернуть реализованный PnL."""
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
        self.state["side"], self.state["entry"] = 0, 0.0
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
        self.state["side"], self.state["entry"] = side, avg
        # стоп — ровно 3% от фактической цены входа, а не от цены до ордера
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
        if not sl_set:                    # позиция без стопа недопустима
            self.close_position(size, "стоп-лосс не выставлен — аварийное закрытие")
            self.risk.halt("не удалось поставить стоп-лосс на позицию")
            return 0.0, 0.0
        log.info("Позиция открыта: %s %s BTC по %.1f, стоп-лосс на бирже %.1f",
                 SIDE_NAME[side], self.fmt_qty(abs(size)), avg, sl_set)
        return abs(size), avg

    def detect_external_close(self, cur):
        """Позиция исчезла без участия бота — сработал стоп-лосс на бирже (или закрыли вручную)."""
        st = self.state
        if st["side"] == 0 or cur != 0:
            return
        st["blocked"] = st["side"]
        pnl = self.client.closed_pnl(self.symbol, limit=1)
        realized = float(pnl[0]["closedPnl"]) if pnl else float("nan")
        log.warning("Позицию %s закрыл стоп-лосс (PnL %s USDT): вне рынка до смены сигнала",
                    SIDE_NAME[st["side"]], pnl[0]["closedPnl"] if pnl else "?")
        append_csv("trades.csv", {"time": now_iso(), "action": "stop_loss", "side": SIDE_NAME[st["side"]],
                                  "qty": "", "price": pnl[0]["avgExitPrice"] if pnl else "",
                                  "stop_loss": "", "order_id": pnl[0].get("orderId") if pnl else "",
                                  "realized_pnl": realized, "reason": "stop-loss на бирже"})
        self.risk.on_close(realized)
        st["side"], st["entry"] = 0, 0.0

    def enforce_risk(self, size):
        """Проверить equity и рубильник; при halt/pause закрыть позицию. Вернуть статус или None."""
        equity, _ = self.client.equity_usdt()
        status = self.risk.check_equity(equity)
        if status and size != 0:
            self.close_position(size, f"риск-менеджер: {self.state['risk'].get('halted') or 'суточная пауза'}",
                                count_pnl=False)
            self.state["blocked"] = 0
        return status, equity

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
            log.info("Знак цели сменился (%s -> %s): блокировка после стопа снята",
                     SIDE_NAME[st["blocked"]], SIDE_NAME[target])
            st["blocked"] = 0
        recommended = 0 if st["blocked"] else target
        desired = 0 if risk_status else recommended

        log.info("Свеча %s close=%.1f | p=%.4f EWM=%.4f (пороги %.4f/%.4f) | сигнал %s | цель %s "
                 "(пересмотр %s) | блок %s | риск %s | на бирже %s %s",
                 s["bar"], s["close"], s["proba"], s["p_smooth"], self.lo, self.hi,
                 SIDE_NAME[s["signal"]], SIDE_NAME[target], s["last_rebalance_bar"],
                 SIDE_NAME[st["blocked"]], risk_status or "OK", SIDE_NAME[cur], abs(size))

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
                reason = f"цель {SIDE_NAME[desired]} (сигнал модели p_ewm={s['p_smooth']:.4f})"
                if cur != 0:
                    self.close_position(size, reason)
                    action = "close"
                if desired != 0 and not self.risk.st["halted"]:
                    qty, _ = self.open_position(desired, reason)
                    if qty:
                        action = "open" if cur == 0 else "reverse"
        elif cur != 0:
            st["side"], st["entry"] = cur, st.get("entry") or avg
            if sl_on_exchange == 0 and not self.dry_run:
                self.repair_stop(size, avg)

        st["last_bar"] = str(s["bar"])
        self.save_state()
        equity, _ = self.client.equity_usdt()
        append_csv("signals.csv", {"time": now_iso(), "bar": str(s["bar"]), "close": s["close"],
                                   "proba": round(s["proba"], 6), "p_ewm": round(s["p_smooth"], 6),
                                   "signal": s["signal"], "target": target, "blocked": st["blocked"],
                                   "recommended": recommended, "desired": desired,
                                   "position_before": cur, "action": action, "veto": veto,
                                   "equity": round(equity, 4), **self.risk.metrics(equity),
                                   "dry_run": int(self.dry_run)})
        self.heartbeat()
        return s, action

    def repair_stop(self, size, avg):
        """Позиция без стопа: восстановить его, а если не получается — закрыть позицию."""
        cur = sign(size)
        sl_px = self.stop_price(self.state.get("entry") or avg, cur)
        log.warning("На позиции нет стоп-лосса — ставлю %s", sl_px)
        try:
            self.client.set_stop_loss(self.symbol, sl_px)
        except Exception as exc:
            log.error("Стоп не восстановлен (%s) — аварийное закрытие позиции", exc)
            self.close_position(size, "стоп-лосс не восстановлен", count_pnl=False)
            self.risk.halt("не удалось восстановить стоп-лосс на позиции")

    def risk_check(self):
        """Лёгкая проверка между часовыми итерациями: стоп-лосс, equity, рубильник."""
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
        """Минимальная сделка по текущему сигналу модели: открыть со стопом, проверить, закрыть."""
        size, _, _ = self.client.position(self.symbol)
        if size != 0:
            raise RuntimeError("на бирже уже есть позиция — тестовая сделка не выполняется")
        s = self.compute_signal()
        side = s["signal"] or (1 if s["p_smooth"] >= 0.5 else -1)
        log.info("ТЕСТ: сигнал модели на свече %s — p=%.4f, EWM=%.4f, hold-сигнал %s, цель %s -> "
                 "тестовая сторона %s", s["bar"], s["proba"], s["p_smooth"], SIDE_NAME[s["signal"]],
                 SIDE_NAME[s["target"]], SIDE_NAME[side])
        qty, entry = self.open_position(side, "тестовая сделка", qty=self.min_qty)
        if not qty:
            return False
        size, avg, sl = self.client.position(self.symbol)
        expected = float(self.stop_price(avg, side))
        ok = sign(size) == side and abs(sl - expected) < self.tick * 1.5
        log.info("ТЕСТ: на бирже %s %s BTC, вход %.1f, стоп-лосс %.1f (ожидался %.1f, %.2f%% от входа) — %s",
                 SIDE_NAME[sign(size)], abs(size), avg, sl, expected, abs(sl / avg - 1) * 100,
                 "OK" if ok else "НЕ СОВПАДАЕТ")
        time.sleep(3)
        self.close_position(size, "тестовая сделка: закрытие", count_pnl=False)
        size_after, _, _ = self.client.position(self.symbol)
        log.info("ТЕСТ: позиция после закрытия %s — %s", size_after, "OK" if size_after == 0 else "ОШИБКА")
        return ok and size_after == 0


# ── команды управления, не требующие запуска бота ─────────────────────────────────────────
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
    while not stop["flag"]:
        try:
            bot.step()
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
