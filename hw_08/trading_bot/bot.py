"""Торговый бот BTCUSDT (бессрочный фьючерс USDT) на Bybit testnet.

Цикл: на закрытии каждого часового бара бот загружает свечи, пересчитывает признаки,
записывает прогнозы моделей в logs/decisions.csv, а в момент решения (раз в `every`
баров, как у агентов ноутбука: k=24 — раз в сутки) переводит целевую долю капитала в
количество BTC и выставляет рыночный ордер на разницу с текущей позицией.

Примеры:
  # один цикл сразу, без ожидания часа (решение принимается принудительно)
  ./.venv/bin/python -m trading_bot.bot --once
  # проверка связи и исполнения: купить минимальный лот и закрыть его
  ./.venv/bin/python -m trading_bot.bot --roundtrip-test
  # постоянная работа (стратегия по умолчанию — RL на вершине ансамбля)
  ./.venv/bin/python -m trading_bot.bot
  # демонстрация: решение на каждом часовом баре вместо раза в сутки
  ./.venv/bin/python -m trading_bot.bot --every 1
"""
import argparse
import csv
import json
import logging
import os
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pandas as pd

from . import config as C
from .bybit_client import BybitClient, round_qty
from .strategy import STRATEGIES, LiveModel

log = logging.getLogger("bot")


# ── состояние и журналы ─────────────────────────────────────────────────────
def load_state(n_seeds, strategy):
    if C.STATE_PATH.exists():
        st = json.loads(C.STATE_PATH.read_text())
        if st.get("strategy") == strategy and len(st.get("seed_positions", [])) == n_seeds:
            return st
        log.warning("состояние от другой стратегии — начинаем с нуля")
    return {"strategy": strategy, "seed_positions": [0.0] * n_seeds, "target": 0.0,
            "last_decision_bar": None, "last_bar": None}


def save_state(st):
    C.STATE_PATH.write_text(json.dumps(st, indent=2, ensure_ascii=False))


def append_csv(path, row):
    new = not path.exists()
    with open(path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)


# ── бот ─────────────────────────────────────────────────────────────────────
class TradingBot:
    def __init__(self, args):
        self.args = args
        key, secret = os.environ.get("BYBIT_API_KEY"), os.environ.get("BYBIT_API_SECRET")
        if not key or not secret:
            raise SystemExit("нет BYBIT_API_KEY / BYBIT_API_SECRET (файл .env или переменные окружения)")
        self.ex = BybitClient(key, secret, args.base_url)
        self.data = BybitClient("", "", args.data_url) if args.data_url != args.base_url else self.ex
        self.model = LiveModel()
        self.every = args.every or self.model.freq["every"]
        self.state = load_state(self.model.n_seeds, args.strategy)

    def setup(self):
        off = self.ex.sync_time()
        log.info("биржа %s, расхождение часов %+d мс", self.args.base_url, off)
        self.inst = self.ex.instrument(C.SYMBOL)
        log.info("инструмент %s: шаг лота %s, мин. лот %s, макс. плечо %s", C.SYMBOL,
                 self.inst["qty_step"], self.inst["min_qty"], self.inst["max_leverage"])
        eq, avail = self.ex.wallet_equity()
        pos, _ = self.ex.position(C.SYMBOL)
        log.info("капитал %.2f USDT (свободно %.2f), позиция %+.4f BTC", eq, avail, pos)
        if not self.args.dry_run:
            self.ex.set_leverage(C.SYMBOL, self.args.leverage)
            log.info("плечо на бирже: %sx", self.args.leverage)

    def closed_bars(self):
        """Закрытые часовые бары: последняя свеча Bybit ещё формируется и отбрасывается."""
        bars = self.data.klines(C.SYMBOL, "60", C.LIVE_HISTORY_BARS + 1)
        now = pd.Timestamp.now(tz="UTC").tz_convert(None)
        bars = bars[bars["date"] + pd.Timedelta(hours=1) <= now].reset_index(drop=True)
        full = pd.date_range(bars["date"].iloc[0], bars["date"].iloc[-1], freq="h")
        if len(full) != len(bars):
            log.warning("в свечах пропущено %d часов — заполняю предыдущим закрытием",
                        len(full) - len(bars))
            bars = bars.set_index("date").reindex(full)
            bars["close"] = bars["close"].ffill()
            for c in ("open", "high", "low"):
                bars[c] = bars[c].fillna(bars["close"])
            bars["volume"] = bars["volume"].fillna(0.0)
            bars = bars.rename_axis("date").reset_index()
        return bars

    def decision_due(self, bar_time, force):
        if force or self.state["last_decision_bar"] is None:
            return True
        hours = int(bar_time.timestamp() // 3600)
        return hours % self.every == self.args.decision_hour % self.every

    def step(self, force=False):
        bars = self.closed_bars()
        feat = self.model.features(bars)
        bar_time = feat["date"].iloc[-1]
        if not force and self.state["last_bar"] == str(bar_time):
            return False                                    # этот бар уже обработан
        due = self.decision_due(bar_time, force)
        seeds_before = list(self.state["seed_positions"])
        target, seeds_after, info = self.model.decide(feat, self.args.strategy, seeds_before)
        if not due:
            target, seeds_after = self.state["target"], seeds_before
        log.info("бар %s close=%.1f | p(рост): LogReg %.3f Tree %.3f CatBoost %.3f ансамбль %.3f | "
                 "%s -> цель %+.3f%s", info["bar"], info["close"], info["proba"]["LogisticRegression"],
                 info["proba"]["DecisionTree"], info["proba"]["CatBoost"], info["proba"]["Ensemble"],
                 "РЕШЕНИЕ" if due else "держим", target,
                 f" (сиды {info.get('raw_actions')})" if due and "raw_actions" in info else "")
        append_csv(C.DECISIONS_LOG, {
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bar": info["bar"], "close": info["close"], "strategy": self.args.strategy,
            **{f"p_{k}": round(v, 5) for k, v in info["proba"].items()},
            "decision": int(due), "raw_actions": json.dumps(info.get("raw_actions")),
            "target": round(target, 4)})
        if due:
            self.state.update(seed_positions=seeds_after, target=target,
                              last_decision_bar=str(bar_time))
        self.state["last_bar"] = str(bar_time)
        self.rebalance(target, reason="decision" if due else "sync")
        save_state(self.state)
        return True

    def rebalance(self, target, reason):
        """Довести позицию на бирже до target × капитал × множитель (в BTC)."""
        equity, _ = self.ex.wallet_equity()
        price = float(self.ex.ticker(C.SYMBOL)["lastPrice"])
        cur, _ = self.ex.position(C.SYMBOL)
        want = target * equity * self.args.capital_frac / price
        diff = want - cur
        qty = round_qty(diff, self.inst["qty_step"])
        qty = min(qty, self.inst["max_qty"])
        # мелкое расхождение (дрейф капитала, округление) не торгуем — тот же смысл,
        # что у deadband агента: сделка меньше порога дороже, чем польза от неё
        min_trade = max(self.inst["min_qty"],
                        Decimal(str(self.args.min_rebalance * equity * self.args.capital_frac / price)))
        log.info("капитал %.2f USDT, цена %.1f, позиция %+.4f BTC, цель %+.4f BTC (Δ %+.4f)",
                 equity, price, cur, want, diff)
        if qty < min_trade:
            log.info("изменение меньше порога (%s BTC) — ордер не нужен", min_trade.normalize())
            return None
        side = "Buy" if diff > 0 else "Sell"
        reduce_only = abs(want) < abs(cur) and want * cur >= 0     # только уменьшение позиции
        if self.args.dry_run:
            log.info("[dry-run] %s %s BTC (reduceOnly=%s)", side, qty, reduce_only)
            return None
        link = f"hw8-{uuid.uuid4().hex[:16]}"
        res = self.ex.market_order(C.SYMBOL, side, qty, reduce_only=reduce_only, link_id=link)
        st = self.ex.wait_filled(C.SYMBOL, res["orderId"]) or {}
        new_pos, _ = self.ex.position(C.SYMBOL)
        log.info("ордер %s %s %s BTC: статус %s, средняя цена %s, позиция теперь %+.4f BTC",
                 res["orderId"], side, qty, st.get("orderStatus"), st.get("avgPrice"), new_pos)
        append_csv(C.TRADES_LOG, {
            "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "reason": reason, "strategy": self.args.strategy, "target_frac": round(target, 4),
            "side": side, "qty": str(qty), "reduce_only": reduce_only,
            "order_id": res["orderId"], "link_id": link, "status": st.get("orderStatus"),
            "avg_price": st.get("avgPrice"), "exec_fee": st.get("cumExecFee"),
            "position_before": cur, "position_after": new_pos, "equity": equity})
        return st

    def roundtrip_test(self):
        """Проверка исполнения без модели: минимальный лот в лонг и обратно (reduceOnly)."""
        qty = self.inst["min_qty"]
        before, _ = self.ex.position(C.SYMBOL)
        for side, ro in (("Buy", False), ("Sell", True)):
            res = self.ex.market_order(C.SYMBOL, side, qty, reduce_only=ro,
                                       link_id=f"hw8-rt-{uuid.uuid4().hex[:12]}")
            st = self.ex.wait_filled(C.SYMBOL, res["orderId"]) or {}
            pos, _ = self.ex.position(C.SYMBOL)
            log.info("roundtrip: %s %s BTC -> %s по %s, позиция %+.4f BTC", side, qty,
                     st.get("orderStatus"), st.get("avgPrice"), pos)
            append_csv(C.TRADES_LOG, {
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "reason": "roundtrip_test", "strategy": "-", "target_frac": "",
                "side": side, "qty": str(qty), "reduce_only": ro, "order_id": res["orderId"],
                "link_id": "", "status": st.get("orderStatus"), "avg_price": st.get("avgPrice"),
                "exec_fee": st.get("cumExecFee"), "position_before": before,
                "position_after": pos, "equity": ""})
            before = pos

    def run(self):
        """Бесконечный цикл: просыпаемся через delay секунд после начала каждого часа."""
        self.step(force=self.state["last_decision_bar"] is None)
        while True:
            now = datetime.now(timezone.utc)
            nxt = (now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
                   + timedelta(seconds=self.args.bar_delay))
            log.info("следующая проверка в %s UTC", nxt.strftime("%H:%M:%S"))
            time.sleep(max((nxt - datetime.now(timezone.utc)).total_seconds(), 1))
            for attempt in range(5):
                try:
                    if self.step():
                        break
                    time.sleep(20)                  # биржа ещё не отдала новый бар
                except Exception:
                    log.exception("ошибка цикла, повтор через 30 с")
                    time.sleep(30)


def main():
    C.load_env()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategy", choices=STRATEGIES, default=os.environ.get("BOT_STRATEGY", "rl_ensemble"))
    ap.add_argument("--base-url", default=os.environ.get("BYBIT_BASE_URL", C.TESTNET_URL),
                    help="REST биржи для ордеров (по умолчанию testnet)")
    ap.add_argument("--data-url", default=os.environ.get("BYBIT_DATA_URL"),
                    help="откуда брать свечи; по умолчанию — тот же хост, что и для ордеров")
    ap.add_argument("--capital-frac", type=float, default=float(os.environ.get("BOT_CAPITAL_FRAC", 0.5)),
                    help="доля капитала, соответствующая позиции 1.0 агента (по умолчанию 0.5)")
    ap.add_argument("--leverage", type=int, default=int(os.environ.get("BOT_LEVERAGE", 2)))
    ap.add_argument("--every", type=int, default=None,
                    help="решение раз в N баров (по умолчанию как при обучении — 24)")
    ap.add_argument("--decision-hour", type=int, default=0, help="час UTC для решений при every=24")
    ap.add_argument("--min-rebalance", type=float, default=0.02,
                    help="не торговать, если нужно изменить позицию меньше чем на эту долю")
    ap.add_argument("--bar-delay", type=int, default=15, help="секунд после закрытия бара")
    ap.add_argument("--once", action="store_true", help="один цикл с принудительным решением")
    ap.add_argument("--roundtrip-test", action="store_true", help="тест исполнения ордеров")
    ap.add_argument("--dry-run", action="store_true", help="считать, но не выставлять ордера")
    ap.add_argument("--reset-state", action="store_true", help="забыть позиции агентов")
    args = ap.parse_args()
    args.data_url = args.data_url or args.base_url

    C.LOGS_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(C.LOGS_DIR / "bot.log", encoding="utf-8")])
    if args.reset_state and C.STATE_PATH.exists():
        C.STATE_PATH.unlink()
    bot = TradingBot(args)
    log.info("стратегия %s, решение раз в %d бар(ов), доля капитала %.2f, плечо %dx%s",
             args.strategy, bot.every, args.capital_frac, args.leverage,
             " [DRY-RUN]" if args.dry_run else "")
    bot.setup()
    if args.roundtrip_test:
        bot.roundtrip_test()
    elif args.once:
        bot.step(force=True)
    else:
        bot.run()


if __name__ == "__main__":
    main()
