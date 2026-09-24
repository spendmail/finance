"""Локальная имитация Bybit API v5 для офлайн-проверки бота.

Нужна там, где настоящий testnet недоступен (Bybit блокирует, например, IP из США).
Сервер проверяет подпись запросов тем же алгоритмом, что и Bybit (HMAC-SHA256 от
timestamp + api_key + recv_window + payload), отвечает в формате v5 и исполняет рыночные
ордера по последней цене с комиссией тейкера 0.055%. Свечи — хвост data/btc_1h.csv,
сдвинутый так, чтобы последний бар приходился на текущий час.

Запуск:  ./.venv/bin/python mock_bybit/server.py --port 8765
Бот:     ./.venv/bin/python -m trading_bot.bot --base-url http://127.0.0.1:8765 --once
Перемотка на сутки вперёд (при запуске с --reserve N):  GET /mock/advance?bars=24
"""
import argparse
import hashlib
import hmac
import json
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TAKER_FEE = 0.00055


class Exchange:
    def __init__(self, api_key, api_secret, csv_path, balance, reserve=0):
        self.key, self.secret = api_key, api_secret
        bars = pd.read_csv(csv_path).tail(3000 + reserve).reset_index(drop=True)
        bars.columns = [c.lower() for c in bars.columns]
        self.all_bars = bars
        self.visible = 3000             # последние `reserve` баров CSV — «будущее» для /mock/advance
        self.balance = balance
        self.pos = 0.0
        self.entry = 0.0
        self.leverage = "1"
        self.orders = {}
        self.lock = threading.Lock()

    @property
    def bars(self):
        """Видимые бары; последний приходится на текущий (ещё не закрытый) час."""
        b = self.all_bars.iloc[:self.visible]
        now_hour = pd.Timestamp.now(tz="UTC").floor("h").tz_convert(None)
        return b.assign(date=pd.date_range(end=now_hour, periods=len(b), freq="h"))

    def price(self):
        return float(self.bars["close"].iloc[-1])

    def equity(self):
        return self.balance + self.pos * (self.price() - self.entry)

    def fill(self, side, qty, reduce_only):
        signed = qty if side == "Buy" else -qty
        if reduce_only and (self.pos == 0 or signed * self.pos > 0 or abs(signed) > abs(self.pos) + 1e-12):
            return None, "110017", "reduce-only order has same side with current position"
        px = self.price() * (1.0002 if side == "Buy" else 0.9998)
        fee = qty * px * TAKER_FEE
        new = self.pos + signed
        if self.pos != 0 and (signed * self.pos < 0):          # закрытие (части) позиции
            closed = min(abs(signed), abs(self.pos))
            self.balance += closed * (px - self.entry) * (1 if self.pos > 0 else -1)
        if new == 0:
            self.entry = 0.0
        elif self.pos == 0 or new * self.pos < 0:              # открытие или переворот
            self.entry = px
        elif abs(new) > abs(self.pos):                          # наращивание
            self.entry = (self.entry * abs(self.pos) + px * abs(signed)) / abs(new)
        self.pos = round(new, 8)
        self.balance -= fee
        return {"px": px, "fee": fee}, None, None


class Handler(BaseHTTPRequestHandler):
    ex: Exchange = None

    def log_message(self, fmt, *args):
        print("[mock-bybit]", self.command, self.path.split("?")[0], flush=True)

    def _send(self, result=None, code=0, msg="OK"):
        body = json.dumps({"retCode": code, "retMsg": msg, "result": result or {},
                           "time": int(time.time() * 1000)}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _auth(self, payload):
        h = self.headers
        if h.get("X-BAPI-API-KEY") != self.ex.key:
            return "10003", "API key is invalid."
        ts, rw = h.get("X-BAPI-TIMESTAMP", "0"), h.get("X-BAPI-RECV-WINDOW", "5000")
        if abs(int(time.time() * 1000) - int(ts)) > int(rw):
            return "10002", "invalid request, please check your server timestamp or recv_window param"
        expect = hmac.new(self.ex.secret.encode(), f"{ts}{self.ex.key}{rw}{payload}".encode(),
                          hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expect, h.get("X-BAPI-SIGN", "")):
            return "10004", "error sign! origin_string[...]"
        return None, None

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        ex = self.ex
        if u.path == "/mock/advance":           # перемотка: открыть следующие n баров CSV
            with ex.lock:
                ex.visible = min(ex.visible + int(q.get("bars", 1)), len(ex.all_bars))
            return self._send({"visible": ex.visible, "price": ex.price()})
        if u.path == "/v5/market/time":
            t = time.time()
            return self._send({"timeSecond": str(int(t)), "timeNano": str(int(t * 1e9))})
        if u.path == "/v5/market/kline":
            b = ex.bars
            if "end" in q:
                b = b[b["date"] <= pd.to_datetime(int(q["end"]), unit="ms")]
            b = b.tail(int(q.get("limit", 200))).iloc[::-1]
            rows = [[str(int(r.date.timestamp() * 1000)), str(r.open), str(r.high), str(r.low),
                     str(r.close), str(r.volume), str(r.volume * r.close)] for r in b.itertuples()]
            return self._send({"category": "linear", "symbol": q.get("symbol"), "list": rows})
        if u.path == "/v5/market/tickers":
            return self._send({"list": [{"symbol": "BTCUSDT", "lastPrice": str(ex.price()),
                                         "markPrice": str(ex.price())}]})
        if u.path == "/v5/market/instruments-info":
            return self._send({"list": [{"symbol": "BTCUSDT",
                                         "lotSizeFilter": {"qtyStep": "0.001", "minOrderQty": "0.001",
                                                           "maxOrderQty": "1190", "maxMktOrderQty": "119",
                                                           "minNotionalValue": "5"},
                                         "leverageFilter": {"maxLeverage": "100.00"}}]})
        err = self._auth(u.query)
        if err[0]:
            return self._send(code=int(err[0]), msg=err[1])
        with ex.lock:
            if u.path == "/v5/account/wallet-balance":
                e = ex.equity()
                return self._send({"list": [{"accountType": "UNIFIED", "totalEquity": str(e),
                                             "coin": [{"coin": "USDT", "equity": str(e),
                                                       "walletBalance": str(ex.balance),
                                                       "availableToWithdraw": str(ex.balance)}]}]})
            if u.path == "/v5/position/list":
                side = "Buy" if ex.pos > 0 else ("Sell" if ex.pos < 0 else "")
                return self._send({"list": [{"symbol": "BTCUSDT", "side": side, "size": str(abs(ex.pos)),
                                             "avgPrice": str(ex.entry), "leverage": ex.leverage,
                                             "positionIdx": 0}]})
            if u.path in ("/v5/order/realtime", "/v5/order/history"):
                o = ex.orders.get(q.get("orderId"))
                return self._send({"list": [o] if o else []})
        return self._send(code=10001, msg=f"unknown path {u.path}")

    def do_POST(self):
        u = urlparse(self.path)
        payload = self.rfile.read(int(self.headers.get("Content-Length", 0))).decode()
        err = self._auth(payload)
        if err[0]:
            return self._send(code=int(err[0]), msg=err[1])
        p = json.loads(payload or "{}")
        ex = self.ex
        with ex.lock:
            if u.path == "/v5/position/set-leverage":
                if p["buyLeverage"] == ex.leverage:
                    return self._send(code=110043, msg="leverage not modified")
                ex.leverage = p["buyLeverage"]
                return self._send({})
            if u.path == "/v5/order/create":
                qty = float(p["qty"])
                if p.get("orderType") != "Market" or p.get("side") not in ("Buy", "Sell") or qty <= 0:
                    return self._send(code=10001, msg="params error")
                if round(qty / 0.001, 6) % 1:
                    return self._send(code=10001, msg="Qty invalid")
                res, code, msg = ex.fill(p["side"], qty, bool(p.get("reduceOnly")))
                if res is None:
                    return self._send(code=int(code), msg=msg)
                oid = str(uuid.uuid4())
                ex.orders[oid] = {"orderId": oid, "orderLinkId": p.get("orderLinkId") or "",
                                  "symbol": "BTCUSDT", "side": p["side"], "qty": p["qty"],
                                  "cumExecQty": p["qty"], "orderStatus": "Filled",
                                  "avgPrice": f"{res['px']:.2f}", "cumExecFee": f"{res['fee']:.6f}",
                                  "reduceOnly": bool(p.get("reduceOnly"))}
                return self._send({"orderId": oid, "orderLinkId": p.get("orderLinkId") or ""})
        return self._send(code=10001, msg=f"unknown path {u.path}")


def main():
    import os
    import sys
    sys.path.insert(0, str(ROOT))
    from trading_bot.config import load_env
    load_env()
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--balance", type=float, default=10_000.0)
    ap.add_argument("--reserve", type=int, default=0,
                    help="сколько последних баров CSV спрятать для перемотки через /mock/advance")
    args = ap.parse_args()
    Handler.ex = Exchange(os.environ["BYBIT_API_KEY"], os.environ["BYBIT_API_SECRET"],
                          ROOT / "data" / "btc_1h.csv", args.balance, args.reserve)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"[mock-bybit] http://127.0.0.1:{args.port}, баланс {args.balance} USDT", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
