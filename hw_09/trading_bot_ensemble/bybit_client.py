"""Минимальный клиент Bybit V5 REST (linear-фьючерсы) на requests: подпись HMAC, прокси, ретраи."""
import hashlib
import hmac
import json
import logging
import time
from collections import Counter
from urllib.parse import urlencode

import requests

log = logging.getLogger("bybit")

MAINNET = "https://api.bybit.com"
TESTNET = "https://api-testnet.bybit.com"


class BybitError(RuntimeError):
    def __init__(self, ret_code, msg, path):
        super().__init__(f"{path}: retCode={ret_code} {msg}")
        self.ret_code = ret_code


class BybitClient:
    def __init__(self, api_key, api_secret, base_url=MAINNET, proxy=None, recv_window=10000,
                 timeout=20, retries=3, ticker_ttl=5.0):
        self.key, self.secret = api_key, api_secret
        self.base = base_url.rstrip("/")
        self.recv_window = str(recv_window)
        self.timeout, self.retries = timeout, retries
        self.session = requests.Session()
        if proxy:
            self.session.proxies = {"http": proxy, "https": proxy}
        self.time_offset_ms = 0
        self.ticker_ttl = ticker_ttl
        self._ticker_cache, self._instrument_cache = {}, {}
        self.stats = Counter()          # число запросов к API по эндпоинтам (и попаданий в кеш)
        self.sync_time()

    # ── транспорт ─────────────────────────────────────────────────────────────────────────
    def sync_time(self):
        t0 = time.time()
        server = int(self._send("GET", "/v5/market/time", {}, auth=False)["timeNano"]) // 1_000_000
        self.time_offset_ms = server - int((t0 + time.time()) / 2 * 1000)

    def _sign(self, ts, payload):
        msg = f"{ts}{self.key}{self.recv_window}{payload}"
        return hmac.new(self.secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

    def _send(self, method, path, params, auth):
        url = self.base + path
        if method == "GET":
            query = urlencode(params)
            payload, data = query, None
            if query:
                url += "?" + query
        else:
            payload = data = json.dumps(params, separators=(",", ":"))
        last_exc = None
        for attempt in range(1, self.retries + 1):
            self.stats[path] += 1
            headers = {"Content-Type": "application/json"} if data is not None else {}
            if auth:
                ts = str(int(time.time() * 1000) + self.time_offset_ms)
                headers.update({"X-BAPI-API-KEY": self.key, "X-BAPI-TIMESTAMP": ts,
                                "X-BAPI-RECV-WINDOW": self.recv_window,
                                "X-BAPI-SIGN": self._sign(ts, payload)})
            try:
                r = self.session.request(method, url, data=data, headers=headers, timeout=self.timeout)
                r.raise_for_status()
                body = r.json()
            except (requests.RequestException, ValueError) as exc:
                last_exc = exc
                log.warning("%s %s: попытка %d/%d не удалась: %s", method, path, attempt, self.retries, exc)
                time.sleep(2 * attempt)
                continue
            if body.get("retCode") == 10002 and auth and attempt < self.retries:   # рассинхрон часов
                self.sync_time()
                continue
            if body.get("retCode") != 0:
                raise BybitError(body.get("retCode"), body.get("retMsg"), path)
            return body.get("result", {})
        raise ConnectionError(f"{method} {path}: сеть недоступна: {last_exc}")

    def get(self, path, params=None, auth=True):
        return self._send("GET", path, params or {}, auth)

    def post(self, path, params):
        return self._send("POST", path, params, auth=True)

    # ── рынок ─────────────────────────────────────────────────────────────────────────────
    def klines(self, symbol, interval="60", limit=1000, end=None, start=None, category="linear"):
        p = {"category": category, "symbol": symbol, "interval": interval, "limit": limit}
        if start is not None:
            p["start"] = int(start)
        if end is not None:
            p["end"] = int(end)
        return self.get("/v5/market/kline", p, auth=False)["list"]     # новые — первыми

    def last_price(self, symbol, category="linear", max_age=None):
        """Последняя цена; ответ тикера кешируется на max_age секунд (по умолчанию ticker_ttl)."""
        max_age = self.ticker_ttl if max_age is None else max_age
        key = (category, symbol)
        hit = self._ticker_cache.get(key)
        if hit and time.time() - hit[0] < max_age:
            self.stats["cache_hit:/v5/market/tickers"] += 1
            return hit[1]
        t = self.get("/v5/market/tickers", {"category": category, "symbol": symbol}, auth=False)
        price = float(t["list"][0]["lastPrice"])
        self._ticker_cache[key] = (time.time(), price)
        return price

    def instrument(self, symbol, category="linear"):
        """Параметры инструмента (шаг лота, тик) за время работы не меняются — кешируются."""
        key = (category, symbol)
        if key not in self._instrument_cache:
            self._instrument_cache[key] = self.get(
                "/v5/market/instruments-info", {"category": category, "symbol": symbol}, auth=False)["list"][0]
        return self._instrument_cache[key]

    # ── аккаунт и позиции ─────────────────────────────────────────────────────────────────
    def equity_usdt(self):
        acc = self.get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})["list"][0]
        coin = next(c for c in acc["coin"] if c["coin"] == "USDT")
        return float(coin["equity"] or 0), float(acc.get("totalAvailableBalance") or 0)

    def position(self, symbol, category="linear"):
        """(signed size, avgPrice, stopLoss) для one-way режима; size > 0 — long, < 0 — short."""
        lst = self.get("/v5/position/list", {"category": category, "symbol": symbol})["list"]
        p = next((x for x in lst if int(x.get("positionIdx", 0)) == 0), lst[0] if lst else None)
        if not p or float(p["size"] or 0) == 0:
            return 0.0, 0.0, 0.0
        size = float(p["size"]) * (1 if p["side"] == "Buy" else -1)
        return size, float(p["avgPrice"] or 0), float(p["stopLoss"] or 0)

    def set_leverage(self, symbol, leverage, category="linear"):
        try:
            self.post("/v5/position/set-leverage", {"category": category, "symbol": symbol,
                                                    "buyLeverage": str(leverage),
                                                    "sellLeverage": str(leverage)})
        except BybitError as e:
            if e.ret_code != 110043:          # leverage not modified
                raise

    def ensure_one_way_mode(self, symbol, category="linear"):
        """Бот работает в one-way режиме (positionIdx=0). Переключение возможно без позиций и ордеров."""
        lst = self.get("/v5/position/list", {"category": category, "symbol": symbol})["list"]
        if all(int(p.get("positionIdx", 0)) == 0 for p in lst):
            return False
        self.post("/v5/position/switch-mode", {"category": category, "symbol": symbol, "mode": 0})
        return True

    def market_order(self, symbol, side, qty, reduce_only=False, stop_loss=None, category="linear",
                     link_id=None):
        p = {"category": category, "symbol": symbol, "side": side, "orderType": "Market",
             "qty": qty, "positionIdx": 0, "reduceOnly": reduce_only, "timeInForce": "IOC"}
        if stop_loss is not None:
            p.update({"stopLoss": stop_loss, "slTriggerBy": "LastPrice", "tpslMode": "Full",
                      "slOrderType": "Market"})
        if link_id:
            p["orderLinkId"] = link_id
        return self.post("/v5/order/create", p)

    def set_stop_loss(self, symbol, stop_loss, category="linear"):
        return self.post("/v5/position/trading-stop", {
            "category": category, "symbol": symbol, "positionIdx": 0, "tpslMode": "Full",
            "stopLoss": stop_loss, "slTriggerBy": "LastPrice", "slOrderType": "Market"})

    def order(self, symbol, order_id, category="linear"):
        for path in ("/v5/order/realtime", "/v5/order/history"):
            lst = self.get(path, {"category": category, "symbol": symbol, "orderId": order_id})["list"]
            if lst:
                return lst[0]
        return None

    def closed_pnl(self, symbol, limit=5, category="linear"):
        return self.get("/v5/position/closed-pnl", {"category": category, "symbol": symbol,
                                                    "limit": limit})["list"]
