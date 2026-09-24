"""Минимальный клиент Bybit API v5 (REST) для бессрочного фьючерса BTCUSDT.

Только то, что нужно боту: свечи, параметры инструмента, баланс, позиция, рыночный
ордер, плечо, статус ордера. Подпись запросов — HMAC-SHA256 по схеме v5:
sign = HMAC(secret, timestamp + api_key + recv_window + payload), где payload — строка
запроса для GET и тело JSON для POST.

Прокси берётся из переменных окружения HTTPS_PROXY / HTTP_PROXY (так работает requests):
Bybit блокирует запросы из ряда стран, в том числе из США.
"""
import hashlib
import hmac
import json
import time
from decimal import ROUND_DOWN, Decimal
from urllib.parse import urlencode

import pandas as pd
import requests


class BybitError(RuntimeError):
    def __init__(self, ret_code, ret_msg, path):
        super().__init__(f"Bybit {path}: retCode={ret_code} {ret_msg}")
        self.ret_code, self.ret_msg = ret_code, ret_msg


class BybitClient:
    def __init__(self, api_key, api_secret, base_url, recv_window=10_000, timeout=15):
        self.api_key, self.api_secret = api_key, api_secret
        self.base_url = base_url.rstrip("/")
        self.recv_window = str(recv_window)
        self.timeout = timeout
        self.session = requests.Session()
        self._time_offset_ms = 0

    # ── транспорт ───────────────────────────────────────────────────────────
    def _sign(self, ts, payload):
        msg = f"{ts}{self.api_key}{self.recv_window}{payload}"
        return hmac.new(self.api_secret.encode(), msg.encode(), hashlib.sha256).hexdigest()

    def _request(self, method, path, params=None, auth=False, retries=3):
        params = {k: v for k, v in (params or {}).items() if v is not None}
        url = self.base_url + path
        for attempt in range(retries):
            headers = {"Content-Type": "application/json"}
            if method == "GET":
                payload = urlencode(params)
                full_url = f"{url}?{payload}" if payload else url
                body = None
            else:
                payload = json.dumps(params, separators=(",", ":"))
                full_url, body = url, payload
            if auth:
                ts = str(int(time.time() * 1000) + self._time_offset_ms)
                headers.update({
                    "X-BAPI-API-KEY": self.api_key,
                    "X-BAPI-TIMESTAMP": ts,
                    "X-BAPI-RECV-WINDOW": self.recv_window,
                    "X-BAPI-SIGN": self._sign(ts, payload),
                })
            try:
                r = self.session.request(method, full_url, data=body, headers=headers,
                                         timeout=self.timeout)
            except requests.RequestException as e:
                if attempt == retries - 1:
                    raise
                time.sleep(1.5 * (attempt + 1))
                continue
            if r.status_code == 403 and "country" in r.text:
                raise RuntimeError(
                    f"Bybit {self.base_url} заблокировал запрос по стране IP-адреса (HTTP 403). "
                    "Запустите бота с сервера в другой стране или задайте прокси в HTTPS_PROXY.")
            try:
                data = r.json()
            except ValueError:
                raise RuntimeError(f"Bybit {path}: HTTP {r.status_code}, не JSON: {r.text[:200]!r}")
            code = data.get("retCode")
            if code == 0:
                return data.get("result", {})
            # 10002 — расхождение часов с сервером: синхронизируемся и повторяем
            if code == 10002 and attempt < retries - 1:
                self.sync_time()
                continue
            # 10006 / 10016 — лимит запросов / внутренняя ошибка сервера: пауза и повтор
            if code in (10006, 10016) and attempt < retries - 1:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise BybitError(code, data.get("retMsg"), path)
        raise RuntimeError(f"Bybit {path}: исчерпаны повторы")

    def sync_time(self):
        res = self._request("GET", "/v5/market/time")
        server_ms = int(res.get("timeNano", 0)) // 1_000_000 or int(res["timeSecond"]) * 1000
        self._time_offset_ms = server_ms - int(time.time() * 1000)
        return self._time_offset_ms

    # ── рыночные данные (публичные) ─────────────────────────────────────────
    def klines(self, symbol, interval="60", bars=1000, category="linear"):
        """Последние `bars` свечей, по возрастанию времени; незакрытая свеча включена.

        Bybit отдаёт максимум 1000 свечей за запрос и в обратном порядке, поэтому
        история собирается несколькими запросами, сдвигая `end` назад.
        """
        rows, end = [], None
        while len(rows) < bars:
            res = self._request("GET", "/v5/market/kline", {
                "category": category, "symbol": symbol, "interval": interval,
                "limit": min(1000, bars - len(rows)), "end": end})
            chunk = res.get("list", [])
            if not chunk:
                break
            rows.extend(chunk)
            end = int(chunk[-1][0]) - 1
            if len(chunk) < 2:
                break
        df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume", "turnover"])
        df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
        for c in ("open", "high", "low", "close", "volume"):
            df[c] = df[c].astype(float)
        return (df.drop_duplicates("date").sort_values("date").reset_index(drop=True)
                [["date", "open", "high", "low", "close", "volume"]])

    def ticker(self, symbol, category="linear"):
        return self._request("GET", "/v5/market/tickers",
                             {"category": category, "symbol": symbol})["list"][0]

    def instrument(self, symbol, category="linear"):
        info = self._request("GET", "/v5/market/instruments-info",
                             {"category": category, "symbol": symbol})["list"][0]
        lot = info["lotSizeFilter"]
        return {"qty_step": Decimal(lot["qtyStep"]), "min_qty": Decimal(lot["minOrderQty"]),
                "max_qty": Decimal(lot.get("maxMktOrderQty") or lot["maxOrderQty"]),
                "min_notional": Decimal(lot.get("minNotionalValue", "0") or "0"),
                "max_leverage": Decimal(info["leverageFilter"]["maxLeverage"])}

    # ── аккаунт (приватные) ─────────────────────────────────────────────────
    def wallet_equity(self, coin="USDT", account_type="UNIFIED"):
        """(equity, available) по монете: капитал с нереализованным PnL и свободные средства."""
        res = self._request("GET", "/v5/account/wallet-balance",
                            {"accountType": account_type, "coin": coin}, auth=True)
        acct = res["list"][0]
        for c in acct.get("coin", []):
            if c["coin"] == coin:
                equity = float(c.get("equity") or c.get("walletBalance") or 0)
                avail = c.get("availableToWithdraw") or acct.get("totalAvailableBalance") or equity
                return equity, float(avail or 0)
        return 0.0, 0.0

    def position(self, symbol, category="linear"):
        """Позиция со знаком (+ long, − short) в BTC и её параметры (режим one-way)."""
        res = self._request("GET", "/v5/position/list",
                            {"category": category, "symbol": symbol}, auth=True)
        size, info = 0.0, {}
        for p in res.get("list", []):
            s = float(p.get("size") or 0)
            if s:
                size += s if p["side"] == "Buy" else -s
                info = p
        if not info and res.get("list"):
            info = res["list"][0]
        return size, info

    def set_leverage(self, symbol, leverage, category="linear"):
        try:
            self._request("POST", "/v5/position/set-leverage", {
                "category": category, "symbol": symbol,
                "buyLeverage": str(leverage), "sellLeverage": str(leverage)}, auth=True)
        except BybitError as e:
            if e.ret_code != 110043:          # 110043 — плечо уже такое
                raise

    def market_order(self, symbol, side, qty, reduce_only=False, category="linear", link_id=None):
        return self._request("POST", "/v5/order/create", {
            "category": category, "symbol": symbol, "side": side, "orderType": "Market",
            "qty": str(qty), "timeInForce": "IOC", "positionIdx": 0,
            "reduceOnly": reduce_only, "orderLinkId": link_id}, auth=True)

    def order_status(self, symbol, order_id, category="linear"):
        for path in ("/v5/order/realtime", "/v5/order/history"):
            res = self._request("GET", path, {"category": category, "symbol": symbol,
                                              "orderId": order_id}, auth=True)
            if res.get("list"):
                return res["list"][0]
        return None

    def wait_filled(self, symbol, order_id, timeout=20.0, category="linear"):
        """Ждём конечный статус рыночного ордера: Filled / Cancelled / Rejected ..."""
        deadline, st = time.time() + timeout, None
        while time.time() < deadline:
            st = self.order_status(symbol, order_id, category)
            if st and st.get("orderStatus") in ("Filled", "Cancelled", "Rejected",
                                                 "PartiallyFilledCanceled", "Deactivated"):
                return st
            time.sleep(0.7)
        return st


def round_qty(qty, step):
    """Округление количества вниз до шага лота."""
    q = (Decimal(str(abs(qty))) / step).to_integral_value(rounding=ROUND_DOWN) * step
    return q
