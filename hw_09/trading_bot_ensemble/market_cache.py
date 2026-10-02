"""Кеш рыночных данных Bybit: закрытые часовые свечи BTCUSDT.

Закрытая свеча после закрытия часа больше не меняется, поэтому скачивать каждый час заново всю
историю (2000 свечей — три запроса по 1000) не нужно. Кеш держит закрытые свечи в памяти и в
CSV-файле в runtime/ и на каждой итерации догружает только новые: обычно один запрос на
3 свечи (последние 2 кешированные — для сверки + новая). Если бот простоял дольше
~1000 часов или кеш не сошёлся с биржей, история скачивается целиком, как раньше.

Защита от рассинхронизации:
  - в кеш попадают только закрытые свечи (open time < начало текущего часа);
  - перекрытие с последними кешированными свечами сверяется с ответом биржи; при расхождении,
    разрывах в сетке или слишком длинной паузе кеш сбрасывается и история скачивается целиком;
  - файл кеша пишется атомарно; повреждённый файл игнорируется.
"""
import logging
import os

import pandas as pd

log = logging.getLogger("cache")

COLS = ["date", "open", "high", "low", "close", "volume"]
PAGE = 1000            # максимум свечей в одном ответе /v5/market/kline
OVERLAP = 2            # сколько уже известных свечей запрашивать повторно для сверки


def current_hour():
    return pd.Timestamp.now(tz="UTC").tz_convert(None).floor("h")


def to_frame(rows):
    df = pd.DataFrame([r[:6] for r in rows], columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.astype(float).drop_duplicates("ts").sort_values("ts")
    df["date"] = pd.to_datetime(df["ts"].astype("int64"), unit="ms")
    return df[COLS].reset_index(drop=True)


class KlineCache:
    def __init__(self, client, symbol, path, keep=3000, interval="60"):
        self.client, self.symbol, self.path = client, symbol, path
        self.keep, self.interval = keep, interval
        self.bars = self._load()
        self.last = {"requests": 0, "from_cache": 0, "downloaded": 0}

    # ── файл ──────────────────────────────────────────────────────────────────────────────
    def _load(self):
        if not os.path.exists(self.path):
            return pd.DataFrame(columns=COLS)
        try:
            df = pd.read_csv(self.path, parse_dates=["date"])[COLS]
            if df.empty or not self._continuous(df):
                raise ValueError("разрывы в сетке свечей")
            log.info("Кеш свечей загружен с диска: %d свечей до %s", len(df), df["date"].iloc[-1])
            return df
        except Exception as exc:
            log.warning("Файл кеша %s не используется (%s) — история будет скачана заново", self.path, exc)
            return pd.DataFrame(columns=COLS)

    def _save(self):
        tmp = self.path + ".tmp"
        self.bars.to_csv(tmp, index=False)
        os.replace(tmp, self.path)

    @staticmethod
    def _continuous(df):
        return bool((df["date"].diff().dropna() == pd.Timedelta(hours=1)).all())

    # ── загрузка с биржи ──────────────────────────────────────────────────────────────────
    def _request(self, limit, end=None, start=None):
        self.last["requests"] += 1
        return self.client.klines(self.symbol, self.interval, limit, end=end, start=start)

    def _download_full(self, n):
        """История целиком: страницы по 1000 свечей назад от текущего момента."""
        rows, end = [], None
        while len(rows) < n + 1:                       # +1 — незакрытая текущая свеча
            chunk = self._request(PAGE, end=end)
            if not chunk:
                break
            rows.extend(chunk)
            end = int(chunk[-1][0]) - 1
        return to_frame(rows)

    def _download_tail(self, since):
        """Свечи от since (open time) до текущей одним запросом; разрыв не больше страницы."""
        need = int((current_hour() - since) / pd.Timedelta(hours=1)) + 1
        chunk = self._request(min(PAGE, need), start=int(since.timestamp() * 1000))
        return to_frame(chunk) if chunk else pd.DataFrame(columns=COLS)

    # ── основной метод ────────────────────────────────────────────────────────────────────
    def get(self, n):
        """Последние n закрытых часовых свечей (date, open, high, low, close, volume)."""
        self.last = {"requests": 0, "from_cache": 0, "downloaded": 0}
        now_hour = current_hour()
        cached = self.bars
        fresh_enough = (len(cached) >= n and
                        (now_hour - cached["date"].iloc[-OVERLAP]) / pd.Timedelta(hours=1) < PAGE)
        if fresh_enough:
            since = cached["date"].iloc[-OVERLAP]
            tail = self._download_tail(since)
            tail = tail[tail["date"] < now_hour]
            overlap = cached[cached["date"] >= since].set_index("date")
            check = tail.set_index("date").reindex(overlap.index)
            same = (len(check.dropna()) == len(overlap) and
                    ((check[["open", "high", "low", "close"]] - overlap[["open", "high", "low", "close"]])
                     .abs().to_numpy().max() < 1e-9))
            if same:
                new = tail[tail["date"] > cached["date"].iloc[-1]]
                merged = pd.concat([cached, new], ignore_index=True)
                if self._continuous(merged):
                    self.last.update(from_cache=len(cached), downloaded=len(new))
                    self._commit(merged)
                    return self.bars.tail(n).reset_index(drop=True)
            log.warning("Кеш свечей не сошёлся с биржей (перекрытие или разрыв) — история скачивается заново")

        full = self._download_full(max(n, self.keep))
        full = full[full["date"] < now_hour].reset_index(drop=True)
        self.last.update(from_cache=0, downloaded=len(full))
        self._commit(full)
        return self.bars.tail(n).reset_index(drop=True)

    def _commit(self, df):
        self.bars = df.tail(self.keep).reset_index(drop=True)
        try:
            self._save()
        except OSError as exc:
            log.warning("Кеш свечей не записан на диск: %s", exc)
