"""Ограничения риска: защита капитала, если модель рекомендует неоптимальные действия.

Риск-менеджер стоит между рекомендацией модели и исполнением. Он не меняет стратегию, а только
запрещает или прерывает торговлю, когда срабатывает одно из ограничений:

  уровень «стоп торговли» (HALT) — позиция закрывается, торговля останавливается до ручного
  возобновления (удалить файл HALT или `python bot.py --resume`):
    - просадка equity от пика больше MAX_DRAWDOWN_PCT;
    - equity ниже абсолютного минимума MIN_EQUITY_USDT;
    - подряд MAX_CONSECUTIVE_LOSSES убыточных сделок;
    - стоп-лосс на позицию не удалось поставить;
    - ручной рубильник: файл HALT (`python bot.py --halt`);

  уровень «пауза до конца суток UTC» — позиция закрывается, новые сделки запрещены до 00:00 UTC:
    - убыток с начала суток больше DAILY_LOSS_PCT;

  ограничения на отдельное действие — сделка не выполняется, позиция остаётся как есть:
    - сделок за сутки больше MAX_TRADES_PER_DAY;
    - позиция больше MAX_POSITION_USDT (объём урезается до лимита);
    - данные устарели больше чем на MAX_DATA_DELAY_MIN минут;
    - цена закрытия свечи отличается от текущей больше чем на MAX_PRICE_DEVIATION_PCT.

Плечо ограничено сверху MAX_LEVERAGE.
"""
import logging
import os
from dataclasses import dataclass, fields
from datetime import datetime, timezone

log = logging.getLogger("risk")


@dataclass
class RiskConfig:
    max_drawdown_pct: float = 20.0
    daily_loss_pct: float = 5.0
    min_equity_usdt: float = 80.0
    max_consecutive_losses: int = 3
    max_trades_per_day: int = 4
    max_position_usdt: float = 150.0
    max_leverage: float = 3.0
    max_data_delay_min: float = 90.0
    max_price_deviation_pct: float = 2.0
    risk_check_sec: int = 60

    @classmethod
    def from_env(cls):
        kw = {}
        for f in fields(cls):
            v = os.environ.get(f.name.upper())
            if v not in (None, ""):
                kw[f.name] = type(f.default)(v)
        return cls(**kw)


def today():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


class RiskManager:
    def __init__(self, cfg, state, halt_path, write_files=True):
        """state — словарь risk внутри state.json бота (изменяется на месте).
        write_files=False (dry-run) — не создавать файл HALT, чтобы не остановить боевой бот."""
        self.cfg, self.st, self.halt_path, self.write_files = cfg, state, halt_path, write_files
        self.st.setdefault("peak_equity", None)
        self.st.setdefault("day", None)
        self.st.setdefault("day_start_equity", None)
        self.st.setdefault("trades_today", 0)
        self.st.setdefault("consecutive_losses", 0)
        self.st.setdefault("halted", None)          # причина остановки или None
        self.st.setdefault("paused_day", None)      # дата UTC, до конца которой пауза

    # ── рубильник ─────────────────────────────────────────────────────────────────────────
    def halt(self, reason):
        if not self.st["halted"]:
            log.error("СТОП ТОРГОВЛИ: %s. Позиция закрывается; возобновление — удалить %s или "
                      "`python bot.py --resume`", reason, self.halt_path)
        self.st["halted"] = reason
        if self.write_files and not os.path.exists(self.halt_path):
            with open(self.halt_path, "w") as fh:
                fh.write(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {reason}\n")

    def sync_halt_file(self, equity):
        """Файл HALT — общий рубильник: создан вручную -> стоп; удалён -> возобновление."""
        if os.path.exists(self.halt_path):
            if not self.st["halted"]:
                with open(self.halt_path) as fh:
                    note = fh.read().strip()
                self.halt(f"ручная остановка (файл HALT{': ' + note if note else ''})")
        elif self.st["halted"]:
            log.warning("Файл HALT удалён — торговля возобновлена (было: %s). Пик equity и "
                        "счётчики убытков сброшены к текущему equity %.2f", self.st["halted"], equity)
            self.st.update(halted=None, peak_equity=equity, consecutive_losses=0)

    # ── проверки по equity ────────────────────────────────────────────────────────────────
    def check_equity(self, equity):
        """Обновить пик и суточную базу; вернуть 'halt', 'pause' или None."""
        self.sync_halt_file(equity)
        st, cfg = self.st, self.cfg
        if st["day"] != today():
            st.update(day=today(), day_start_equity=equity, trades_today=0)
            if st["paused_day"] and st["paused_day"] != today():
                log.info("Новые сутки UTC — суточная пауза снята")
                st["paused_day"] = None
        if st["peak_equity"] is None or equity > st["peak_equity"]:
            st["peak_equity"] = equity

        dd = (1 - equity / st["peak_equity"]) * 100 if st["peak_equity"] else 0.0
        day_loss = (1 - equity / st["day_start_equity"]) * 100 if st["day_start_equity"] else 0.0
        if dd >= cfg.max_drawdown_pct:
            self.halt(f"просадка {dd:.1f}% от пика {st['peak_equity']:.2f} USDT "
                      f"(лимит {cfg.max_drawdown_pct}%)")
        elif equity < cfg.min_equity_usdt:
            self.halt(f"equity {equity:.2f} USDT ниже минимума {cfg.min_equity_usdt}")
        elif day_loss >= cfg.daily_loss_pct and st["paused_day"] != today():
            st["paused_day"] = today()
            log.error("ПАУЗА ДО КОНЦА СУТОК: убыток за сутки %.1f%% (лимит %s%%) — позиция "
                      "закрывается", day_loss, cfg.daily_loss_pct)
        if st["halted"]:
            return "halt"
        if st["paused_day"] == today():
            return "pause"
        return None

    def metrics(self, equity):
        st = self.st
        dd = (1 - equity / st["peak_equity"]) * 100 if st.get("peak_equity") else 0.0
        day = (equity / st["day_start_equity"] - 1) * 100 if st.get("day_start_equity") else 0.0
        return {"drawdown_pct": round(dd, 2), "day_pnl_pct": round(day, 2),
                "trades_today": st["trades_today"], "consecutive_losses": st["consecutive_losses"],
                "halted": st["halted"] or "", "paused": int(st["paused_day"] == today())}

    # ── проверки отдельного действия ──────────────────────────────────────────────────────
    def data_ok(self, bar_open_time, bar_close, last_price):
        delay = (datetime.now(timezone.utc).replace(tzinfo=None) - bar_open_time.to_pydatetime()
                 ).total_seconds() / 60 - 60          # свеча закрывается через час после открытия
        if delay > self.cfg.max_data_delay_min:
            return False, f"данные устарели: последняя свеча закрылась {delay:.0f} мин назад"
        dev = abs(last_price / bar_close - 1) * 100
        if dev > self.cfg.max_price_deviation_pct:
            return False, (f"цена свечи {bar_close:.1f} и текущая {last_price:.1f} расходятся на "
                           f"{dev:.1f}% (лимит {self.cfg.max_price_deviation_pct}%)")
        return True, ""

    def can_open(self):
        if self.st["halted"]:
            return False, f"торговля остановлена: {self.st['halted']}"
        if self.st["paused_day"] == today():
            return False, "суточная пауза после лимита убытка"
        if self.st["trades_today"] >= self.cfg.max_trades_per_day:
            return False, f"лимит сделок за сутки ({self.cfg.max_trades_per_day}) исчерпан"
        return True, ""

    def cap_notional(self, notional):
        return min(notional, self.cfg.max_position_usdt)

    def on_open(self):
        self.st["trades_today"] += 1

    def on_close(self, realized_pnl):
        if realized_pnl is None or realized_pnl != realized_pnl:     # NaN — PnL неизвестен
            return
        if realized_pnl < 0:
            self.st["consecutive_losses"] += 1
            if self.st["consecutive_losses"] >= self.cfg.max_consecutive_losses:
                self.halt(f"{self.st['consecutive_losses']} убыточных сделок подряд "
                          f"(лимит {self.cfg.max_consecutive_losses})")
        else:
            self.st["consecutive_losses"] = 0
