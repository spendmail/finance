# Торговый бот BTC/USDT perpetual (Bybit): ансамбль LogReg + DecisionTree + CatBoost, SL 3% + безубыток 5%
FROM python:3.11-slim

# libgomp1 — OpenMP-рантайм для CatBoost и scikit-learn; tini — корректная передача SIGTERM боту
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 tini \
 && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    BOT_RUNTIME_DIR=/app/runtime

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# код и обученные модели; секреты (.env) в образ не попадают — см. .dockerignore
COPY bot.py bybit_client.py features.py market_cache.py risk.py strategy.py ./
COPY artifacts/ ./artifacts/

RUN useradd --create-home --uid 1000 bot \
 && mkdir -p /app/runtime && chown bot:bot /app/runtime
USER bot
VOLUME ["/app/runtime"]

# здоров, если бот отметился в heartbeat за последние 5 минут (проверки риска — каждую минуту)
HEALTHCHECK --interval=60s --timeout=10s --start-period=180s --retries=3 \
  CMD python -c "import os,sys,time; p='/app/runtime/heartbeat'; sys.exit(0 if os.path.exists(p) and time.time()-os.path.getmtime(p) < 300 else 1)"

ENTRYPOINT ["/usr/bin/tini", "--", "python", "bot.py"]
