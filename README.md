# High-Precision Forex Technical Scalper

A lightweight, sub-second deterministic Forex scalping bot engineered for the **OANDA v20 REST API**. Optimized specifically for a **$1,000 paper trading account**, targeting small, continuous compounding wins with strict risk management, dynamic spread gating, automated breakeven stops, and production-ready K3s homelab deployment.

---

## Architecture Overview

```mermaid
graph TD
    A[main.py: Scalper Engine] --> B{Session Filter & Rollover Check}
    B -- Rollover Blackout / Inactive --> C[Idle until London / NY Session]
    B -- Active Window --> D{Daily Circuit Breakers}
    D -- Hit +$25 or -$25 --> E[Halt trading for the day]
    D -- Active Trading Window --> F[Market Scanner]
    
    F --> G[Real-Time Pricing & Dynamic Spread Gate]
    G -- Spread > 30% of ATR or > 2.5p --> H[Skip Pair]
    G -- Spread Passes --> I[Technical Strategy: Confluence Engine]
    
    I --> J{ADX >= 15 & M15 Trend + M1 Pullback?}
    J -- Ranging Chop / No Setup --> K[Wait for next cycle]
    J -- Strong Confluence Setup --> L[Risk Manager: 10k Units]
    
    L --> M[OANDA API: Execute Bracket Order with Tenacity Retries]
    M --> N[Redis Pub/Sub: Broadcast TRADE_OPENED]
    M --> O[PostgreSQL / SQLite Ledger: Record Trade]
    
    A --> P[Active Trade Monitor: Trailing Breakeven]
    P -- Profit >= +5.0 pips --> Q[Move SL to Entry + 0.5 pip Buffer via OANDA API]
    
    R[External AI Agent / Dashboard] --> S[Embedded FastAPI Monitoring Server]
    S --> T[Redis Sub-Second Cache & SSE Stream: /api/stream]
    S --> O
```

---

## Key Features & Archetype

### 1. Pure Technical Confluence Strategy ([strategy.py](file:///Users/joshuacooper/forex-trading-bot/strategy.py))
- **Macro Trend Direction (M15 / M5):** Exponential Moving Averages (EMA 9 vs. EMA 21) establish high-timeframe trend alignment.
- **ADX Chop Filter (M5):** Calculates Average Directional Index (ADX); automatically forces `HOLD` if $\text{ADX} < 15$ to prevent bleeding capital in flat, non-trending markets.
- **Lower-Timeframe Pullback Trigger (M1):** RSI (14) oversold/overbought pullbacks combined with Bollinger Band (20, 2) boundary rejections.
- **Volume & Candlestick Confirmation (M1):** Volume ratio compared against 20-bar SMA; bullish/bearish pinbars and engulfing patterns provide precise triggers.
- **Dynamic Risk-to-Reward Scaling:**
  - High conviction ($\ge 0.85$ score) $\rightarrow$ **1:2.0 R:R**
  - Moderate conviction ($\ge 0.75$ score) $\rightarrow$ **1:1.75 R:R**
  - Baseline conviction ($\ge 0.65$ score) $\rightarrow$ **1:1.50 R:R**

### 2. Dynamic Spread Gating ([market_scanner.py](file:///Users/joshuacooper/forex-trading-bot/market_scanner.py))
- Volatility-adaptive gate: **spread must be $< 30\%$ of current ATR**, clamped between a strict floor (0.8 pips) and ceiling (2.5 pips).
- Discards wide-spread pairs before burning API calls on candle retrieval.
- **Contrarian Position Book Sentiment Modifier:** Evaluates OANDA crowd positioning as an edge modifier (+0.10 confidence boost if contrarian sentiment confirms, -0.05 penalty if crowded).

### 3. Automated Trailing Breakeven Stop Loss ([main.py](file:///Users/joshuacooper/forex-trading-bot/main.py))
- Active position monitoring tracks real-time unrealized pip gains.
- Once a trade reaches **+5.0 pips in profit**, the bot automatically calls OANDA's `TradeCRCDO` endpoint to move the stop loss to **entry price + 0.5 pips** (locking in breakeven and covering spread costs).

### 4. Micro-Lot Risk Management for $1,000 Capital ([risk_manager.py](file:///Users/joshuacooper/forex-trading-bot/risk_manager.py))
- Standard size: **10,000 units (0.10 lot / $1.00 per pip)**.
- Conservative size: Drops to **5,000 units ($0.50 per pip)** if available margin tightens below $350.
- Strict single position guarantee: `MAX_CONCURRENT_TRADES = 1`.

### 5. Resilient Connection Layer with `tenacity` Retries ([connection.py](file:///Users/joshuacooper/forex-trading-bot/connection.py))
- Idempotent read calls (`get_candles`, `get_account_details`, `get_pricing_quote`, `get_trade_details`) use exponential backoff retries (1–10s) on transient network issues and rate limits (429/503).
- Non-idempotent order executions (`create_order`, `close_trade`) are strictly not retried to prevent double-fills.

### 6. Decoupled Persistence & Real-Time Caching
- **Dual-Engine Trade Ledger ([trade_ledger.py](file:///Users/joshuacooper/forex-trading-bot/trade_ledger.py)):**
  - **PostgreSQL in Production:** Uses a pooled connection pool (`SimpleConnectionPool`, 2–10 connections) for concurrent reads/writes from multiple cluster services.
  - **SQLite Local Fallback:** Uses thread locking (`threading.Lock()`) for zero-setup local testing.
- **Redis Pub/Sub & Caching ([redis_client.py](file:///Users/joshuacooper/forex-trading-bot/redis_client.py)):**
  - Broadcasts instant events to `forex:events` and `forex:alerts`.
  - Caches account balances and quotes with sub-second TTL to shield OANDA from rate-limiting.

---

## AI Agent & Monitoring API

The embedded FastAPI server runs concurrently on port **`8000`** with CORS enabled for Tailscale (`100.*.*.*`):

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| **`/health`** | GET | Kubernetes readiness probe checking OANDA and database connectivity. |
| **`/api/status`** | GET | Current bot state (`ACTIVE`, `PAUSED_PROFIT_TARGET`, `HALTED_LOSS_LIMIT`), circuit breaker limits, and today's PnL. |
| **`/api/account`** | GET | Real-time account balance and available margin (cached in Redis). |
| **`/api/trades/active`** | GET | Currently open positions on OANDA with live unrealized PnL. |
| **`/api/trades/recent?limit=20`** | GET | Detailed trade history from the database (entry/exit prices, realized PnL, spread, timestamps). |
| **`/api/pnl/daily`** | GET | Daily aggregate summary (today's PnL, total trades, win rate %, win/loss counts). |
| **`/api/quote/{pair}`** | GET | Real-time bid, ask, mid, and live spread in pips for any currency pair. |
| **`/api/stream`** | GET | **Server-Sent Events (SSE)** stream pushing real-time trade events to your AI agent without polling. |
| **`/docs`** | GET | Interactive Swagger UI documentation. |

---

## Currency Pairs

- `EUR_USD` (Tightest spread baseline)
- `GBP_USD`
- `USD_JPY`
- `USD_CAD`
- `AUD_USD`

---

## Local Development & Testing

### 1. Environment Setup
Create a `.env` file:
```env
OANDA_ACCESS_TOKEN=your_oanda_v20_token
OANDA_ACCOUNT_ID=your_oanda_practice_account_id
OANDA_ENV=practice
DAILY_PROFIT_TARGET=25.0
MAX_DAILY_LOSS=25.0
IGNORE_SESSION_FILTER=false
DATABASE_URL=postgresql://forex:password@localhost:5432/forex  # Optional, defaults to SQLite
REDIS_URL=redis://localhost:6379/0                           # Optional, defaults to mock
```

### 2. Run Tests
The test suite validates indicators, ADX gating, dynamic spread limits, risk sizing, trade ledger, Redis client, and API endpoints:
```bash
uv run pytest
```

### 3. Run Locally
```bash
# Standalone with SQLite
uv run python main.py

# Multi-service stack (Bot + Postgres + Redis)
docker compose up -d
```

---

## Homelab K3s Deployment (Longhorn / Tailscale)

The application is fully decoupled across separate Kubernetes deployments adhering strictly to homelab standards:

1. **[k8s/postgres.yaml](file:///Users/joshuacooper/forex-trading-bot/k8s/postgres.yaml):** Dedicated PostgreSQL 16 container with 5Gi Longhorn storage and `PGDATA=/var/lib/postgresql/data/pgdata` subfolder quirk fix.
2. **[k8s/redis.yaml](file:///Users/joshuacooper/forex-trading-bot/k8s/redis.yaml):** Dedicated Redis 7 container with 2Gi Longhorn storage and AOF persistence.
3. **[k8s/pvc.yaml](file:///Users/joshuacooper/forex-trading-bot/k8s/pvc.yaml):** Longhorn persistent volume claim for bot data.
4. **[k8s/deployment.yaml](file:///Users/joshuacooper/forex-trading-bot/k8s/deployment.yaml):** Trading bot container with liveness probe (`/`), readiness probe (`/health`), and secrets injected from `forex-bot-secrets`.
5. **[k8s/service.yaml](file:///Users/joshuacooper/forex-trading-bot/k8s/service.yaml):** NodePort service exposing the API to the Tailscale mesh.

### Automated Cross-Compilation & Deployment
```bash
./deploy.sh
```
This script:
1. Cross-compiles for AMD64 nodes: `docker build --platform linux/amd64 -t jcooper23/forex-trading-bot:latest .`
2. Pushes the image to Docker Hub (`jcooper23/forex-trading-bot:latest`).
3. Outputs clear rollout instructions for Portainer.
