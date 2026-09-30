# High-Precision Forex Technical Scalper

A lightweight, sub-second deterministic Forex scalping bot engineered for **OANDA v20 REST API**. Optimized specifically for a **$1,000 paper trading account**, targeting small, continuous compounding wins with strict risk management, spread gating, and homelab K3s deployment readiness.

---

## Key Features

- **Pure Technical Confluence Strategy:**
  - **Higher Timeframe Trend Filter (M15 / M5):** Exponential Moving Averages (EMA 9 vs. EMA 21) define macro market regime.
  - **Lower Timeframe Trigger (M1):** RSI (14) overbought/oversold pullbacks combined with Bollinger Band (20, 2) boundary rejections and candlestick pattern detection (Pinbar / Engulfing).
  - **Dynamic Volatility Brackets:** ATR-driven Stop Loss (5.0–8.0 pips) and Take Profit (5.0–8.0 pips) for fast in-and-out executions.
- **Strict Real-Time Spread Gate:**
  - Automatically measures the real-time broker bid-ask spread before every order.
  - Rejects entries if `spread > 1.2 pips` (majors) or `> 1.8 pips` (crosses) to prevent transaction friction from eating profits.
- **Micro-Lot Risk Management ($1,000 Account Profile):**
  - Standard size: **10,000 units (0.10 lot / $1.00 per pip)**.
  - Conservative size: Drops to **5,000 units ($0.50 per pip)** if available margin tightens.
  - Single Position Cap: `MAX_CONCURRENT_TRADES = 1` eliminates cross-pair correlation risk and margin exhaustion.
- **Session & Rollover Blackout Filter:**
  - Active during high-liquidity volume windows: **London & New York Overlap (07:00 – 17:00 UTC / 3:00 AM – 1:00 PM EST)**.
  - **Daily Rollover Blackout (21:00 – 22:15 UTC / 5:00 PM – 6:15 PM EST):** Complete trading blackout when interbank spreads artificially widen.
- **Daily Circuit Breakers:**
  - **Daily Profit Target:** Locks in gains at **+$25.00/day** (2.5% daily return) and pauses trading until the next calendar day.
  - **Daily Loss Circuit Breaker:** Hard stop at **-$25.00/day** to preserve capital and prevent revenge trading.
- **State Persistence (SQLite):**
  - All trades, entry/exit prices, spreads paid, and daily PnL are recorded in `trades.db`, surviving container restarts.
- **Embedded REST API for AI Agents:**
  - Blazing-fast FastAPI server running on port 8000 for AI agents (e.g. Jarvis) and dashboards to query balance, active trades, and daily PnL without reading log files.

---

## AI Agent & Monitoring API

The bot includes an embedded FastAPI server exposing real-time endpoints for AI agents and external dashboards (with CORS configured for Tailscale `100.*.*.*` networks):

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/api/status` | GET | Current bot state (`ACTIVE`, `PAUSED_PROFIT_TARGET`, `HALTED_LOSS_LIMIT`), circuit breaker limits, and today's PnL. |
| `/api/account` | GET | Real-time account balance, available margin, and currency from OANDA. |
| `/api/trades/active` | GET | Currently open trades on OANDA with live unrealized PnL. |
| `/api/trades/recent?limit=20` | GET | Detailed history of recent trades from the SQLite ledger (entry/exit prices, realized PnL, spread paid, timestamps). |
| `/api/pnl/daily` | GET | Today's aggregate PnL, total trades, win rate %, and win/loss counts. |
| `/api/quote/{pair}` | GET | Live bid, ask, mid, and real-time spread in pips for any currency pair. |
| `/docs` | GET | Interactive Swagger UI documentation. |

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
```

### 2. Run Tests
The test suite validates indicator math, spread gates, risk sizing, and SQLite trade logging:
```bash
uv run pytest
```

### 3. Run the Bot Locally
```bash
uv run python main.py
```

---

## Homelab K3s Deployment (Longhorn / Tailscale)

The bot is containerized for deployment on a multi-node K3s cluster.

### 1. Deploy Script (AMD64 Cross-Compilation)
```bash
./deploy.sh
```
This automates:
1. Cross-compiling the image for x86_64 nodes: `docker build --platform linux/amd64 -t jcooper23/forex-trading-bot:latest .`
2. Pushing the image to Docker Hub: `docker push jcooper23/forex-trading-bot:latest`

### 2. Configure Portainer Secret
Create a Kubernetes secret named `forex-bot-secrets` containing:
- `OANDA_ACCESS_TOKEN`
- `OANDA_ACCOUNT_ID`
- `OANDA_ENV`
- `DAILY_PROFIT_TARGET`
- `MAX_DAILY_LOSS`

### 3. Apply Kubernetes Manifests
```bash
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/deployment.yaml
```
- **Storage:** Persists `trades.db` and logs on Longhorn distributed block storage via `forex-trading-bot-data` PVC.
- **Pull Policy:** `imagePullPolicy: Always` ensures Portainer rolling restarts pick up new builds.
