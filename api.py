import os
import asyncio
from datetime import datetime, timezone
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from connection import OandaConnection
from trade_ledger import TradeLedger
from redis_client import RedisClient

def create_app(connection=None, ledger=None, redis_client=None):
    app = FastAPI(
        title="Forex Scalper Monitoring API",
        version="0.3.0",
        description="Fast JSON API for AI agents and dashboards with PostgreSQL storage and Redis Pub/Sub event streaming."
    )

    # Rule 4 Homelab Standard: Allow CORS from Tailscale network (100.*.*.*) and local/internal IPs
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_origin_regex=r"https?://(100\.\d{1,3}\.\d{1,3}\.\d{1,3}|localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    conn = connection or OandaConnection()
    trade_ledger = ledger or TradeLedger()
    r_client = redis_client or RedisClient()

    @app.get("/")
    def root():
        return {
            "app": "Forex Scalper Monitoring API",
            "version": "0.4.0",
            "status": "ONLINE",
            "database": "PostgreSQL" if trade_ledger.is_postgres else "SQLite",
            "redis_connected": r_client.is_connected(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/health")
    def health_check():
        """
        Health check endpoint for K8s readiness probes.
        Returns 200 if both OANDA API and database are reachable, 503 otherwise.
        """
        issues = []

        # Check OANDA connectivity
        try:
            balance, margin = conn.get_account_details()
            if balance == 0.0 and margin == 0.0:
                issues.append("OANDA returned zero balance/margin (possible auth failure)")
        except Exception as e:
            issues.append(f"OANDA unreachable: {str(e)[:100]}")

        # Check database connectivity
        try:
            with trade_ledger._get_connection() as db_conn:
                cursor = db_conn.cursor()
                cursor.execute("SELECT 1")
        except Exception as e:
            issues.append(f"Database unreachable: {str(e)[:100]}")

        if issues:
            raise HTTPException(status_code=503, detail={
                "status": "degraded",
                "issues": issues,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })

        return {
            "status": "healthy",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/api/status")
    def get_status():
        """Get overall bot status, circuit breakers, and daily PnL summary."""
        daily_stats = trade_ledger.get_daily_pnl()
        daily_target = float(os.getenv("DAILY_PROFIT_TARGET", "25.0"))
        max_loss = float(os.getenv("MAX_DAILY_LOSS", "25.0"))
        
        today_pl = daily_stats["realized_pl"]
        profit_target_hit = today_pl >= daily_target
        loss_circuit_hit = today_pl <= -max_loss

        state = "PAUSED_PROFIT_TARGET" if profit_target_hit else ("HALTED_LOSS_LIMIT" if loss_circuit_hit else "ACTIVE")

        return {
            "bot_state": state,
            "today_pnl": today_pl,
            "daily_target": daily_target,
            "max_daily_loss": max_loss,
            "stats": daily_stats,
            "database": "PostgreSQL" if trade_ledger.is_postgres else "SQLite",
            "redis": r_client.is_connected(),
            "timestamp": datetime.now(timezone.utc).isoformat()
        }

    @app.get("/api/account")
    def get_account():
        """
        Fetch real-time account balance and available margin.
        Caches in Redis for 5 seconds to shield OANDA from rate-limiting.
        """
        CACHE_KEY = "oanda:account_summary"
        cached = r_client.cache_get(CACHE_KEY)
        if cached:
            return {**cached, "cached": True}

        try:
            balance, margin_avail = conn.get_account_details()
            data = {
                "balance": balance,
                "margin_available": margin_avail,
                "environment": conn.env,
                "currency": "USD",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            r_client.cache_set(CACHE_KEY, data, ttl_seconds=5)
            return {**data, "cached": False}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/trades/active")
    def get_active_trades():
        """Fetch all currently open trades on OANDA."""
        try:
            open_trades = conn.get_open_trades()
            formatted = []
            for t in open_trades:
                units = int(t.get('currentUnits', 0))
                formatted.append({
                    "trade_id": t.get("id"),
                    "instrument": t.get("instrument"),
                    "direction": "BUY" if units > 0 else "SELL",
                    "units": units,
                    "entry_price": float(t.get("price", 0.0)),
                    "unrealized_pl": float(t.get("unrealizedPL", 0.0)),
                    "open_time": t.get("openTime")
                })
            return {
                "active_count": len(formatted),
                "trades": formatted
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/trades/recent")
    def get_recent_trades(limit: int = Query(20, ge=1, le=100)):
        """Fetch recent trades recorded in the database (Postgres or SQLite)."""
        trades = trade_ledger.get_recent_trades(limit=limit)
        return {
            "total_returned": len(trades),
            "trades": trades
        }

    @app.get("/api/pnl/daily")
    def get_daily_pnl(date: str = Query(None, description="UTC Date in YYYY-MM-DD")):
        """Fetch daily PnL, trade volume, and win/loss count."""
        summary = trade_ledger.get_daily_pnl(target_date=date)
        total = summary["total_trades"]
        win_rate = round((summary["wins"] / total) * 100, 1) if total > 0 else 0.0
        return {
            **summary,
            "win_rate_pct": win_rate
        }

    @app.get("/api/quote/{instrument}")
    def get_quote(instrument: str):
        """Fetch live bid, ask, mid, and spread in pips for a currency pair."""
        CACHE_KEY = f"oanda:quote:{instrument.upper()}"
        cached = r_client.cache_get(CACHE_KEY)
        if cached:
            return {**cached, "cached": True}

        quote = conn.get_pricing_quote(instrument.upper())
        if not quote:
            raise HTTPException(status_code=404, detail=f"Pricing unavailable for {instrument}")
        
        data = {
            "instrument": instrument.upper(),
            **quote,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        r_client.cache_set(CACHE_KEY, data, ttl_seconds=3)
        return {**data, "cached": False}

    @app.get("/api/stream")
    async def stream_events():
        """
        Server-Sent Events (SSE) endpoint for AI agents to stream real-time trade events.
        Connect using standard HTTP streaming (curl -N http://.../api/stream).
        """
        async def event_generator():
            if not r_client.is_connected():
                yield f"data: {{\"event\": \"CONNECTED\", \"message\": \"Streaming active (Redis disabled)\"}}\n\n"
                while True:
                    await asyncio.sleep(15)
                    yield f"data: {{\"event\": \"HEARTBEAT\", \"timestamp\": \"{datetime.now(timezone.utc).isoformat()}\"}}\n\n"
                return

            pubsub = r_client.client.pubsub()
            pubsub.subscribe("forex:events", "forex:alerts")
            yield f"data: {{\"event\": \"CONNECTED\", \"message\": \"Subscribed to forex:events and forex:alerts\"}}\n\n"

            while True:
                message = pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message and message.get("type") == "message":
                    channel = message["channel"]
                    data = message["data"]
                    yield f"data: {{\"channel\": \"{channel}\", \"payload\": {data}}}\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(event_generator(), media_type="text/event-stream")

    return app

# Default app instance for uvicorn
app = create_app()
