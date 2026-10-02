import time
import logging
import sys
import os
import signal
import threading
import warnings
import uvicorn
from datetime import datetime, timezone
from dotenv import load_dotenv

from connection import OandaConnection
from strategy import TechnicalScalpStrategy
from risk_manager import RiskManager
from market_scanner import MarketScanner
from trade_ledger import TradeLedger
from redis_client import RedisClient
from utils import pip_unit as get_pip_unit
from api import create_app

def start_api_server(app_instance, host="0.0.0.0", port=8000):
    config = uvicorn.Config(app_instance, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

load_dotenv()

# Configuration (all env-configurable for K8s/Portainer without rebuilds)
_instruments_raw = os.getenv("INSTRUMENTS", "EUR_USD,GBP_USD,USD_JPY,USD_CAD,AUD_USD")
INSTRUMENTS = [i.strip() for i in _instruments_raw.split(",") if i.strip()]
CYCLE_SLEEP_SECONDS = int(os.getenv("CYCLE_SLEEP_SECONDS", "60"))
MAX_CONCURRENT_TRADES = int(os.getenv("MAX_CONCURRENT_TRADES", "1"))

# Daily Circuit Breakers for $1,000 Paper Account
DAILY_PROFIT_TARGET = float(os.getenv("DAILY_PROFIT_TARGET", "25.0"))  # Lock profit at +$25.00
MAX_DAILY_LOSS = float(os.getenv("MAX_DAILY_LOSS", "25.0"))           # Stop trading if loss hits -$25.00
IGNORE_SESSION_FILTER = os.getenv("IGNORE_SESSION_FILTER", "false").lower() == "true"

# Suppress library warnings
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")
logging.getLogger("oandapyV20").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Logging: stdout only (K8s captures via kubectl logs, trade history lives in Postgres)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("trading_bot")

def is_trading_session_active():
    """
    Check if current UTC time is within an optimal trading session and avoid rollover.
    - Rollover Blackout: 21:00 UTC to 22:15 UTC (5:00 PM - 6:15 PM EST)
    - Optimal Scalp Window: 07:00 UTC to 17:00 UTC (London & NY Overlap)
    """
    if IGNORE_SESSION_FILTER:
        return True, "Session filter bypassed by config"

    now = datetime.now(timezone.utc)
    hour = now.hour
    minute = now.minute

    # 1. Rollover Blackout: 21:00 to 22:15 UTC
    if (hour == 21) or (hour == 22 and minute <= 15):
        return False, "Daily Rollover Window (Spreads Widened). Trading paused."

    # 2. Optimal Window: 07:00 to 17:00 UTC
    if 7 <= hour < 17:
        return True, "London / NY Session Active"

    # Outside peak, still allow if spread gate passes, but inform
    return True, "Off-peak Session (Strict spread gating enforced)"

def sync_closed_trades(conn, ledger, redis_client=None, risk_manager=None):
    """
    Check if any trades marked as OPEN in our local ledger have been closed by OANDA
    (e.g., hit Stop Loss or Take Profit). If so, update their status, real P/L, and exit price.
    Also notifies risk_manager so post-loss cooldowns and circuit breakers update.
    """
    try:
        open_trades_oanda = conn.get_open_trades()
        active_trade_ids = {str(t['id']) for t in open_trades_oanda}

        # Query our ledger for open trades
        with ledger._get_connection() as db_conn:
            cursor = db_conn.cursor()
            cursor.execute("SELECT trade_id, instrument, direction, entry_price FROM trades WHERE status = 'OPEN'")
            ledger_open_trades = cursor.fetchall()

        for t_id, inst, direction, entry_p in ledger_open_trades:
            if t_id not in active_trade_ids:
                # Trade was closed on OANDA. Fetch real P/L from trade details.
                trade_details = conn.get_trade_details(t_id)

                if trade_details:
                    realized_pl = trade_details["realizedPL"]
                    exit_price = trade_details["averageClosePrice"] or entry_p
                else:
                    # Fallback if trade details fetch fails (e.g., trade purged from history)
                    logger.warning(f"Sync: Could not fetch details for trade {t_id}. Recording with estimated values.")
                    quote = conn.get_pricing_quote(inst)
                    exit_price = quote["mid"] if quote else entry_p
                    realized_pl = 0.0

                ledger.record_close(trade_id=t_id, exit_price=exit_price, realized_pl=realized_pl)
                logger.info(f"Sync: Trade {t_id} on {inst} ({direction}) closed by OANDA | P/L: ${realized_pl:.2f} | Exit: {exit_price}")

                if risk_manager:
                    risk_manager.record_trade_result(inst, direction, realized_pl)

                if redis_client:
                    redis_client.publish_event("forex:events", {
                        "event": "TRADE_CLOSED",
                        "trade_id": t_id,
                        "instrument": inst,
                        "direction": direction,
                        "exit_price": exit_price,
                        "realized_pl": realized_pl
                    })
    except Exception as e:
        logger.error(f"Error syncing closed trades: {e}")

def shutdown_bot(conn, ledger=None, redis_client=None, scanner=None):
    """Graceful shutdown handler. Cleans up all resources."""
    logger.info("Initiating graceful shutdown...")
    try:
        open_trades = conn.get_open_trades()
        if open_trades:
            logger.info(f"Closing {len(open_trades)} open trade(s) before exit...")
            for t in open_trades:
                t_id = t['id']
                inst = t['instrument']
                pl = float(t.get('unrealizedPL', 0.0))
                logger.info(f"Closing trade {t_id} ({inst}) | P/L: ${pl:.2f}")
                conn.close_trade(t_id)
                if ledger:
                    quote = conn.get_pricing_quote(inst)
                    exit_p = quote["mid"] if quote else 0.0
                    ledger.record_close(t_id, exit_price=exit_p, realized_pl=pl)
                if redis_client:
                    redis_client.publish_event("forex:events", {
                        "event": "TRADE_CLOSED_SHUTDOWN",
                        "trade_id": t_id,
                        "instrument": inst,
                        "realized_pl": pl
                    })
    except Exception as e:
        logger.error(f"Error closing trades during shutdown: {e}")

    # Clean up resources
    if scanner:
        scanner.shutdown()
    if ledger and hasattr(ledger, '_pool') and ledger._pool:
        try:
            ledger._pool.closeall()
            logger.info("TradeLedger: Connection pool closed.")
        except Exception:
            pass
    if ledger and hasattr(ledger, '_sqlite_conn'):
        try:
            ledger._sqlite_conn.close()
            logger.info("TradeLedger: SQLite connection closed.")
        except Exception:
            pass
    logger.info("Shutdown complete.")

def main():
    logger.info("====================================================")
    logger.info("Starting High-Precision Forex Scalper ($1,000 Target)")
    logger.info(f"Instruments: {', '.join(INSTRUMENTS)}")
    logger.info(f"Max Concurrent Positions: {MAX_CONCURRENT_TRADES}")
    logger.info(f"Daily Targets: +${DAILY_PROFIT_TARGET:.2f} Profit / -${MAX_DAILY_LOSS:.2f} Max Loss")
    logger.info("====================================================")

    conn = OandaConnection()
    strategy = TechnicalScalpStrategy()
    risk_manager = RiskManager()
    scanner = MarketScanner(INSTRUMENTS)
    ledger = TradeLedger()
    redis_client = RedisClient()

    # Track trailing stop state per trade (trade_id -> last SL price set)
    trailing_stops: dict[str, float] = {}

    # Signal handlers for clean shutdown
    def handle_signal(sig, frame):
        logger.info(f"Received exit signal ({sig}).")
        shutdown_bot(conn, ledger, redis_client, scanner)
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    try:
        balance, margin_avail = conn.get_account_details()
        logger.info(f"Connected to OANDA. Initial Balance: ${balance:.2f} | Free Margin: ${margin_avail:.2f}")

        # Start Monitoring API in background thread with its own OANDA connection
        api_port = int(os.getenv("API_PORT", "8000"))
        api_conn = OandaConnection()  # Separate connection for thread safety
        api_app = create_app(connection=api_conn, ledger=ledger, redis_client=redis_client)
        api_thread = threading.Thread(
            target=start_api_server,
            args=(api_app, "0.0.0.0", api_port),
            daemon=True
        )
        api_thread.start()
        logger.info(f"Monitoring API active at http://0.0.0.0:{api_port} (Swagger docs: http://0.0.0.0:{api_port}/docs)")
    except Exception as e:
        logger.error(f"Failed initial startup sequence: {e}")
        return

    while True:
        try:
            # 1. Daily Circuit Breaker Check
            daily_stats = ledger.get_daily_pnl()
            today_pl = daily_stats["realized_pl"]
            total_trades_today = daily_stats["total_trades"]
            
            logger.info(f"--- Cycle Update | Today's PnL: ${today_pl:.2f} ({daily_stats['wins']}W / {daily_stats['losses']}L on {total_trades_today} trades) ---")

            if today_pl >= DAILY_PROFIT_TARGET:
                logger.info(f"🏆 Daily Target Reached! (${today_pl:.2f} >= ${DAILY_PROFIT_TARGET:.2f}). Pausing trading to lock in gains.")
                redis_client.publish_event("forex:alerts", {
                    "event": "PROFIT_TARGET_HIT",
                    "today_pnl": today_pl,
                    "target": DAILY_PROFIT_TARGET
                })
                time.sleep(CYCLE_SLEEP_SECONDS * 4)
                continue

            if today_pl <= -MAX_DAILY_LOSS:
                logger.warning(f"🛑 Daily Loss Circuit Breaker Triggered (-${abs(today_pl):.2f} <= -${MAX_DAILY_LOSS:.2f}). Halting to protect capital.")
                redis_client.publish_event("forex:alerts", {
                    "event": "LOSS_LIMIT_HIT",
                    "today_pnl": today_pl,
                    "max_loss": MAX_DAILY_LOSS
                })
                time.sleep(CYCLE_SLEEP_SECONDS * 4)
                continue

            # 2. Session & Rollover Window Check
            session_active, session_msg = is_trading_session_active()
            if not session_active:
                logger.info(f"Session Status: {session_msg}")
                time.sleep(CYCLE_SLEEP_SECONDS * 2)
                continue

            # 3. Check Open Trades & Sync
            open_trades = conn.get_open_trades()
            active_count = len(open_trades)
            sync_closed_trades(conn, ledger, redis_client, risk_manager=risk_manager)

            active_instruments = {t['instrument'] for t in open_trades}

            for t in open_trades:
                inst = t['instrument']
                t_id = t['id']
                pl = float(t.get('unrealizedPL', 0.0))
                units = int(t.get('currentUnits', 0))
                entry_price = float(t.get('price', 0.0))
                direction = "BUY" if units > 0 else "SELL"
                logger.info(f" >> ACTIVE POSITION: {inst} ({direction} {units}u) | Unrealized P/L: ${pl:.2f}")

                # Trailing & Breakeven Stop Logic:
                # - When profit reaches 50% of Take-Profit distance, move SL to breakeven + 0.5 pip buffer (Risk-Free).
                # - When profit reaches +5.0 pips, continue trailing every 3 pips (+5, +8, +11, ...)
                if entry_price > 0 and pl > 0:
                    pu = get_pip_unit(inst)
                    tp_order = t.get('takeProfitOrder', {})
                    tp_price = float(tp_order.get('price', 0.0)) if tp_order else 0.0
                    tp_pips = abs(tp_price - entry_price) / pu if tp_price > 0 else 7.0
                    be_threshold_pips = max(3.5, tp_pips * 0.50)

                    current_quote = conn.get_pricing_quote(inst)
                    if current_quote:
                        current_mid = current_quote["mid"]
                        if direction == "BUY":
                            profit_pips = (current_mid - entry_price) / pu
                        else:
                            profit_pips = (entry_price - current_mid) / pu

                        be_buffer = 0.5 * pu
                        be_sl = (entry_price + be_buffer) if direction == "BUY" else (entry_price - be_buffer)

                        new_sl = None
                        status_label = ""
                        locked_pips = 0.0

                        if profit_pips >= 5.0:
                            locked_pips = 5.0 + (int((profit_pips - 5.0) / 3.0) * 3.0)
                            trail_distance = locked_pips * pu
                            if direction == "BUY":
                                new_sl = entry_price + be_buffer + (trail_distance - 5.0 * pu)
                            else:
                                new_sl = entry_price - be_buffer - (trail_distance - 5.0 * pu)
                            status_label = f"Trailing SL: Locked {locked_pips:.0f}p of {profit_pips:.1f}p profit"
                        elif profit_pips >= be_threshold_pips:
                            new_sl = be_sl
                            locked_pips = 0.5
                            status_label = f"Breakeven SL: 50% TP reached ({profit_pips:.1f}p / {tp_pips:.1f}p) -> SL to BE+0.5p (Risk-Free)"

                        if new_sl is not None:
                            last_sl = trailing_stops.get(t_id)
                            sl_improved = (
                                last_sl is None
                                or (direction == "BUY" and new_sl > last_sl)
                                or (direction == "SELL" and new_sl < last_sl)
                            )
                            if sl_improved:
                                result = conn.modify_trade_sl(t_id, new_sl, inst)
                                if result:
                                    trailing_stops[t_id] = new_sl
                                    logger.info(f"{status_label} | Trade {t_id} ({inst}) SL -> {new_sl:.5f}")

            # Clean up tracking for trades that are no longer open
            open_ids = {t['id'] for t in open_trades}
            for closed_id in list(trailing_stops.keys()):
                if closed_id not in open_ids:
                    del trailing_stops[closed_id]

            # 4. Strict Single Position Enforcement
            if active_count >= MAX_CONCURRENT_TRADES:
                logger.info(f"Max active trades ({active_count}/{MAX_CONCURRENT_TRADES}) reached. Monitoring open position...")
                time.sleep(CYCLE_SLEEP_SECONDS)
                continue

            # 5. Scan Market for Confluence Setup (filtering out cooled-down pairs)
            active_cooldowns = risk_manager.get_active_cooldowns()
            cooled_down_pairs = {k.split('_')[0] for k, v in active_cooldowns.items() if 'PAIR_LOCKOUT' in v}

            logger.info("Scanning pairs for high-probability technical setups...")
            candidates = scanner.scan(
                conn,
                strategy,
                open_instruments=active_instruments,
                cooled_down_instruments=cooled_down_pairs
            )

            if not candidates:
                logger.info("No candidates passing spread gate and technical confluence. Waiting...")
                time.sleep(CYCLE_SLEEP_SECONDS)
                continue

            # 6. Execute Top Candidate
            top_cand = candidates[0]
            instrument = top_cand["instrument"]
            decision = top_cand["decision"]
            confidence = top_cand["confidence"]
            current_price = top_cand["current_price"]
            spread_pips = top_cand["spread_pips"]
            sl_dist = top_cand["sl_dist"]
            tp_dist = top_cand["tp_dist"]

            # Pre-flight Cooldown Check (Anti-Revenge Trading)
            is_cooled, rem_sec, cool_reason = risk_manager.is_instrument_cooled_down(instrument, decision)
            if is_cooled:
                logger.warning(f"Cooldown active on {instrument} {decision} ({rem_sec}s remaining): {cool_reason}. Skipping execution.")
                time.sleep(CYCLE_SLEEP_SECONDS)
                continue

            # Pre-flight NFA Anti-Hedging & Single-Position Check
            no_hedge, hedge_reason = risk_manager.validate_no_hedging(instrument, decision, open_trades)
            if not no_hedge:
                logger.warning(f"Anti-Hedging veto on {instrument}: {hedge_reason}. Skipping execution.")
                time.sleep(CYCLE_SLEEP_SECONDS)
                continue

            balance, margin_avail = conn.get_account_details()
            units = risk_manager.calculate_units(balance, margin_avail, pair=instrument)

            if units <= 0:
                logger.warning(f"Calculated 0 units for {instrument} (margin or balance limit). Skipping.")
                time.sleep(CYCLE_SLEEP_SECONDS)
                continue

            sl_price, tp_price = risk_manager.calculate_sl_tp_prices(
                decision, current_price, sl_dist, tp_dist, instrument
            )

            logger.info(f"Executing Scalp: {instrument} {decision} | Units: {units:,} | Current: {current_price} | SL: {sl_price} | TP: {tp_price}")

            order_resp = conn.create_order(
                instrument=instrument,
                units=units if decision == "BUY" else -units,
                stop_loss_price=sl_price,
                take_profit_price=tp_price
            )

            if order_resp:
                # Extract OANDA trade ID
                trade_opened = order_resp.get("orderFillTransaction", {}).get("tradeOpened", {})
                trade_id = trade_opened.get("tradeID") or order_resp.get("lastTransactionID", "UNKNOWN")

                ledger.record_open(
                    trade_id=trade_id,
                    instrument=instrument,
                    direction=decision,
                    units=units,
                    entry_price=current_price,
                    stop_loss=sl_price,
                    take_profit=tp_price,
                    spread_pips=spread_pips
                )

                redis_client.publish_event("forex:events", {
                    "event": "TRADE_OPENED",
                    "trade_id": trade_id,
                    "instrument": instrument,
                    "direction": decision,
                    "units": units,
                    "entry_price": current_price,
                    "stop_loss": sl_price,
                    "take_profit": tp_price,
                    "spread_pips": spread_pips
                })

                logger.info("=" * 40)
                logger.info(f"TRADE EXECUTED: {decision} {instrument}")
                logger.info(f"Trade ID:   {trade_id}")
                logger.info(f"Units:      {units:,} (${units * 0.0001:.2f}/pip)")
                logger.info(f"Entry:      {current_price}")
                logger.info(f"Spread:     {spread_pips:.1f} pips")
                logger.info(f"Target TP:  {tp_price}")
                logger.info(f"Stop Loss:  {sl_price}")
                logger.info("=" * 40)
            else:
                logger.error(
                    f"Order placement failed for {instrument} | {decision} {units:,}u | "
                    f"Price: {current_price} | SL: {sl_price} | TP: {tp_price}"
                )

        except Exception as e:
            logger.error(f"Unexpected error in main loop: {e}", exc_info=True)

        time.sleep(CYCLE_SLEEP_SECONDS)

if __name__ == "__main__":
    main()
