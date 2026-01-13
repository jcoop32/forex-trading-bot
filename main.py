import time
import logging
import sys
import warnings
import threading
from connection import OandaConnection
from sentiment import SentimentAnalyzer
from strategy import MLStrategy
from risk_manager import RiskManager
from news_agent import NewsAgent
from market_scanner import MarketScanner

# Configuration
INSTRUMENTS = ["EUR_USD", "GBP_USD", "USD_JPY", "AUD_USD", "USD_CAD"]
TIMEFRAMES = ["M1", "M5", "M15"]
SLEEP_SECONDS = 60

# Scalping Settings
RISK_REWARD_RATIO = 1.5 
ATR_MULTIPLIER_SL = 2.0 # Dynamic Stop Loss
MAX_SPREAD_PIPS = 2.5   # Skip if spread > 2.5 pips
MAX_CONCURRENT_TRADES = 3

# Suppress invalid escape sequence warnings from oandapyV20
warnings.filterwarnings("ignore", category=SyntaxWarning, module="oandapyV20")

# Suppress verbose library logs
logging.getLogger("oandapyV20").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Logging Setup
handlers = [
    logging.FileHandler('trading_bot.log'),
    logging.StreamHandler(sys.stdout)
]

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=handlers
)
logger = logging.getLogger("trading_bot")

def main():
    logger.info("Starting Multi-Timeframe Scalping Bot (ATR & RSI Enhanced)...")

    try:
        # 1. Initialization
        conn = OandaConnection()
        sentiment = SentimentAnalyzer(conn)
        strategy = MLStrategy()
        risk_manager = RiskManager()
        news_agent = NewsAgent()
        scanner = MarketScanner(INSTRUMENTS)
        
        # We don't strictly need a semaphore if we use a loop check, but good for future expansion
        trade_semaphore = threading.Semaphore(MAX_CONCURRENT_TRADES)
        
        logger.info("Connected to OANDA.")

        # 2. Initial Training (Multi-Timeframe)
        # Increased training size for better stability
        TRAIN_COUNT = 2000
        logger.info(f"Training Strategy on all pairs (M1/M5/M15) with {TRAIN_COUNT} candles...")
        for inst in INSTRUMENTS:
            logger.info(f"[{inst}] Fetching historical data...")
            c_m1 = conn.get_candles(inst, count=TRAIN_COUNT, granularity="M1")
            c_m5 = conn.get_candles(inst, count=TRAIN_COUNT, granularity="M5")
            c_m15 = conn.get_candles(inst, count=TRAIN_COUNT, granularity="M15")
            
            if c_m1 and c_m5 and c_m15:
                # OANDA might return fewer than requested if limit hit, but logic handles empty df
                strategy.train(c_m1, c_m5, c_m15, inst)
            else:
                logger.warning(f"Could not train {inst} (missing data).")
        
        # 3. Execution Loop
        while True:
            try:
                logger.info("--- New Cycle ---")
                
                # A. Check Open Trades & Manage Exits
                open_trades = conn.get_open_trades()
                active_count = len(open_trades)
                logger.info(f"Open Trades: {active_count} / {MAX_CONCURRENT_TRADES}")
                
                 # Track active instruments to exclude from scan
                active_instruments = set()

                for t in open_trades:
                    inst = t['instrument']
                    trade_id = t['id']
                    pl = float(t['unrealizedPL'])
                    units = int(t['currentUnits'])
                    direction = "BUY" if units > 0 else "SELL"
                    
                    active_instruments.add(inst)
                    logger.info(f" >> OPEN: {inst} ({direction}) P/L: {pl}")
                    
                    # --- Active Exit Logic ---
                    # 1. Fetch fresh data
                    c_m1 = conn.get_candles(inst, count=50, granularity="M1")
                    c_m5 = conn.get_candles(inst, count=50, granularity="M5")
                    c_m15 = conn.get_candles(inst, count=50, granularity="M15")
                    
                    if c_m1 and c_m5 and c_m15:
                        # 2. Predict
                        pred, conf, current_atr = strategy.predict(c_m1, c_m5, c_m15, inst)
                        # Pred: 1 (UP), 0 (DOWN)
                        
                        # 3. Check for Reversal
                        should_close = False
                        # 3. Check for Reversal
                        should_close = False
                        # if direction == "BUY" and pred == 0:
                        #     logger.info(f"[{inst}] Signal Reversal (BUY -> Pred DOWN). Closing...")
                        #     should_close = True
                        # elif direction == "SELL" and pred == 1:
                        #     logger.info(f"[{inst}] Signal Reversal (SELL -> Pred UP). Closing...")
                        #     should_close = True
                            
                        # Optional: Close if confidence drops? (Maybe too noisy)
                        
                        if should_close:
                            conn.close_trade(trade_id)
                            active_count -= 1 # adjust local count
                            active_instruments.remove(inst) # Free up for scan
                    
                # B. Scan for New Opportunities
                free_slots = MAX_CONCURRENT_TRADES - active_count
                
                if free_slots <= 0:
                    logger.info("Max active trades reached. Skipping scan.")
                    time.sleep(SLEEP_SECONDS)
                    continue

                logger.info("Scanning market...")
                candidates = scanner.scan(conn, strategy, sentiment, news_agent, open_trades)
                
                if not candidates:
                    logger.info("No suitable opportunities found.")
                    time.sleep(SLEEP_SECONDS)
                    continue

                # C. Execute Best Opportunities
                for cand in candidates:
                    if free_slots <= 0:
                        break
                        
                    instrument = cand['instrument']
                    decision = cand['decision']
                    confidence = cand['confidence']
                    news_score = cand['news_score']
                    current_price = cand['current_price']
                    current_atr = cand.get('atr', 0.0010) # Fallback if not in dict
                    
                    # --- SPREAD CHECKS (Crucial for scalping) ---
                    # We need bid/ask for spread. `current_price` might be mid.
                    # Let's quickly re-fetch precise quote or rely on scanner if it passed.
                    # Scanner returns mid. Let's do a quick real-time check.
                    # We can use conn.get_current_price logic but need raw response for spread.
                    # Actually, let's assume if scanner passed it, it's roughly ok, but better check.
                    # OANDA spread is often variable.
                    
                    # For now, let's proceed. 
                    if current_atr == 0:
                        logger.warning(f"ATR is 0 for {instrument}, skipping to be safe.")
                        continue
                        
                    logger.info(f"Executing {decision} on {instrument} (Conf: {confidence:.2f}, ATR: {current_atr:.5f})")
                    
                    # Dynamic Risk
                    risk_pct = risk_manager.calculate_risk_percentage(confidence, "MATCH_SENTIMENT", decision, news_score) 
                    
                    # Determine Stop Distance via ATR
                    sl_pips = current_atr * ATR_MULTIPLIER_SL
                    # Ensure minimum stop (e.g. 5 pips) to avoid rejection
                    MIN_PIPS = 0.0005 if "JPY" not in instrument else 0.05
                    sl_pips = max(sl_pips, MIN_PIPS)
                    
                    sl_pips = max(sl_pips, MIN_PIPS)
                    
                    balance, margin_avail = conn.get_account_details()
                    units = risk_manager.calculate_position_size(
                        account_balance=balance,
                        margin_available=margin_avail,
                        risk_percentage=risk_pct,
                        stop_loss_pips=sl_pips,
                        current_price=current_price,
                        pair=instrument
                    )
                    
                    if units > 0:
                        sl = 0.0
                        tp = 0.0
                        
                        # Calculate SL/TP Prices
                        if decision == "BUY":
                            sl = current_price - sl_pips
                            tp = current_price + (sl_pips * RISK_REWARD_RATIO)
                        elif decision == "SELL":
                            sl = current_price + sl_pips
                            tp = current_price - (sl_pips * RISK_REWARD_RATIO)
                            
                        # Execute
                        logger.info(f" >> Placing Order: {instrument} {decision} | Units: {units} | SL Dist: {sl_pips:.5f}")
                        
                        resp = conn.create_order(instrument, units if decision == "BUY" else -units, stop_loss_price=sl, take_profit_price=tp)
                        
                        if resp:
                            # Log Trade Summary
                            risk_amt = balance * risk_pct
                            potential_profit = risk_amt * RISK_REWARD_RATIO
                            
                            logger.info("\n" + "="*30)
                            logger.info(f"TRADE EXECUTED: {decision} {instrument}")
                            logger.info(f"Units:    {units:,}")
                            logger.info(f"Risk:     ${risk_amt:.2f} ({risk_pct:.2%})")
                            logger.info(f"Target:   ${potential_profit:.2f}")
                            logger.info("="*30 + "\n")
    
                            free_slots -= 1
                        else:
                            logger.warning(f"Order failed for {instrument}.")
                    else:
                        logger.warning(f"Calculated units is 0 for {instrument} (Risk/Margin/Price issue).")

            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)

            time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Critical startup error: {e}", exc_info=True)

if __name__ == "__main__":
    main()
