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
STOP_LOSS_PIPS = 8  # 8 pips (Integer)
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
    logger.info("Starting Multi-Timeframe Scalping Bot...")

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
        logger.info("Training Strategy on all pairs (M1/M5/M15)...")
        for inst in INSTRUMENTS:
            logger.info(f"[{inst}] Fetching historical data...")
            c_m1 = conn.get_candles(inst, count=500, granularity="M1")
            c_m5 = conn.get_candles(inst, count=500, granularity="M5")
            c_m15 = conn.get_candles(inst, count=500, granularity="M15")
            
            if c_m1 and c_m5 and c_m15:
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
                        pred, conf = strategy.predict(c_m1, c_m5, c_m15, inst)
                        # Pred: 1 (UP), 0 (DOWN)
                        
                        # 3. Check for Reversal
                        # If BUY (units > 0) and Pred == 0 (DOWN) -> Close
                        # If SELL (units < 0) and Pred == 1 (UP) -> Close
                        
                        should_close = False
                        if direction == "BUY" and pred == 0:
                            logger.info(f"[{inst}] Signal Reversal (BUY -> Pred DOWN). Closing...")
                            should_close = True
                        elif direction == "SELL" and pred == 1:
                            logger.info(f"[{inst}] Signal Reversal (SELL -> Pred UP). Closing...")
                            should_close = True
                            
                        # Optional: Close if confidence drops? (Maybe too noisy)
                        
                        if should_close:
                            conn.close_trade(trade_id)
                            active_count -= 1 # adjust local count
                            active_instruments.remove(inst) # Free up for scan? prefer waiting next cycle
                    
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
                    
                    logger.info(f"Executing {decision} on {instrument} (Conf: {confidence:.2f})")
                    
                    # Dynamic Risk
                    risk_pct = risk_manager.calculate_risk_percentage(confidence, "MATCH_SENTIMENT", decision, news_score) 
                    
                    # Determine Pip Size (Moved up for calc usage)
                    if "JPY" in instrument:
                         pip_unit = 0.01
                    else:
                         pip_unit = 0.0001
                         
                    sl_dist = STOP_LOSS_PIPS * pip_unit
                    
                    balance = conn.get_account_balance()
                    units = risk_manager.calculate_position_size(
                        account_balance=balance,
                        risk_percentage=risk_pct,
                        stop_loss_pips=sl_dist, # Pass the calculated price distance (e.g. 0.08 or 0.0008)
                        current_price=current_price,
                        pair=instrument
                    )
                    
                    if units > 0:
                        sl = 0.0
                        tp = 0.0
                        
                        if decision == "BUY":
                            sl = current_price - sl_dist
                            tp = current_price + (sl_dist * RISK_REWARD_RATIO)
                            conn.create_order(instrument, units, stop_loss_price=sl, take_profit_price=tp)
                        elif decision == "SELL":
                            sl = current_price + sl_dist
                            tp = current_price - (sl_dist * RISK_REWARD_RATIO)
                            conn.create_order(instrument, -units, stop_loss_price=sl, take_profit_price=tp)
                            
                        # Log Trade Summary
                        risk_amt = balance * risk_pct
                        potential_profit = risk_amt * RISK_REWARD_RATIO
                        
                        logger.info("\n" + "="*30)
                        logger.info(f"TRADE EXECUTED: {decision} {instrument}")
                        logger.info(f"Units:    {units:,}")
                        logger.info(f"Risk:     ${risk_amt:.2f} ({risk_pct:.2%})")
                        logger.info(f"Est. P/L: ${potential_profit:.2f}")
                        logger.info("="*30 + "\n")

                        free_slots -= 1
                    else:
                        logger.warning(f"Calculated units is 0 for {instrument}.")

            except Exception as e:
                logger.error(f"Error in main loop: {e}")

            time.sleep(SLEEP_SECONDS)

    except KeyboardInterrupt:
        logger.info("Bot stopped by user.")
    except Exception as e:
        logger.error(f"Critical startup error: {e}")

if __name__ == "__main__":
    main()
