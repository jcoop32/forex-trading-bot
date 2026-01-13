import logging
import concurrent.futures
import time

class MarketScanner:
    def __init__(self, instruments):
        self.logger = logging.getLogger("trading_bot")
        self.instruments = instruments
        self.TIMEFRAME = "M1" # Use one timeframe for simplicity in this version

    def scan(self, connection, strategy, sentiment_analyzer, news_agent, open_trades):
        """
        Scans all instruments concurrently to find the best trading opportunities.
        
        Args:
            connection (OandaConnection): API connection
            strategy (MLStrategy): Strategy instance
            sentiment_analyzer (SentimentAnalyzer): Sentiment tool
            news_agent (NewsAgent): News tool
            open_trades (list): List of currently open trade objects from OANDA
            
        Returns:
            list: List of dicts [{"instrument": "EUR_USD", "signal": "BUY", "confidence": 0.85, "news_score": 0.5}, ...]
                  Sorted by confidence (descending).
        """
        
        # 1. Identify pairs that are already active to skip or manage
        active_instruments = {t['instrument'] for t in open_trades}
        
        candidates = []
        
        # 2. Define the worker function for a single pair
        def analyze_pair(instrument):
            if instrument in active_instruments:
                # We already have a trade, skip scanning for new entry to avoid complexity
                # (Or we could scan for exit signals, but let's stick to entry finding for now)
                return None
            
            try:
                # A. Fetch Data (Multi-Timeframe)
                # We need M1, M5, M15
                candidates_m1 = connection.get_candles(instrument, count=50, granularity="M1")
                candidates_m5 = connection.get_candles(instrument, count=50, granularity="M5")
                candidates_m15 = connection.get_candles(instrument, count=50, granularity="M15")
                
                if not candidates_m1 or not candidates_m5 or not candidates_m15:
                    return None
                    
                # B. Strategy Prediction
                # Pass all timeframes
                ml_signal, ml_confidence = strategy.predict(candidates_m1, candidates_m5, candidates_m15, instrument)
                
                # C. Sentiment Analysis
                sent_signal = sentiment_analyzer.get_market_sentiment(instrument)
                
                # D. News Sentiment
                # (Optional: might be slow to fetch for ALL pairs every cycle. 
                #  Maybe only fetch for high confidence ones? 
                #  For now, let's fetch, but rely on NewsAgent's caching)
                news_score = news_agent.get_sentiment_score(instrument)
                
                # E. Combine Logic (mimic main.py logic)
                decision = "HOLD"
                if ml_signal == 1 and sent_signal != "SELL":
                    decision = "BUY"
                elif ml_signal == 0 and sent_signal != "BUY":
                    decision = "SELL"
                
                if decision != "HOLD":
                    return {
                        "instrument": instrument,
                        "decision": decision,
                        "confidence": ml_confidence,
                        "news_score": news_score,
                        "current_price": connection.get_current_price(instrument) # Needed for sizing
                    }
                return None
                
            except Exception as e:
                self.logger.error(f"Error analyzing {instrument}: {e}")
                return None

        # 3. specific Concurrency
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            future_to_instrument = {executor.submit(analyze_pair, instr): instr for instr in self.instruments}
            
            for future in concurrent.futures.as_completed(future_to_instrument):
                instr = future_to_instrument[future]
                try:
                    result = future.result()
                    if result:
                        candidates.append(result)
                        self.logger.info(f"Candidate Found: {result['instrument']} ({result['decision']}, Conf: {result['confidence']:.2f})")
                except Exception as e:
                    self.logger.error(f"Scanner exception for {instr}: {e}")

        # 4. Sort candidates by confidence
        candidates.sort(key=lambda x: x['confidence'], reverse=True)
        return candidates
