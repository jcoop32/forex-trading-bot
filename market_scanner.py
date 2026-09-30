import logging
import concurrent.futures

class MarketScanner:
    """
    Concurrent market scanner equipped with strict real-time Spread Gating.
    Scans major currency pairs and surfaces high-confidence technical setups.
    """
    def __init__(self, instruments=None):
        self.logger = logging.getLogger("trading_bot")
        self.instruments = instruments or ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "AUD_USD"]
        
        # Max allowed spread in pips before aborting
        self.MAX_SPREAD_MAJORS = 1.2    # EUR_USD, GBP_USD
        self.MAX_SPREAD_CROSSES = 1.8   # USD_JPY, USD_CAD, AUD_USD

    def _get_max_allowed_spread(self, instrument):
        if instrument in ["EUR_USD", "GBP_USD"]:
            return self.MAX_SPREAD_MAJORS
        return self.MAX_SPREAD_CROSSES

    def scan(self, connection, strategy, open_instruments=None):
        """
        Scan all configured pairs concurrently.
        
        Args:
            connection (OandaConnection): API connection
            strategy (TechnicalScalpStrategy): Technical strategy instance
            open_instruments (set): Set of instruments with active open trades to skip
            
        Returns:
            list: List of trade candidates sorted by confidence (descending)
        """
        open_instruments = open_instruments or set()
        candidates = []

        def analyze_pair(instrument):
            if instrument in open_instruments:
                return None

            try:
                # 1. Real-time Pricing & Spread Gate
                quote = connection.get_pricing_quote(instrument)
                if not quote:
                    return None

                spread = quote["spread_pips"]
                max_spread = self._get_max_allowed_spread(instrument)

                if spread > max_spread:
                    self.logger.info(f"[{instrument}] Spread {spread:.1f} pips exceeds max {max_spread:.1f}. Skipping.")
                    return None

                # 2. Fetch Multi-Timeframe Candles
                c_m1 = connection.get_candles(instrument, count=40, granularity="M1")
                c_m5 = connection.get_candles(instrument, count=40, granularity="M5")
                c_m15 = connection.get_candles(instrument, count=40, granularity="M15")

                if not c_m1 or not c_m5:
                    return None

                # 3. Strategy Evaluation
                decision, conf, atr, sl_dist, tp_dist = strategy.evaluate(c_m1, c_m5, c_m15, instrument)

                if decision in ["BUY", "SELL"]:
                    return {
                        "instrument": instrument,
                        "decision": decision,
                        "confidence": conf,
                        "current_price": quote["mid"],
                        "spread_pips": spread,
                        "atr": atr,
                        "sl_dist": sl_dist,
                        "tp_dist": tp_dist
                    }
                return None

            except Exception as e:
                self.logger.error(f"Error scanning {instrument}: {e}")
                return None

        # Concurrently analyze pairs
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.instruments)) as executor:
            futures = {executor.submit(analyze_pair, inst): inst for inst in self.instruments}
            for future in concurrent.futures.as_completed(futures):
                inst = futures[future]
                try:
                    res = future.result()
                    if res:
                        candidates.append(res)
                        self.logger.info(f"Candidate: {res['instrument']} {res['decision']} (Conf: {res['confidence']:.2f}, Spread: {res['spread_pips']:.1f}p)")
                except Exception as e:
                    self.logger.error(f"Scanner exception for {inst}: {e}")

        # Sort by confidence descending
        candidates.sort(key=lambda x: x["confidence"], reverse=True)
        return candidates
