import logging
import time
import concurrent.futures
from utils import pip_unit as get_pip_unit

class MarketScanner:
    """
    Concurrent market scanner equipped with dynamic Spread Gating and
    time-based candle caching. Scans major currency pairs and surfaces
    high-confidence technical setups.
    """
    def __init__(self, instruments=None):
        self.logger = logging.getLogger("trading_bot")
        self.instruments = instruments or ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "AUD_USD"]
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.instruments))

        # Absolute spread boundaries (pips) - dynamic gate adjusts within these
        self.SPREAD_FLOOR = 0.8   # Never reject a spread tighter than this
        self.SPREAD_CEILING = 2.5 # Never allow a spread wider than this
        # Fallback static limits used when ATR is unavailable
        self.MAX_SPREAD_MAJORS = 1.2    # EUR_USD, GBP_USD
        self.MAX_SPREAD_CROSSES = 1.8   # USD_JPY, USD_CAD, AUD_USD

        # Candle cache: {(instrument, granularity): (timestamp, data)}
        # M1 is never cached (always fresh), M5 cached 5min, M15 cached 15min
        self._candle_cache = {}
        self._cache_ttl = {"M1": 0, "M5": 300, "M15": 900}

    def _get_candles(self, connection, instrument, granularity, count=40):
        """
        Fetch candles with time-based caching for higher timeframes.
        M5 candles are cached for 5 minutes, M15 for 15 minutes.
        M1 is always fetched fresh.
        """
        ttl = self._cache_ttl.get(granularity, 0)
        cache_key = (instrument, granularity)

        if ttl > 0:
            cached = self._candle_cache.get(cache_key)
            if cached:
                cached_at, data = cached
                age = time.time() - cached_at
                if age < ttl:
                    return data

        # Fetch fresh data from OANDA
        data = connection.get_candles(instrument, count=count, granularity=granularity)

        if data and ttl > 0:
            self._candle_cache[cache_key] = (time.time(), data)

        return data

    def shutdown(self):
        """Clean up the thread pool executor."""
        self._executor.shutdown(wait=False)
        self.logger.info("MarketScanner: Thread pool executor shut down.")

    def _get_max_allowed_spread(self, instrument, atr_pips=None):
        """
        Dynamic spread gate: spread should be < 30% of ATR.
        Falls back to static limits when ATR is unavailable.
        Clamped between SPREAD_FLOOR and SPREAD_CEILING.
        """
        if atr_pips and atr_pips > 0:
            dynamic_limit = atr_pips * 0.3
            return max(self.SPREAD_FLOOR, min(self.SPREAD_CEILING, dynamic_limit))

        # Static fallback
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
                # 1. Real-time Pricing & Initial Spread Check
                quote = connection.get_pricing_quote(instrument)
                if not quote:
                    return None

                spread = quote["spread_pips"]

                # Quick reject on absolute ceiling before burning API calls on candles
                if spread > self.SPREAD_CEILING:
                    self.logger.info(f"[{instrument}] Spread {spread:.1f}p exceeds absolute ceiling {self.SPREAD_CEILING:.1f}. Skipping.")
                    return None

                # 2. Fetch Multi-Timeframe Candles (M5/M15 cached)
                c_m1 = self._get_candles(connection, instrument, "M1")
                c_m5 = self._get_candles(connection, instrument, "M5")
                c_m15 = self._get_candles(connection, instrument, "M15")

                if not c_m1 or not c_m5:
                    return None

                # 3. Strategy Evaluation
                decision, conf, atr, sl_dist, tp_dist = strategy.evaluate(c_m1, c_m5, c_m15, instrument)

                if decision not in ["BUY", "SELL"]:
                    return None

                # 4. Dynamic Spread Gate (now informed by ATR from strategy)
                pip_unit = get_pip_unit(instrument)
                atr_pips = atr / pip_unit if atr else None
                max_spread = self._get_max_allowed_spread(instrument, atr_pips)

                if spread > max_spread:
                    self.logger.info(f"[{instrument}] Spread {spread:.1f}p exceeds dynamic limit {max_spread:.1f}p (ATR: {atr_pips:.1f}p). Skipping.")
                    return None

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

            except Exception as e:
                self.logger.error(f"Error scanning {instrument}: {e}")
                return None

        # Concurrently analyze pairs using the persistent thread pool
        futures = {self._executor.submit(analyze_pair, inst): inst for inst in self.instruments}
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
