import logging
import time
import concurrent.futures
from utils import pip_unit as get_pip_unit

class MarketScanner:
    """
    Concurrent market scanner equipped with dynamic Take-Profit-relative Spread Gating
    and time-based candle caching. Scans major currency pairs and surfaces
    high-confidence technical setups.
    """
    def __init__(self, instruments=None):
        self.logger = logging.getLogger("trading_bot")
        self.instruments = instruments or ["EUR_USD", "GBP_USD", "USD_JPY", "USD_CAD", "AUD_USD"]
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.instruments))

        # Spread boundaries (pips)
        self.SPREAD_CEILING = 2.8  # Absolute safety ceiling (e.g. news spike / rollover)
        self.SPREAD_FLOOR = 1.0    # Absolute minimum floor

        # Realistic pair-specific baseline spread floors for retail broker feeds
        self.PAIR_SPREAD_FLOORS = {
            "EUR_USD": 1.2,
            "AUD_USD": 1.4,
            "USD_CAD": 1.7,
            "USD_JPY": 1.7,
            "GBP_USD": 1.9,
        }

        # Fallback static limits used when TP/ATR is unavailable
        self.MAX_SPREAD_MAJORS = 1.4    # EUR_USD, GBP_USD
        self.MAX_SPREAD_CROSSES = 2.0   # USD_JPY, USD_CAD, AUD_USD

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

    def _get_max_allowed_spread(self, instrument, tp_pips=None, atr_pips=None):
        """
        Dynamic spread gate:
        1. If tp_pips is provided, spread must be <= 20% of TP distance (ensures >= 5:1 reward:spread).
        2. Clamped by pair-specific baseline floor and absolute SPREAD_CEILING.
        3. Falls back to ATR-based or static limits if TP is unavailable.
        """
        pair_floor = self.PAIR_SPREAD_FLOORS.get(instrument, self.SPREAD_FLOOR)

        if tp_pips and tp_pips > 0:
            tp_limit = tp_pips * 0.20
            dynamic_limit = max(pair_floor, tp_limit)
            return min(self.SPREAD_CEILING, dynamic_limit)

        if atr_pips and atr_pips > 0:
            dynamic_limit = max(pair_floor, atr_pips * 0.3)
            return min(self.SPREAD_CEILING, dynamic_limit)

        # Static fallback
        if instrument in ["EUR_USD", "GBP_USD"]:
            return self.MAX_SPREAD_MAJORS
        return self.MAX_SPREAD_CROSSES

    def scan(self, connection, strategy, open_instruments=None, cooled_down_instruments=None):
        """
        Scan all configured pairs concurrently.
        
        Args:
            connection (OandaConnection): API connection
            strategy (TechnicalScalpStrategy): Technical strategy instance
            open_instruments (set): Set of instruments with active open trades to skip
            cooled_down_instruments (set): Set of instruments currently in post-loss cooldown
            
        Returns:
            list: List of trade candidates sorted by confidence (descending)
        """
        open_instruments = open_instruments or set()
        cooled_down_instruments = cooled_down_instruments or set()
        candidates = []

        def analyze_pair(instrument):
            if instrument in open_instruments:
                return None

            if instrument in cooled_down_instruments:
                self.logger.info(f"[{instrument}] Pair on cooldown. Skipping scan.")
                return None

            try:
                # 1. Real-time Pricing & Initial Spread Check
                quote = connection.get_pricing_quote(instrument)
                if not quote:
                    return None

                spread = quote["spread_pips"]

                # Quick reject on absolute ceiling before burning API calls on candles
                if spread > self.SPREAD_CEILING:
                    self.logger.info(f"[{instrument}] Spread {spread:.1f}p exceeds absolute ceiling {self.SPREAD_CEILING:.1f}p. Skipping.")
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

                # 4. Dynamic Spread Gate (Relative to Take-Profit distance)
                pip_unit = get_pip_unit(instrument)
                tp_pips = tp_dist / pip_unit if tp_dist else None
                atr_pips = atr / pip_unit if atr else None
                max_spread = self._get_max_allowed_spread(instrument, tp_pips=tp_pips, atr_pips=atr_pips)

                if spread > max_spread:
                    self.logger.info(
                        f"[{instrument}] Spread {spread:.1f}p exceeds dynamic limit {max_spread:.1f}p "
                        f"(TP: {tp_pips:.1f}p, Floor: {self.PAIR_SPREAD_FLOORS.get(instrument, 1.2):.1f}p). Skipping."
                    )
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
