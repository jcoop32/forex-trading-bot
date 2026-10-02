import unittest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from strategy import TechnicalScalpStrategy

class TestTechnicalScalpStrategy(unittest.TestCase):
    def setUp(self):
        self.strategy = TechnicalScalpStrategy()

    def _generate_synthetic_candles(self, count=50, base_price=1.1000, trend=0.0):
        start = datetime.now(timezone.utc) - timedelta(minutes=count)
        candles = []
        price = base_price
        for i in range(count):
            price += trend + (0.0001 if i % 2 == 0 else -0.00008)
            t_str = (start + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
            candles.append({
                "time": t_str,
                "mid": {
                    "o": f"{price:.5f}",
                    "h": f"{(price + 0.0002):.5f}",
                    "l": f"{(price - 0.0002):.5f}",
                    "c": f"{price:.5f}"
                },
                "volume": 150
            })
        return candles

    def test_parse_candles(self):
        candles = self._generate_synthetic_candles(count=30)
        df = self.strategy.parse_candles(candles)
        self.assertEqual(len(df), 30)
        self.assertIn("close", df.columns)
        self.assertIn("open", df.columns)

    def test_indicators_calculation(self):
        candles = self._generate_synthetic_candles(count=40)
        df = self.strategy.parse_candles(candles)
        
        rsi = self.strategy.calculate_rsi(df["close"])
        self.assertEqual(len(rsi), 40)
        
        atr = self.strategy.calculate_atr(df)
        self.assertEqual(len(atr), 40)

        bb_up, bb_low = self.strategy.calculate_bollinger_bands(df["close"])
        valid_idx = bb_up.dropna().index
        self.assertTrue((bb_up.loc[valid_idx] >= bb_low.loc[valid_idx]).all())

    def test_pattern_detection(self):
        # Create an engulfing pattern
        df = pd.DataFrame([
            {"open": 1.1000, "high": 1.1005, "low": 1.0995, "close": 1.0998}, # red
            {"open": 1.0996, "high": 1.1010, "low": 1.0994, "close": 1.1008}, # green engulfing
        ])
        is_bull, is_bear = self.strategy.detect_pinbar_engulfing(df)
        self.assertTrue(is_bull)
        self.assertFalse(is_bear)

    def test_evaluate_returns_valid_structure(self):
        c_m1 = self._generate_synthetic_candles(count=40, base_price=1.1000)
        c_m5 = self._generate_synthetic_candles(count=40, base_price=1.1000)
        c_m15 = self._generate_synthetic_candles(count=40, base_price=1.1000)

        decision, conf, atr, sl_dist, tp_dist = self.strategy.evaluate(c_m1, c_m5, c_m15, "EUR_USD")
        self.assertIn(decision, ["BUY", "SELL", "HOLD"])
        self.assertIsInstance(conf, float)
        self.assertGreater(sl_dist, 0.0)
        self.assertGreater(tp_dist, 0.0)

    def test_momentum_veto_on_surging_m1(self):
        # Create a surging upward M1 scenario where last candle is strongly bullish
        c_m1 = self._generate_synthetic_candles(count=38, base_price=1.1000, trend=0.00005)
        last_time = pd.to_datetime(c_m1[-1]["time"])
        t1 = (last_time + timedelta(minutes=1)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        t2 = (last_time + timedelta(minutes=2)).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")

        # Append two explosive green candles at the end
        c_m1.append({
            "time": t1,
            "mid": {"o": "1.10200", "h": "1.10280", "l": "1.10190", "c": "1.10275"},
            "volume": 300
        })
        c_m1.append({
            "time": t2,
            "mid": {"o": "1.10275", "h": "1.10390", "l": "1.10270", "c": "1.10385"},
            "volume": 350
        })
        # M15 in downtrend
        c_m5 = self._generate_synthetic_candles(count=40, base_price=1.1050, trend=-0.0002)
        c_m15 = self._generate_synthetic_candles(count=40, base_price=1.1100, trend=-0.0003)

        decision, conf, _, _, _ = self.strategy.evaluate(c_m1, c_m5, c_m15, "EUR_USD")
        # Even if M15 is bearish and RSI is overbought, the strong bull bars on M1 must trigger momentum veto -> HOLD
        self.assertEqual(decision, "HOLD")


if __name__ == '__main__':
    unittest.main()

