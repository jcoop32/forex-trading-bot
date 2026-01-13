import unittest
import pandas as pd
import numpy as np
from strategy import MLStrategy
import logging

# Configure logging to show info during tests
logging.basicConfig(level=logging.INFO)

class TestStrategyAdxPatterns(unittest.TestCase):
    def setUp(self):
        self.strategy = MLStrategy()
        
    def create_mock_candles(self, start_time, count, freq='1min', start_price=1.0):
        # Generate timestamps
        times = pd.date_range(start=start_time, periods=count, freq=freq)
        candles = []
        price = start_price
        
        # Consistent small moves
        for t in times:
            c = {
                'time': t.isoformat(),
                'mid': {'o': str(price), 'h': str(price+0.0005), 'l': str(price-0.0005), 'c': str(price+0.0001)}, # small bullish
                'volume': "100"
            }
            candles.append(c)
            price += 0.0001
        
        return candles

    def test_adx_calculation(self):
        """Test ADX calculation on synthetic trending data."""
        # Create a strong trend to ensure high ADX
        times = pd.date_range(start="2023-01-01", periods=50, freq='1min')
        data = []
        price = 1.0
        for t in times:
            # Strong Bullish Trend: Highs making higher highs, lows making higher lows
            data.append({
                'time': t,
                'open': price,
                'high': price + 0.0010,
                'low': price - 0.0002,
                'close': price + 0.0008,
                'volume': 100
            })
            price += 0.0005
            
        df = pd.DataFrame(data)
        adx = self.strategy.calculate_adx(df, period=14)
        
        # Check output type and length
        self.assertIsInstance(adx, pd.Series)
        self.assertEqual(len(adx), 50)
        
        # Last values should be high due to strong trend (e.g., > 20 or 30)
        print(f"Final ADX: {adx.iloc[-1]}")
        self.assertGreater(adx.iloc[-1], 20.0)
        
    def test_pattern_pinbar(self):
        """Test Pinbar detection logic."""
        # Create 3 candles:
        # 0: Normal
        # 1: Bullish Pinbar (Long lower wick)
        # 2: Bearish Pinbar (Long upper wick)
        
        data = [
            {'open': 1.0, 'high': 1.0005, 'low': 0.9995, 'close': 1.0002}, # Normal
            # Bullish: Open 1.0, Low 0.9950 (Wick 0.0050), Close 1.0010, High 1.0012. Body 0.0010. Lower Wick 5x Body.
            {'open': 1.0, 'high': 1.0012, 'low': 0.9950, 'close': 1.0010}, 
            # Bearish: Open 1.0, High 1.0050 (Wick 0.0050), Close 0.9990, Low 0.9988. Body 0.0010. Upper Wick 5x Body.
            {'open': 1.0, 'high': 1.0050, 'low': 0.9988, 'close': 0.9990}, 
        ]
        df = pd.DataFrame(data)
        pinbar, _ = self.strategy.detect_patterns(df)
        
        self.assertEqual(pinbar[0], 0)
        self.assertEqual(pinbar[1], 1)  # Bullish
        self.assertEqual(pinbar[2], -1) # Bearish
        
    def test_pattern_engulfing(self):
        """Test Engulfing detection logic."""
        # 0: Red Candle
        # 1: Green Candle (Bullish Engulfing 0)
        # 2: Green Candle
        # 3: Red Candle (Bearish Engulfing 2)
        
        data = [
            # 0: Red, Body 0.0010
            {'open': 1.0020, 'high': 1.0025, 'low': 1.0005, 'close': 1.0010}, 
            # 1: Green, Open 1.0005, Close 1.0030 (Body 0.0025 > 0.0010). Bullish Engulfing.
            {'open': 1.0005, 'high': 1.0035, 'low': 1.0000, 'close': 1.0030},
            # 2: Green, Body small
            {'open': 1.0030, 'high': 1.0040, 'low': 1.0025, 'close': 1.0035},
            # 3: Red, Open 1.0040, Close 1.0020 (Body 0.0020 > 0.0005). Bearish Engulfing.
            {'open': 1.0040, 'high': 1.0045, 'low': 1.0015, 'close': 1.0020},
        ]
        df = pd.DataFrame(data)
        _, engulfing = self.strategy.detect_patterns(df)
        
        self.assertEqual(engulfing[1], 1.0) # Bullish Engulfing
        self.assertEqual(engulfing[3], -1.0) # Bearish Engulfing

    def test_integration(self):
        """Test full pipeline integration."""
        start = "2023-01-01T10:00:00Z"
        # Need enough M1 data to overlap with VALID M15 data (which needs ~20 candles = 300 mins)
        m1 = self.create_mock_candles(start, 1000, '1min')
        m5 = self.create_mock_candles(start, 200, '5min')
        m15 = self.create_mock_candles(start, 200, '15min')
        
        # Train
        self.strategy.train(m1, m5, m15, "EUR_USD")
        
        # Predict
        pred, conf, atr = self.strategy.predict(m1, m5, m15, "EUR_USD")
        
        # Check if model exists and has correct features
        self.assertIn("EUR_USD", self.strategy.models)
        model = self.strategy.models["EUR_USD"]
        self.assertTrue(hasattr(model, "feature_importances_"))
        
        # Check feature count: 19 features in the list
        # We can check input shape of trained model
        self.assertEqual(model.n_features_in_, 19)

if __name__ == '__main__':
    unittest.main()
