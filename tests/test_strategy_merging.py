import unittest
from unittest.mock import MagicMock
import sys

# Mock sklearn
sys.modules['sklearn'] = MagicMock()
sys.modules['sklearn.ensemble'] = MagicMock()

import pandas as pd
import numpy as np
from strategy import MLStrategy

class TestStrategyMerging(unittest.TestCase):
    def setUp(self):
        self.strategy = MLStrategy()
        
    def create_mock_candles(self, start_time, count, freq='1min', start_price=1.0):
        # Generate timestamps
        times = pd.date_range(start=start_time, periods=count, freq=freq)
        candles = []
        price = start_price
        for t in times:
            c = {
                'time': t.isoformat(),
                'mid': {'o': str(price), 'h': str(price+0.0005), 'l': str(price-0.0005), 'c': str(price)},
                'volume': "100"
            }
            candles.append(c)
            price += 0.0001 # strict trend
        return candles

    def test_multi_timeframe_merge(self):
        # Increased to 300 to ensure SMA(20) is valid for all timeframes
        
        start = "2023-01-01T10:00:00Z"
        m1 = self.create_mock_candles(start, 300, '1min')
        m5 = self.create_mock_candles(start, 300, '5min')
        m15 = self.create_mock_candles(start, 300, '15min')
        
        df = self.strategy.prepare_data(m1, m5, m15)
        
        # Check if columns exist
        self.assertIn('close', df.columns)      # M1
        self.assertIn('m5_returns', df.columns) # M5
        self.assertIn('m15_returns', df.columns)# M15
        
        # Check alignment logic
        self.assertGreater(len(df), 10)
        
        print("Dataframe Head:")
        print(df[['time', 'close', 'm5_returns', 'm15_returns']].head())

    def test_predict_shape(self):
        # Ensure predict handles shapes correctly
        start = "2023-01-01T10:00:00Z"
        m1 = self.create_mock_candles(start, 300, '1min')
        m5 = self.create_mock_candles(start, 300, '5min')
        m15 = self.create_mock_candles(start, 300, '15min')
        
        # Mock a trained model
        self.strategy.models['TEST'] = unittest.mock.MagicMock()
        self.strategy.models['TEST'].predict.return_value = [1]
        self.strategy.models['TEST'].predict_proba.return_value = [[0.2, 0.8]]
        
        pred, conf, atr = self.strategy.predict(m1, m5, m15, 'TEST')
        self.assertEqual(pred, 1)

if __name__ == '__main__':
    unittest.main()
