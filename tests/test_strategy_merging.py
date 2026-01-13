import unittest
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
        # Create M1 data (30 mins = 30 candles)
        # Create M5 data (30 mins = 6 candles)
        # Create M15 data (30 mins = 2 candles)
        
        start = "2023-01-01T10:00:00Z"
        m1 = self.create_mock_candles(start, 30, '1min')
        m5 = self.create_mock_candles(start, 6, '5min')
        m15 = self.create_mock_candles(start, 2, '15min')
        
        df = self.strategy.prepare_data(m1, m5, m15)
        
        # Check if columns exist
        self.assertIn('close', df.columns)      # M1
        self.assertIn('m5_returns', df.columns) # M5
        self.assertIn('m15_returns', df.columns)# M15
        
        # Check alignment logic
        # For M1 at 10:04, it should map to M5 at 10:00 (since 10:05 hasn't closed yet or just closed)
        # prepare_data uses direction='backward', so 10:04 looks for <= 10:04 in M5.
        # M5 candles: 10:00, 10:05...
        # So 10:04 matches 10:00. 10:06 matches 10:05.
        
        # Check row count
        # dropna removes rows where lagged features (shift 1, 2) are NaN.
        # Lag 2 requires 3rd row. 
        # Plus M5/M15 merging might introduce NaNs at start.
        self.assertGreater(len(df), 10)
        
        print("Dataframe Head:")
        print(df[['time', 'close', 'm5_returns', 'm15_returns']].head())

    def test_predict_shape(self):
        # Ensure predict handles shapes correctly
        start = "2023-01-01T10:00:00Z"
        m1 = self.create_mock_candles(start, 50, '1min')
        m5 = self.create_mock_candles(start, 10, '5min')
        m15 = self.create_mock_candles(start, 5, '15min')
        
        # Mock a trained model
        self.strategy.models['TEST'] = unittest.mock.MagicMock()
        self.strategy.models['TEST'].predict.return_value = [1]
        self.strategy.models['TEST'].predict_proba.return_value = [[0.2, 0.8]]
        
        pred, conf = self.strategy.predict(m1, m5, m15, 'TEST')
        self.assertEqual(pred, 1)

if __name__ == '__main__':
    unittest.main()
