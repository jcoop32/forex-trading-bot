import unittest
from unittest.mock import MagicMock
from market_scanner import MarketScanner

class TestMarketScanner(unittest.TestCase):
    def setUp(self):
        self.instruments = ["EUR_USD", "GBP_USD", "USD_JPY"]
        self.scanner = MarketScanner(self.instruments)
        
        # Mocks
        self.conn = MagicMock()
        self.strategy = MagicMock()
        self.sentiment = MagicMock()
        self.news_agent = MagicMock()
        
        # Default mock returns
        self.conn.get_candles.return_value = [{'mid': {'c': '1.0'}}] # partial candle mock
        self.conn.get_current_price.return_value = 1.1000
        self.news_agent.get_sentiment_score.return_value = 0.5
        self.strategy.predict.return_value = (0, 0.0) # Default safe return

    def test_scan_finds_opportunities(self):
        # Setup specific returns for different pairs to ensure sorting
        
        # EUR_USD: BUY, Conf 0.9 (Best)
        # GBP_USD: SELL, Conf 0.7 
        # USD_JPY: HOLD (No signal)
        
        def strategy_predict_side_effect(c1, c2, c3, instrument):
            if instrument == "EUR_USD": return 1, 0.9  # BUY Signal (1), High Conf
            if instrument == "GBP_USD": return 0, 0.7  # SELL Signal (0), Med Conf
            if instrument == "USD_JPY": return 0, 0.4  # SELL Signal (0), Low Conf
            return 0, 0.0

        self.strategy.predict.side_effect = strategy_predict_side_effect
        
        self.sentiment.get_market_sentiment.return_value = "NEUTRAL"
        
        # Run scan
        candidates = self.scanner.scan(self.conn, self.strategy, self.sentiment, self.news_agent, [])
        
        # Logic check: 
        # EUR_USD: ML=1, Sent=Neutral -> BUY
        # GBP_USD: ML=0, Sent=Neutral -> SELL
        # USD_JPY: ML=0, Sent=Neutral -> SELL
        
        # The candidates list should have 3 items.
        self.assertEqual(len(candidates), 3)
        
        # Ensure sorting by confident
        self.assertEqual(candidates[0]['instrument'], "EUR_USD")
        self.assertEqual(candidates[0]['confidence'], 0.9)
        
        self.assertEqual(candidates[1]['instrument'], "GBP_USD")
        self.assertEqual(candidates[1]['confidence'], 0.7)

    def test_scan_skips_open_trades(self):
        # Scan should skip EUR_USD if it's already open
        open_trades = [{'instrument': 'EUR_USD', 'unrealizedPL': '10'}]
        
        active_instruments = {t['instrument'] for t in open_trades}
        # Mock logic mimics scan() internal skipping
        
        # Run
        candidates = self.scanner.scan(self.conn, self.strategy, self.sentiment, self.news_agent, open_trades)
        
        # Candidates should NOT contain EUR_USD
        found_eur = any(c['instrument'] == 'EUR_USD' for c in candidates)
        self.assertFalse(found_eur)

if __name__ == '__main__':
    unittest.main()
