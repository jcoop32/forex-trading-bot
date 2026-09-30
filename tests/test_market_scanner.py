import unittest
from unittest.mock import MagicMock
from market_scanner import MarketScanner

class TestMarketScanner(unittest.TestCase):
    def setUp(self):
        self.scanner = MarketScanner(["EUR_USD", "GBP_USD"])
        self.mock_conn = MagicMock()
        self.mock_strategy = MagicMock()

    def test_spread_gate_rejects_wide_spread(self):
        # Quote with 2.5 pips spread on EUR_USD (max allowed is 1.2)
        self.mock_conn.get_pricing_quote.return_value = {
            "bid": 1.10000,
            "ask": 1.10025,
            "mid": 1.10012,
            "spread_pips": 2.5
        }

        candidates = self.scanner.scan(self.mock_conn, self.mock_strategy)
        self.assertEqual(len(candidates), 0)
        # Verify candles were never even fetched because spread gate stopped it early!
        self.mock_conn.get_candles.assert_not_called()

    def test_spread_gate_allows_tight_spread_and_finds_candidate(self):
        # Quote with 0.8 pips spread on EUR_USD (within 1.2 limit)
        self.mock_conn.get_pricing_quote.return_value = {
            "bid": 1.10000,
            "ask": 1.10008,
            "mid": 1.10004,
            "spread_pips": 0.8
        }
        self.mock_conn.get_candles.return_value = [{"dummy": "candle"}] * 40
        # Strategy outputs strong BUY
        self.mock_strategy.evaluate.return_value = ("BUY", 0.85, 0.00060, 0.00060, 0.00070)

        candidates = self.scanner.scan(self.mock_conn, self.mock_strategy)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["decision"], "BUY")
        self.assertEqual(candidates[0]["confidence"], 0.85)

    def test_skips_already_active_instruments(self):
        candidates = self.scanner.scan(self.mock_conn, self.mock_strategy, open_instruments={"EUR_USD", "GBP_USD"})
        self.assertEqual(len(candidates), 0)
        self.mock_conn.get_pricing_quote.assert_not_called()

if __name__ == '__main__':
    unittest.main()
