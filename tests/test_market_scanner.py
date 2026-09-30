import unittest
from unittest.mock import MagicMock
from market_scanner import MarketScanner

class TestMarketScanner(unittest.TestCase):
    def setUp(self):
        # No sentiment analyzer in tests to keep them focused on scanner logic
        self.scanner = MarketScanner(["EUR_USD", "GBP_USD"])
        self.mock_conn = MagicMock()
        self.mock_strategy = MagicMock()

    def test_spread_gate_rejects_wide_spread(self):
        # Quote with 3.0 pips spread on EUR_USD (above absolute ceiling of 2.5)
        self.mock_conn.get_pricing_quote.return_value = {
            "bid": 1.10000,
            "ask": 1.10030,
            "mid": 1.10015,
            "spread_pips": 3.0
        }

        candidates = self.scanner.scan(self.mock_conn, self.mock_strategy)
        self.assertEqual(len(candidates), 0)
        # Verify candles were never fetched because absolute ceiling stopped it early
        self.mock_conn.get_candles.assert_not_called()

    def test_spread_gate_allows_tight_spread_and_finds_candidate(self):
        # Quote with 0.8 pips spread on EUR_USD (well within limits)
        self.mock_conn.get_pricing_quote.return_value = {
            "bid": 1.10000,
            "ask": 1.10008,
            "mid": 1.10004,
            "spread_pips": 0.8
        }
        self.mock_conn.get_candles.return_value = [{"dummy": "candle"}] * 40
        # Strategy outputs strong BUY with ATR that keeps dynamic spread gate happy
        self.mock_strategy.evaluate.return_value = ("BUY", 0.85, 0.00060, 0.00060, 0.00070)

        candidates = self.scanner.scan(self.mock_conn, self.mock_strategy)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["decision"], "BUY")
        # Confidence should remain 0.85 since no sentiment analyzer is configured
        self.assertEqual(candidates[0]["confidence"], 0.85)

    def test_skips_already_active_instruments(self):
        candidates = self.scanner.scan(self.mock_conn, self.mock_strategy, open_instruments={"EUR_USD", "GBP_USD"})
        self.assertEqual(len(candidates), 0)
        self.mock_conn.get_pricing_quote.assert_not_called()

    def test_dynamic_spread_gate_rejects_when_atr_is_small(self):
        """Spread of 2.0 pips is under the absolute ceiling but > 30% of a 5-pip ATR."""
        self.mock_conn.get_pricing_quote.return_value = {
            "bid": 1.10000,
            "ask": 1.10020,
            "mid": 1.10010,
            "spread_pips": 2.0
        }
        self.mock_conn.get_candles.return_value = [{"dummy": "candle"}] * 40
        # ATR = 0.00050 = 5.0 pips -> dynamic limit = 5.0 * 0.3 = 1.5 pips -> 2.0 > 1.5 = reject
        self.mock_strategy.evaluate.return_value = ("BUY", 0.70, 0.00050, 0.00050, 0.00075)

        candidates = self.scanner.scan(self.mock_conn, self.mock_strategy)
        self.assertEqual(len(candidates), 0)

if __name__ == '__main__':
    unittest.main()

