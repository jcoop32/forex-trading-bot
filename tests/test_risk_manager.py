import unittest
from unittest.mock import MagicMock
from risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()
        self.rm.logger = MagicMock()

    def test_size_eur_usd(self):
        # Account: 10,000 USD, Margin Avail: 10,000
        # Risk: 1% = 100 USD
        # SL Dist: 0.0008 (8 pips)
        # Pair: EUR_USD (Quote USD)
        
        # Units = Risk / Dist = 100 / 0.0008 = 125,000
        # Check Margin: 125,000 * 1.10 = $137,500 Nominal. Max = 10,000 * 20 * 0.95 = 190,000. OK.
        units = self.rm.calculate_position_size(10000, 10000, 0.01, 0.0008, 1.10, "EUR_USD")
        self.assertEqual(units, 125000)

    def test_size_usd_jpy(self):
        # Account: 10,000 USD
        # Risk: 1% = 100 USD
        # SL Dist: 0.08 (8 pips for JPY)
        # Pair: USD_JPY (Base USD)
        # Price: 150.00
        
        # Units = 187,500. Nominal = 187,500 USD. Max = 190,000. OK.
        units = self.rm.calculate_position_size(10000, 10000, 0.01, 0.08, 150.00, "USD_JPY")
        self.assertEqual(units, 187500)

    def test_margin_cap(self):
        # Test capping logic
        # Balance 10k, Margin Avail ONLY 1k (lots of open trades)
        # Risk 1% = 100 USD.
        # SL Dist 0.0010 (10 pips).
        # Target Units = 100 / 0.0010 = 100,000 Units.
        
        # Max Nominal = 1,000 * 20 * 0.95 = 19,000 USD.
        # EUR_USD Price 1.1. Max Units = 19,000 / 1.1 = 17,272.
        
        units = self.rm.calculate_position_size(10000, 1000, 0.01, 0.0010, 1.10, "EUR_USD")
        self.assertEqual(units, 17272)

if __name__ == '__main__':
    unittest.main()
