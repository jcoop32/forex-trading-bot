import unittest
from unittest.mock import MagicMock
from risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager()
        self.rm.logger = MagicMock()

    def test_size_eur_usd(self):
        # Account: 10,000 USD
        # Risk: 1% = 100 USD
        # SL Dist: 0.0008 (8 pips)
        # Pair: EUR_USD (Quote USD)
        
        # Units = Risk / Dist = 100 / 0.0008 = 125,000
        units = self.rm.calculate_position_size(10000, 0.01, 0.0008, 1.10, "EUR_USD")
        self.assertEqual(units, 125000)

    def test_size_usd_jpy(self):
        # Account: 10,000 USD
        # Risk: 1% = 100 USD
        # SL Dist: 0.08 (8 pips for JPY)
        # Pair: USD_JPY (Base USD)
        # Price: 150.00
        
        # Logic: 
        # Loss per unit (USD) = SL_Dist / Price = 0.08 / 150 = 0.0005333 USD
        # Units = Risk / Loss = 100 / (0.08/150) = 100 * 150 / 0.08 = 15,000 / 0.08 = 187,500
        
        units = self.rm.calculate_position_size(10000, 0.01, 0.08, 150.00, "USD_JPY")
        self.assertEqual(units, 187500)

if __name__ == '__main__':
    unittest.main()
