import unittest
from risk_manager import RiskManager

class TestRiskManager(unittest.TestCase):
    def setUp(self):
        self.rm = RiskManager(target_balance=1000.0)

    def test_units_on_standard_balance(self):
        # On $1,000 balance with ample margin, should return 10,000 units
        units = self.rm.calculate_units(account_balance=1000.0, margin_available=600.0, pair="EUR_USD")
        self.assertEqual(units, 10000)

    def test_units_on_low_margin(self):
        # Margin tight (< 350 but >= 180), drops to 5,000 units
        units = self.rm.calculate_units(account_balance=1000.0, margin_available=250.0, pair="EUR_USD")
        self.assertEqual(units, 5000)

    def test_units_on_insufficient_margin(self):
        # Margin < 180, aborts with 0 units
        units = self.rm.calculate_units(account_balance=1000.0, margin_available=100.0, pair="EUR_USD")
        self.assertEqual(units, 0)

    def test_units_scale_down_on_small_balance(self):
        # Balance < $700, trades half-size (5,000 units)
        units = self.rm.calculate_units(account_balance=650.0, margin_available=500.0, pair="EUR_USD")
        self.assertEqual(units, 5000)

    def test_calculate_sl_tp_prices_standard(self):
        sl, tp = self.rm.calculate_sl_tp_prices(
            decision="BUY",
            current_price=1.10000,
            sl_dist=0.00060,  # 6 pips
            tp_dist=0.00070,  # 7 pips
            instrument="EUR_USD"
        )
        self.assertAlmostEqual(sl, 1.09940, places=5)
        self.assertAlmostEqual(tp, 1.10070, places=5)

    def test_calculate_sl_tp_prices_jpy(self):
        sl, tp = self.rm.calculate_sl_tp_prices(
            decision="SELL",
            current_price=150.000,
            sl_dist=0.060,  # 6 pips on JPY
            tp_dist=0.070,  # 7 pips on JPY
            instrument="USD_JPY"
        )
        self.assertAlmostEqual(sl, 150.060, places=3)
        self.assertAlmostEqual(tp, 149.930, places=3)

if __name__ == '__main__':
    unittest.main()
