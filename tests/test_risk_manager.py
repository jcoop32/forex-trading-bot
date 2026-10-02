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

    def test_post_loss_cooldown(self):
        now = 1000000.0
        # Trade closes at a loss
        self.rm.record_trade_result("USD_JPY", "SELL", -5.50, timestamp=now)
        
        # 5 minutes later, SELL should still be blocked by cooldown
        blocked, remaining, reason = self.rm.is_instrument_cooled_down("USD_JPY", "SELL", current_time=now + 300)
        self.assertTrue(blocked)
        self.assertEqual(remaining, 600)
        self.assertIn("Direction cooldown active", reason)

        # Opposite direction (BUY) should NOT be blocked
        blocked_buy, _, _ = self.rm.is_instrument_cooled_down("USD_JPY", "BUY", current_time=now + 300)
        self.assertFalse(blocked_buy)

        # 16 minutes later, cooldown expired
        blocked_expired, _, _ = self.rm.is_instrument_cooled_down("USD_JPY", "SELL", current_time=now + 960)
        self.assertFalse(blocked_expired)

    def test_consecutive_loss_lockout(self):
        now = 1000000.0
        # First loss
        self.rm.record_trade_result("USD_JPY", "SELL", -5.00, timestamp=now)
        # Second consecutive loss 10 minutes later
        self.rm.record_trade_result("USD_JPY", "SELL", -4.50, timestamp=now + 600)

        # Pair should now be completely locked out for 2 hours (7200s)
        blocked_buy, rem, reason = self.rm.is_instrument_cooled_down("USD_JPY", "BUY", current_time=now + 700)
        self.assertTrue(blocked_buy)
        self.assertIn("Pair lockout active", reason)

        blocked_sell, _, _ = self.rm.is_instrument_cooled_down("USD_JPY", "SELL", current_time=now + 700)
        self.assertTrue(blocked_sell)

        # Other pairs are not affected
        blocked_eur, _, _ = self.rm.is_instrument_cooled_down("EUR_USD", "BUY", current_time=now + 700)
        self.assertFalse(blocked_eur)

    def test_validate_no_hedging(self):
        open_trades = [
            {"instrument": "EUR_USD", "currentUnits": 10000}  # Active BUY
        ]

        # Cannot open SELL on EUR_USD (hedging violation)
        valid, reason = self.rm.validate_no_hedging("EUR_USD", "SELL", open_trades)
        self.assertFalse(valid)
        self.assertIn("NFA Anti-Hedging Violation", reason)

        # Cannot open duplicate BUY on EUR_USD (single-position invariant)
        valid_dup, reason_dup = self.rm.validate_no_hedging("EUR_USD", "BUY", open_trades)
        self.assertFalse(valid_dup)
        self.assertIn("Single-position invariant", reason_dup)

        # Can open BUY or SELL on USD_JPY (no active position)
        valid_jpy, _ = self.rm.validate_no_hedging("USD_JPY", "BUY", open_trades)
        self.assertTrue(valid_jpy)

if __name__ == '__main__':
    unittest.main()

