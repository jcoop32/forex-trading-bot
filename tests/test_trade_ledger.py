import unittest
import os
import tempfile
from trade_ledger import TradeLedger

class TestTradeLedger(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_trades.db")
        self.ledger = TradeLedger(db_path=self.db_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_record_open_and_close(self):
        self.ledger.record_open(
            trade_id="101",
            instrument="EUR_USD",
            direction="BUY",
            units=10000,
            entry_price=1.10000,
            stop_loss=1.09940,
            take_profit=1.10070,
            spread_pips=1.0
        )

        # Check daily pnl before close (should be 0.0)
        summary = self.ledger.get_daily_pnl()
        self.assertEqual(summary["total_trades"], 0)
        self.assertEqual(summary["realized_pl"], 0.0)

        # Close trade with profit
        self.ledger.record_close(trade_id="101", exit_price=1.10070, realized_pl=7.00)

        summary = self.ledger.get_daily_pnl()
        self.assertEqual(summary["total_trades"], 1)
        self.assertEqual(summary["realized_pl"], 7.00)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 0)

    def test_daily_summary_aggregates_multiple_trades(self):
        self.ledger.record_open("101", "EUR_USD", "BUY", 10000, 1.1000)
        self.ledger.record_close("101", 1.1006, 6.00)

        self.ledger.record_open("102", "GBP_USD", "SELL", 10000, 1.3000)
        self.ledger.record_close("102", 1.3006, -6.00)

        summary = self.ledger.get_daily_pnl()
        self.assertEqual(summary["total_trades"], 2)
        self.assertEqual(summary["realized_pl"], 0.00)
        self.assertEqual(summary["wins"], 1)
        self.assertEqual(summary["losses"], 1)

if __name__ == '__main__':
    unittest.main()
