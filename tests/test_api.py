import unittest
from unittest.mock import MagicMock
import tempfile
import os
from fastapi.testclient import TestClient

from api import create_app
from trade_ledger import TradeLedger

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_trades.db")
        self.ledger = TradeLedger(db_path=self.db_path)

        self.mock_conn = MagicMock()
        self.mock_conn.env = "practice"
        self.mock_conn.get_account_details.return_value = (1000.01, 1000.01)
        self.mock_conn.get_open_trades.return_value = [
            {
                "id": "201",
                "instrument": "EUR_USD",
                "currentUnits": "10000",
                "price": "1.10000",
                "unrealizedPL": "5.50",
                "openTime": "2026-09-30T14:00:00Z"
            }
        ]
        self.mock_conn.get_pricing_quote.return_value = {
            "bid": 1.10000,
            "ask": 1.10010,
            "mid": 1.10005,
            "spread_pips": 1.0
        }

        # Pre-populate a closed trade in the ledger
        self.ledger.record_open("101", "EUR_USD", "BUY", 10000, 1.09900)
        self.ledger.record_close("101", 1.09960, 6.00)

        self.app = create_app(connection=self.mock_conn, ledger=self.ledger)
        self.client = TestClient(self.app)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_root_endpoint(self):
        resp = self.client.get("/")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ONLINE")

    def test_get_status_endpoint(self):
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("bot_state", data)
        self.assertEqual(data["today_pnl"], 6.00)

    def test_get_account_endpoint(self):
        resp = self.client.get("/api/account")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["balance"], 1000.01)
        self.assertEqual(data["margin_available"], 1000.01)

    def test_get_active_trades_endpoint(self):
        resp = self.client.get("/api/trades/active")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["active_count"], 1)
        self.assertEqual(data["trades"][0]["trade_id"], "201")
        self.assertEqual(data["trades"][0]["direction"], "BUY")

    def test_get_recent_trades_endpoint(self):
        resp = self.client.get("/api/trades/recent?limit=10")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["total_returned"], 1)
        self.assertEqual(data["trades"][0]["trade_id"], "101")
        self.assertEqual(data["trades"][0]["realized_pl"], 6.00)

    def test_get_daily_pnl_endpoint(self):
        resp = self.client.get("/api/pnl/daily")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["realized_pl"], 6.00)
        self.assertEqual(data["wins"], 1)

    def test_get_quote_endpoint(self):
        resp = self.client.get("/api/quote/EUR_USD")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["spread_pips"], 1.0)
        self.assertEqual(data["bid"], 1.10000)

if __name__ == '__main__':
    unittest.main()
