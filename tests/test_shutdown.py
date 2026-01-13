import unittest
from unittest.mock import MagicMock
import logging
import sys
import os

# Ensure src is in path if needed, but main.py is in root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import shutdown_bot

class TestShutdown(unittest.TestCase):
    def setUp(self):
        # Suppress logging during tests
        logging.getLogger("trading_bot").setLevel(logging.CRITICAL)

    def test_shutdown_closes_trades(self):
        # Setup mock connection
        mock_conn = MagicMock()
        
        # Mock open trades
        mock_conn.get_open_trades.return_value = [
            {'id': '101', 'instrument': 'EUR_USD', 'currentUnits': '1000', 'unrealizedPL': '10.50'},
            {'id': '102', 'instrument': 'USD_JPY', 'currentUnits': '-500', 'unrealizedPL': '-5.00'}
        ]
        
        # Run shutdown
        shutdown_bot(mock_conn)
        
        # Verify get_open_trades was called
        mock_conn.get_open_trades.assert_called_once()
        
        # Verify close_trade was called for each trade
        self.assertEqual(mock_conn.close_trade.call_count, 2)
        mock_conn.close_trade.assert_any_call('101')
        mock_conn.close_trade.assert_any_call('102')

    def test_shutdown_no_trades(self):
        mock_conn = MagicMock()
        mock_conn.get_open_trades.return_value = []
        
        shutdown_bot(mock_conn)
        
        mock_conn.get_open_trades.assert_called_once()
        mock_conn.close_trade.assert_not_called()

if __name__ == '__main__':
    unittest.main()
