import os
import logging
from dotenv import load_dotenv
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails

load_dotenv()

class OandaConnection:
    def __init__(self):
        self.access_token = os.getenv("OANDA_ACCESS_TOKEN")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.env = os.getenv("OANDA_ENV", "practice")
        
        if not self.access_token or not self.account_id:
            raise ValueError("Missing OANDA_ACCESS_TOKEN or OANDA_ACCOUNT_ID in .env")

        self.api = oandapyV20.API(access_token=self.access_token, environment=self.env)
        self.logger = logging.getLogger("trading_bot")

    def get_candles(self, instrument, count=500, granularity="M1"):
        """Fetch historical candle data."""
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M" # Midpoint
        }
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        try:
            self.api.request(r)
            return r.response.get("candles")
        except Exception as e:
            self.logger.error(f"Error fetching candles: {e}")
            return []

    def get_account_details(self):
        """Fetch current account balance and margin available."""
        r = accounts.AccountSummary(accountID=self.account_id)
        try:
            self.api.request(r)
            # We use 'marginAvailable' for safety checks and 'balance' for risk sizing
            balance = float(r.response['account']['balance'])
            margin_avail = float(r.response['account']['marginAvailable'])
            return balance, margin_avail
        except Exception as e:
            self.logger.error(f"Error fetching account details: {e}")
            return 0.0, 0.0

    def get_current_price(self, instrument):
        """Fetch current price for an instrument."""
        params = {"instruments": instrument}
        r = pricing.PricingInfo(accountID=self.account_id, params=params)
        try:
            self.api.request(r)
            prices = r.response.get("prices")
            if prices:
                # return the closeoutBid as a simple 'current price' proxy for selling or closeoutAsk for buying
                # For simplicity, returning the mid of the first price entry
                p = prices[0]
                return (float(p['bids'][0]['price']) + float(p['asks'][0]['price'])) / 2.0
            return None
        except Exception as e:
            self.logger.error(f"Error fetching price: {e}")
            return None

    def create_order(self, instrument, units, stop_loss_price=None, take_profit_price=None):
        """Place a market order with optional SL/TP."""
        
        # Ensure units is integer (OANDA requires int for units)
        units = int(units)
        
        data = {
            "instrument": instrument,
            "units": units,
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }

        if "JPY" in instrument:
            precision = 3
        else:
            precision = 5


        if stop_loss_price:
            data["stopLossOnFill"] = {"price": f"{stop_loss_price:.{precision}f}"}
        
        if take_profit_price:
            data["takeProfitOnFill"] = {"price": f"{take_profit_price:.{precision}f}"}

        r = orders.OrderCreate(accountID=self.account_id, data={"order": data})
        try:
            self.api.request(r)
            self.logger.info(f"Order created: {r.response}")
            return r.response
        except Exception as e:
            self.logger.error(f"Error creating order: {e}")
            return None

    def get_open_trades(self):
        """
        Fetch all open trades.
        Returns a list of trade dictionaries.
        """
        r = oandapyV20.endpoints.trades.TradesList(accountID=self.account_id, params={"state": "OPEN"})
        try:
            self.api.request(r)
            return r.response.get("trades", [])
        except Exception as e:
            self.logger.error(f"Error fetching open trades: {e}")
            return []

    def close_trade(self, trade_id, units=None):
        """
        Close an open trade.
        Args:
            trade_id (str): The ID of the trade to close.
            units (str, optional): Number of units to close. If None, closes all.
        """
        data = {}
        if units:
            data["units"] = str(units)
        else:
            data["units"] = "ALL"

        r = trades.TradeClose(accountID=self.account_id, tradeID=trade_id, data=data)
        try:
            self.api.request(r)
            self.logger.info(f"Trade {trade_id} closed: {r.response}")
            return r.response
        except Exception as e:
            self.logger.error(f"Error closing trade {trade_id}: {e}")
            return None
