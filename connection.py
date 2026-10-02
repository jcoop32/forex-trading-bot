import os
import logging
import oandapyV20
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.pricing as pricing
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import oandapyV20.endpoints.positions as positions
import oandapyV20.endpoints.accounts as accounts
from oandapyV20.contrib.requests import MarketOrderRequest, TakeProfitDetails, StopLossDetails
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception, before_sleep_log
from utils import pip_unit, price_precision, format_price

logger = logging.getLogger("trading_bot")


def _is_retriable_error(exception):
    """Only retry on transient network/rate-limit errors, not auth or validation failures."""
    if isinstance(exception, oandapyV20.exceptions.V20Error):
        code = getattr(exception, 'code', None)
        # 429 = rate limited, 503 = service unavailable
        return code in (429, 503)
    if isinstance(exception, (ConnectionError, TimeoutError, OSError)):
        return True
    return False


# Shared retry decorator for read-only API calls
_api_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception(_is_retriable_error),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


class OandaConnection:
    def __init__(self):
        self.access_token = os.getenv("OANDA_ACCESS_TOKEN")
        self.account_id = os.getenv("OANDA_ACCOUNT_ID")
        self.env = os.getenv("OANDA_ENV", "practice")
        
        if not self.access_token or not self.account_id:
            raise ValueError("Missing OANDA_ACCESS_TOKEN or OANDA_ACCOUNT_ID in .env")

        self.api = oandapyV20.API(access_token=self.access_token, environment=self.env)
        self.logger = logging.getLogger("trading_bot")

    @_api_retry
    def get_candles(self, instrument, count=500, granularity="M1"):
        """Fetch historical candle data."""
        params = {
            "count": count,
            "granularity": granularity,
            "price": "M"  # Midpoint
        }
        r = instruments.InstrumentsCandles(instrument=instrument, params=params)
        try:
            self.api.request(r)
            return r.response.get("candles", [])
        except oandapyV20.exceptions.V20Error:
            raise  # Let tenacity decide whether to retry based on error code
        except Exception as e:
            self.logger.error(f"Error fetching candles for {instrument}: {e}")
            return []

    @_api_retry
    def get_account_details(self):
        """Fetch current account balance and margin available."""
        r = accounts.AccountSummary(accountID=self.account_id)
        try:
            self.api.request(r)
            balance = float(r.response['account']['balance'])
            margin_avail = float(r.response['account']['marginAvailable'])
            return balance, margin_avail
        except oandapyV20.exceptions.V20Error:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching account details: {e}")
            return 0.0, 0.0

    @_api_retry
    def get_pricing_quote(self, instrument):
        """
        Fetch real-time bid, ask, mid, and spread in pips.
        Returns dict with 'bid', 'ask', 'mid', 'spread_pips', or None.
        """
        params = {"instruments": instrument}
        r = pricing.PricingInfo(accountID=self.account_id, params=params)
        try:
            self.api.request(r)
            prices = r.response.get("prices", [])
            if not prices:
                return None
            p = prices[0]
            bid = float(p['bids'][0]['price'])
            ask = float(p['asks'][0]['price'])
            mid = (bid + ask) / 2.0
            
            pu = pip_unit(instrument)
            spread_pips = (ask - bid) / pu
            
            return {
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "spread_pips": round(spread_pips, 2)
            }
        except oandapyV20.exceptions.V20Error:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching pricing quote for {instrument}: {e}")
            return None

    def get_current_price(self, instrument):
        """Fetch mid price for an instrument."""
        quote = self.get_pricing_quote(instrument)
        if quote:
            return quote["mid"]
        return None

    def create_order(self, instrument, units, stop_loss_price=None, take_profit_price=None):
        """Place a market order with optional SL/TP. Not retried (market orders are not idempotent)."""
        units = int(units)
        
        data = {
            "instrument": instrument,
            "units": units,
            "type": "MARKET",
            "positionFill": "DEFAULT"
        }

        if stop_loss_price:
            data["stopLossOnFill"] = {"price": format_price(stop_loss_price, instrument)}
        
        if take_profit_price:
            data["takeProfitOnFill"] = {"price": format_price(take_profit_price, instrument)}

        r = orders.OrderCreate(accountID=self.account_id, data={"order": data})
        try:
            self.api.request(r)
            self.logger.info(f"Order created for {instrument} ({units}u): {r.response}")
            return r.response
        except oandapyV20.exceptions.V20Error as e:
            self.logger.error(
                f"OANDA rejected order for {instrument} ({units}u, SL={stop_loss_price}, TP={take_profit_price}): "
                f"Code={e.code} | {e.msg}"
            )
            return None
        except Exception as e:
            self.logger.error(f"Error creating order for {instrument} ({units}u): {e}")
            return None

    @_api_retry
    def get_open_trades(self):
        """Fetch all open trades."""
        r = trades.TradesList(accountID=self.account_id, params={"state": "OPEN"})
        try:
            self.api.request(r)
            return r.response.get("trades", [])
        except oandapyV20.exceptions.V20Error:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching open trades: {e}")
            return []

    @_api_retry
    def get_trade_details(self, trade_id):
        """
        Fetch full details for a trade (open or closed).
        Returns dict with 'realizedPL', 'averageClosePrice', 'state', etc., or None on failure.
        """
        r = trades.TradeDetails(accountID=self.account_id, tradeID=str(trade_id))
        try:
            self.api.request(r)
            trade = r.response.get("trade", {})
            return {
                "trade_id": trade.get("id"),
                "instrument": trade.get("instrument"),
                "state": trade.get("state"),
                "realizedPL": float(trade.get("realizedPL", 0.0)),
                "averageClosePrice": float(trade.get("averageClosePrice", 0.0)) if trade.get("averageClosePrice") else None,
                "currentUnits": int(trade.get("currentUnits", 0)),
                "unrealizedPL": float(trade.get("unrealizedPL", 0.0)),
            }
        except oandapyV20.exceptions.V20Error:
            raise
        except Exception as e:
            self.logger.error(f"Error fetching trade details for {trade_id}: {e}")
            return None

    def close_trade(self, trade_id, units=None):
        """Close an open trade. Not retried (closing is not idempotent)."""
        data = {}
        if units:
            data["units"] = str(units)
        else:
            data["units"] = "ALL"

        r = trades.TradeClose(accountID=self.account_id, tradeID=str(trade_id), data=data)
        try:
            self.api.request(r)
            self.logger.info(f"Trade {trade_id} closed: {r.response}")
            return r.response
        except Exception as e:
            self.logger.error(f"Error closing trade {trade_id}: {e}")
            return None

    def close_position_fifo(self, instrument, direction="ALL"):
        """
        Close positions using OANDA's native FIFO-compliant position endpoint.
        Automatically offsets the oldest units first in compliance with NFA Rule 2-43(b).
        """
        data = {}
        if direction in ["BUY", "LONG", "ALL"]:
            data["longUnits"] = "ALL"
        if direction in ["SELL", "SHORT", "ALL"]:
            data["shortUnits"] = "ALL"

        r = positions.PositionClose(accountID=self.account_id, instrument=instrument, data=data)
        try:
            self.api.request(r)
            self.logger.info(f"Position for {instrument} closed (FIFO compliant): {r.response}")
            return r.response
        except Exception as e:
            self.logger.error(f"Error closing position for {instrument} (FIFO): {e}")
            return None


    def modify_trade_sl(self, trade_id, new_sl_price, instrument):
        """
        Modify an open trade's stop loss price via OANDA Trade CRCDO (Client Rate Change Dependent Orders).
        Returns the response dict on success, None on failure.
        """
        data = {
            "stopLoss": {
                "price": format_price(new_sl_price, instrument)
            }
        }
        r = trades.TradeCRCDO(accountID=self.account_id, tradeID=str(trade_id), data=data)
        try:
            self.api.request(r)
            self.logger.info(f"Trade {trade_id} SL modified to {format_price(new_sl_price, instrument)}")
            return r.response
        except Exception as e:
            self.logger.error(f"Error modifying SL for trade {trade_id}: {e}")
            return None
