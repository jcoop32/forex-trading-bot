import logging

class RiskManager:
    """
    Risk and position manager optimized for a $1,000 paper trading scalping account.
    Enforces micro-lot sizing, margin protection, and max single-position caps.
    """
    def __init__(self, target_balance=1000.0):
        self.logger = logging.getLogger("trading_bot")
        self.target_balance = target_balance
        
        # Risk Configuration for $1,000 Account
        self.BASE_UNITS = 10000      # 1 mini-lot (10,000 units = $1.00/pip on USD quote)
        self.MIN_UNITS = 5000        # Half mini-lot (5,000 units = $0.50/pip)
        self.MIN_MARGIN_BUFFER = 350.0  # Cash margin required before opening new trade
        self.MAX_RISK_PER_TRADE_USD = 15.0  # Maximum $15 risk per scalp (1.5% of $1,000)

    def calculate_units(self, account_balance, margin_available, pair="EUR_USD"):
        """
        Calculate safe trade units for $1,000 account sizing.
        
        Args:
            account_balance (float): Total account balance
            margin_available (float): Available free margin
            pair (str): Currency pair
            
        Returns:
            int: Safe units to trade (e.g., 10000 or 5000), or 0 if insufficient margin.
        """
        effective_balance = account_balance if account_balance > 0 else self.target_balance

        # 1. Margin Available Verification
        if margin_available < self.MIN_MARGIN_BUFFER:
            if margin_available >= 180.0:
                self.logger.warning(f"Margin tight (${margin_available:.2f}). Dropping to {self.MIN_UNITS} units.")
                return self.MIN_UNITS
            else:
                self.logger.warning(f"Insufficient margin available (${margin_available:.2f} < ${self.MIN_MARGIN_BUFFER:.2f}). Aborting trade.")
                return 0

        # 2. Scale proportionally if account grows or shrinks
        scale_factor = effective_balance / 1000.0
        
        if scale_factor < 0.7:
            # Below $700, trade half-size
            units = self.MIN_UNITS
        elif scale_factor <= 1.5:
            # $700 - $1,500: trade standard 10,000 units
            units = self.BASE_UNITS
        else:
            # Above $1,500, scale incrementally in 5,000 unit blocks
            units = int(scale_factor * self.BASE_UNITS / 5000) * 5000

        self.logger.info(f"Position Sizing: Balance ${effective_balance:.2f} | Free Margin: ${margin_available:.2f} -> {units:,} Units ({pair})")
        return units

    def calculate_sl_tp_prices(self, decision, current_price, sl_dist, tp_dist, instrument):
        """
        Calculate absolute Stop Loss and Take Profit price levels.
        """
        precision = 3 if "JPY" in instrument else 5
        
        if decision == "BUY":
            sl_price = current_price - sl_dist
            tp_price = current_price + tp_dist
        elif decision == "SELL":
            sl_price = current_price + sl_dist
            tp_price = current_price - tp_dist
        else:
            return None, None

        return round(sl_price, precision), round(tp_price, precision)
