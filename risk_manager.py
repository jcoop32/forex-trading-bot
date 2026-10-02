import logging
import time
from utils import price_precision

class RiskManager:
    """
    Risk and position manager optimized for a $1,000 scalping account.
    Enforces micro-lot sizing, margin protection, NFA Rule 2-43(b) anti-hedging,
    and anti-revenge trading cooldowns (post-loss and consecutive loss lockouts).
    """
    def __init__(self, target_balance=1000.0):
        self.logger = logging.getLogger("trading_bot")
        self.target_balance = target_balance
        
        # Risk Configuration for $1,000 Account
        self.BASE_UNITS = 10000          # 1 mini-lot (10,000 units = $1.00/pip on USD quote)
        self.MIN_UNITS = 5000            # Half mini-lot (5,000 units = $0.50/pip)
        self.MIN_MARGIN_BUFFER = 350.0   # Cash margin required before opening new trade
        self.MAX_RISK_PER_TRADE_USD = 15.0  # Maximum $15 risk per scalp (1.5% of $1,000)

        # Anti-Revenge & Circuit Breaker Durations (Seconds)
        self.POST_LOSS_COOLDOWN_SECONDS = 15 * 60       # 15 minutes freeze on instrument/direction
        self.CONSECUTIVE_LOSS_LOCKOUT_SECONDS = 120 * 60 # 2 hours freeze after 2 consecutive losses
        self.CONSECUTIVE_LOSS_WINDOW = 60 * 60           # 1-hour evaluation window

        # State registries
        # {(instrument, direction): timestamp_cooled_down_until}
        self._direction_cooldowns = {}
        # {instrument: timestamp_locked_until}
        self._pair_lockouts = {}
        # {instrument: [{"timestamp": float, "pl": float, "direction": str}]}
        self._trade_history = {}

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
            # Above $1,500, scale in 2,500-unit steps per $500 increment
            extra_blocks = int((effective_balance - 1000.0) / 500.0)
            units = self.BASE_UNITS + (extra_blocks * 2500)

        self.logger.info(f"Position Sizing: Balance ${effective_balance:.2f} | Free Margin: ${margin_available:.2f} -> {units:,} Units ({pair})")
        return units

    def calculate_sl_tp_prices(self, decision, current_price, sl_dist, tp_dist, instrument):
        """
        Calculate absolute Stop Loss and Take Profit price levels.
        """
        precision = price_precision(instrument)
        
        if decision == "BUY":
            sl_price = current_price - sl_dist
            tp_price = current_price + tp_dist
        elif decision == "SELL":
            sl_price = current_price + sl_dist
            tp_price = current_price - tp_dist
        else:
            return None, None

        return round(sl_price, precision), round(tp_price, precision)

    def record_trade_result(self, instrument, direction, realized_pl, timestamp=None):
        """
        Record the outcome of a closed trade to update cooldown and lockout registries.
        Enforces 15m post-loss cooldown and 2h consecutive loss lockout.
        """
        now = timestamp or time.time()
        if instrument not in self._trade_history:
            self._trade_history[instrument] = []

        self._trade_history[instrument].append({
            "timestamp": now,
            "direction": direction,
            "realized_pl": realized_pl
        })

        # Trim history older than 24 hours
        self._trade_history[instrument] = [
            t for t in self._trade_history[instrument] if now - t["timestamp"] <= 86400
        ]

        if realized_pl < 0:
            # 1. Apply single-direction post-loss cooldown (15 mins)
            cool_until = now + self.POST_LOSS_COOLDOWN_SECONDS
            self._direction_cooldowns[(instrument, direction)] = cool_until
            self.logger.warning(
                f"🛑 [{instrument}] Trade closed at loss (${realized_pl:.2f}). "
                f"Post-loss cooldown active for {direction} until {time.strftime('%H:%M:%S', time.gmtime(cool_until))} UTC (15m)."
            )

            # 2. Check for consecutive losses within the evaluation window (60 mins)
            recent_losses = [
                t for t in self._trade_history[instrument]
                if now - t["timestamp"] <= self.CONSECUTIVE_LOSS_WINDOW and t["realized_pl"] < 0
            ]
            
            # Check if last 2 trades were consecutive losses
            history = self._trade_history[instrument]
            if len(history) >= 2 and history[-1]["realized_pl"] < 0 and history[-2]["realized_pl"] < 0:
                lock_until = now + self.CONSECUTIVE_LOSS_LOCKOUT_SECONDS
                self._pair_lockouts[instrument] = lock_until
                self.logger.warning(
                    f"🚨 [{instrument}] 2 consecutive losses detected! Pair locked out completely until "
                    f"{time.strftime('%H:%M:%S', time.gmtime(lock_until))} UTC (2 hours) to prevent tilt/churn."
                )
        else:
            self.logger.info(f"🏆 [{instrument}] Trade closed at profit (${realized_pl:.2f}). Registry updated.")

    def is_instrument_cooled_down(self, instrument, direction=None, current_time=None):
        """
        Check if an instrument or direction is currently blocked by a cooldown or lockout.
        Returns:
            tuple: (is_blocked: bool, remaining_seconds: int, reason: str)
        """
        now = current_time or time.time()

        # 1. Check pair-level lockout (e.g. 2 consecutive losses)
        if instrument in self._pair_lockouts:
            locked_until = self._pair_lockouts[instrument]
            if now < locked_until:
                remaining = int(locked_until - now)
                return True, remaining, f"Pair lockout active (consecutive losses: {remaining}s remaining)"
            else:
                del self._pair_lockouts[instrument]

        # 2. Check direction-specific cooldown (e.g. 15m post-loss)
        if direction:
            key = (instrument, direction)
            if key in self._direction_cooldowns:
                cool_until = self._direction_cooldowns[key]
                if now < cool_until:
                    remaining = int(cool_until - now)
                    return True, remaining, f"Direction cooldown active ({direction} post-loss: {remaining}s remaining)"
                else:
                    del self._direction_cooldowns[key]

        return False, 0, ""

    def validate_no_hedging(self, instrument, proposed_direction, open_trades):
        """
        Enforce OANDA US NFA Rule 2-43(b) Anti-Hedging:
        Disallow opening a trade if an opposing or identical position already exists.
        
        Returns:
            tuple: (is_valid: bool, reason: str)
        """
        for t in open_trades:
            if t.get("instrument") == instrument:
                units = float(t.get("currentUnits", 0))
                existing_dir = "BUY" if units > 0 else "SELL"
                if existing_dir != proposed_direction:
                    return False, f"NFA Anti-Hedging Violation: cannot open {proposed_direction} when {existing_dir} is already open on {instrument}"
                else:
                    return False, f"Single-position invariant: position already active on {instrument}"
        return True, ""

    def get_active_cooldowns(self, current_time=None):
        """Return human-readable summary of active cooldowns and lockouts."""
        now = current_time or time.time()
        active = {}
        for inst, until in list(self._pair_lockouts.items()):
            rem = int(until - now)
            if rem > 0:
                active[inst] = f"PAIR_LOCKOUT ({rem}s)"
        for (inst, d), until in list(self._direction_cooldowns.items()):
            rem = int(until - now)
            if rem > 0:
                key = f"{inst}_{d}"
                if key not in active:
                    active[key] = f"DIRECTION_COOLDOWN ({rem}s)"
        return active
