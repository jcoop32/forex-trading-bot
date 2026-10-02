"""
High-Fidelity Historical Backtesting Engine for Forex Scalper Bot.
Simulates exact multi-timeframe strategy execution (M1/M5/M15), realistic spread costs,
SL/TP tick execution, trailing stops, 50% TP breakeven ratchets, and anti-revenge cooldowns.
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# Load local environment if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from utils import pip_unit as get_pip_unit, price_precision, format_price
from strategy import TechnicalScalpStrategy
from risk_manager import RiskManager
from connection import OandaConnection


# Baseline realistic retail broker spreads per pair (in pips)
DEFAULT_PAIR_SPREADS = {
    "EUR_USD": 1.3,
    "AUD_USD": 1.4,
    "USD_CAD": 1.8,
    "USD_JPY": 1.6,
    "GBP_USD": 2.0,
}


class HistoricalDataDownloader:
    """Manages downloading, paginating, and local caching of OANDA candle data."""
    def __init__(self, cache_dir="data/candles"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        self.conn = None

    def _get_connection(self):
        if not self.conn:
            self.conn = OandaConnection()
        return self.conn

    def get_candles(self, instrument, granularity="M1", count=5000, refresh=False):
        """
        Fetch candles from local cache or OANDA API.
        Returns clean pandas DataFrame sorted by time.
        """
        cache_path = os.path.join(self.cache_dir, f"{instrument}_{granularity}.json")

        if not refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, "r") as f:
                    cached_data = json.load(f)
                    df = self._parse_json_candles(cached_data)
                    if len(df) >= min(count, 500):
                        return df
            except Exception as e:
                print(f"[!] Warning reading cache for {instrument} {granularity}: {e}. Re-fetching.")

        # Fetch fresh data from OANDA
        conn = self._get_connection()
        print(f"[*] Downloading {count} {granularity} candles for {instrument} from OANDA...")
        raw_candles = conn.get_candles(instrument, count=count, granularity=granularity)

        if raw_candles:
            try:
                with open(cache_path, "w") as f:
                    json.dump(raw_candles, f)
            except Exception as e:
                print(f"[!] Warning caching {instrument}: {e}")

        return self._parse_json_candles(raw_candles)

    def _parse_json_candles(self, raw_candles):
        if not raw_candles:
            return pd.DataFrame()
        records = []
        for c in raw_candles:
            if not c.get("complete", True):
                continue
            mid = c.get("mid", {})
            records.append({
                "time": pd.to_datetime(c["time"]),
                "open": float(mid.get("o", 0.0)),
                "high": float(mid.get("h", 0.0)),
                "low": float(mid.get("l", 0.0)),
                "close": float(mid.get("c", 0.0)),
                "volume": int(c.get("volume", 0))
            })
        df = pd.DataFrame(records)
        if not df.empty:
            df.sort_values("time", inplace=True)
            df.reset_index(drop=True, inplace=True)
        return df


class BacktestSimulator:
    """
    Event-driven and vectorized backtester that executes the strategy step-by-step.
    """
    def __init__(self, mode="enhanced", initial_balance=1000.0):
        """
        mode:
          - 'enhanced': with 15m cooldown, 2-loss lockout, TP-relative spread gate,
                        M1 momentum veto, and 50% TP breakeven ratchet.
          - 'baseline': without cooldowns, with legacy 0.8p spread gate, no momentum veto.
        """
        self.mode = mode
        self.initial_balance = initial_balance
        self.strategy = TechnicalScalpStrategy()

    def run(self, instrument, df_m1, df_m5, df_m15, spread_pips=None):
        """Run backtest on synchronized candle data for a single pair."""
        if len(df_m1) < 50 or len(df_m5) < 30:
            return {"error": "Insufficient candle data"}

        pip_unit = get_pip_unit(instrument)
        simulated_spread = spread_pips if spread_pips is not None else DEFAULT_PAIR_SPREADS.get(instrument, 1.5)

        # Setup Risk Manager for cooldown tracking
        rm = RiskManager(target_balance=self.initial_balance)
        balance = self.initial_balance

        trades = []
        active_trade = None
        trailing_sl = None

        # Precompute indicators vectorially on M1 for fast replay
        df_m1 = df_m1.copy()
        df_m1["rsi"] = self.strategy.calculate_rsi(df_m1["close"])
        df_m1["atr"] = self.strategy.calculate_atr(df_m1)
        bb_upper, bb_lower = self.strategy.calculate_bollinger_bands(df_m1["close"])
        df_m1["bb_upper"] = bb_upper
        df_m1["bb_lower"] = bb_lower
        df_m1["m1_ema9"] = df_m1["close"].ewm(span=9, adjust=False).mean()

        # Precompute M5 ADX and HTF trend
        df_m5 = df_m5.copy()
        df_m15 = df_m15.copy()
        m15_fast = df_m15["close"].ewm(span=9, adjust=False).mean()
        m15_slow = df_m15["close"].ewm(span=21, adjust=False).mean()
        df_m15["htf_trend"] = np.where(m15_fast > m15_slow, "BULL", "BEAR")

        # Step through M1 candles starting at index 40 to allow indicator warmup
        for i in range(40, len(df_m1)):
            current_bar = df_m1.iloc[i]
            prev_bar = df_m1.iloc[i - 1]
            curr_time = current_bar["time"]
            curr_time_sec = curr_time.timestamp()

            # -------------------------------------------------------------
            # 1. Manage Active Position (Check SL, TP, BE Ratchet, Trailing)
            # -------------------------------------------------------------
            if active_trade is not None:
                entry_price = active_trade["entry_price"]
                direction = active_trade["direction"]
                sl_price = active_trade["current_sl"]
                tp_price = active_trade["tp_price"]
                units = active_trade["units"]
                tp_dist_pips = active_trade["tp_dist_pips"]

                hit_sl = False
                hit_tp = False
                exit_price = 0.0

                if direction == "BUY":
                    # Check if Low hit SL
                    if current_bar["low"] <= sl_price:
                        hit_sl = True
                        exit_price = sl_price
                    # Check if High hit TP
                    elif current_bar["high"] >= tp_price:
                        hit_tp = True
                        exit_price = tp_price

                    profit_pips = (current_bar["close"] - entry_price) / pip_unit
                else:  # SELL
                    # Check if High hit SL
                    if current_bar["high"] >= sl_price:
                        hit_sl = True
                        exit_price = sl_price
                    # Check if Low hit TP
                    elif current_bar["low"] <= tp_price:
                        hit_tp = True
                        exit_price = tp_price

                    profit_pips = (entry_price - current_bar["close"]) / pip_unit

                if hit_sl or hit_tp:
                    # Calculate realized PnL
                    if direction == "BUY":
                        raw_pnl_pips = (exit_price - entry_price) / pip_unit
                    else:
                        raw_pnl_pips = (entry_price - exit_price) / pip_unit

                    # For USD quote pairs (EUR_USD, GBP_USD, AUD_USD), 10,000 units = $1.00/pip
                    # For USD_JPY (150.00), 10,000 units = ~$0.67/pip
                    pip_usd_val = 1.0
                    if "JPY" in instrument:
                        pip_usd_val = 10000 * 0.01 / exit_price
                    elif "CAD" in instrument:
                        pip_usd_val = 10000 * 0.0001 / exit_price

                    realized_pl = raw_pnl_pips * pip_usd_val
                    balance += realized_pl

                    active_trade["closed_at"] = curr_time
                    active_trade["exit_price"] = exit_price
                    active_trade["realized_pl"] = round(realized_pl, 2)
                    active_trade["outcome"] = "WIN" if realized_pl > 0 else "LOSS"
                    active_trade["duration_min"] = int((curr_time - active_trade["opened_at"]).total_seconds() / 60)
                    trades.append(active_trade)

                    # Update Risk Manager cooldowns
                    if self.mode == "enhanced":
                        rm.record_trade_result(instrument, direction, realized_pl, timestamp=curr_time_sec)

                    active_trade = None
                    trailing_sl = None
                    continue

                # Position still open: update Breakeven and Trailing Stop
                if profit_pips > 0:
                    be_buffer = 0.5 * pip_unit
                    be_sl = (entry_price + be_buffer) if direction == "BUY" else (entry_price - be_buffer)
                    new_sl = None

                    if self.mode == "enhanced":
                        # 50% TP Breakeven Ratchet
                        be_threshold = max(3.5, tp_dist_pips * 0.50)
                        if profit_pips >= be_threshold and profit_pips < 5.0:
                            new_sl = be_sl

                    if profit_pips >= 5.0:
                        locked_pips = 5.0 + (int((profit_pips - 5.0) / 3.0) * 3.0)
                        trail_dist = locked_pips * pip_unit
                        new_sl = (entry_price + be_buffer + (trail_dist - 5.0 * pip_unit)) if direction == "BUY" else (entry_price - be_buffer - (trail_dist - 5.0 * pip_unit))

                    if new_sl is not None:
                        last_sl = active_trade["current_sl"]
                        if (direction == "BUY" and new_sl > last_sl) or (direction == "SELL" and new_sl < last_sl):
                            active_trade["current_sl"] = new_sl

                continue  # Single position rule: don't open new trade while one is open

            # -------------------------------------------------------------
            # 2. Check Cooldowns & Lockouts (Enhanced Mode)
            # -------------------------------------------------------------
            if self.mode == "enhanced":
                is_cooled, _, _ = rm.is_instrument_cooled_down(instrument, current_time=curr_time_sec)
                if is_cooled:
                    continue

            # -------------------------------------------------------------
            # 3. Strategy Evaluation
            # -------------------------------------------------------------
            # Match latest available M5 and M15 bars prior to current M1 time (no lookahead)
            slice_m5 = df_m5[df_m5["time"] <= curr_time]
            slice_m15 = df_m15[df_m15["time"] <= curr_time]
            if len(slice_m5) < 25 or len(slice_m15) < 25:
                continue

            htf_trend = slice_m15["htf_trend"].iloc[-1]
            adx_val = self.strategy.calculate_adx(slice_m5)
            if adx_val < 15:
                continue

            current_close = current_bar["close"]
            current_rsi = current_bar["rsi"]
            current_atr = current_bar["atr"]
            if pd.isna(current_atr) or current_atr <= 0:
                current_atr = 5.0 * pip_unit

            upper_band = current_bar["bb_upper"]
            lower_band = current_bar["bb_lower"]
            band_width = max(upper_band - lower_band, 1e-6)

            # Candlestick pattern
            sub_m1 = df_m1.iloc[i - 2:i + 1]
            is_bull_pattern, is_bear_pattern = self.strategy.detect_pinbar_engulfing(sub_m1)

            buy_score = 0.0
            sell_score = 0.0

            if htf_trend == "BULL":
                buy_score += 0.35
                if current_rsi < 40:
                    buy_score += 0.25
                elif current_rsi < 50:
                    buy_score += 0.10
                if current_close <= (lower_band + 0.25 * band_width):
                    buy_score += 0.25
                if is_bull_pattern:
                    buy_score += 0.15
                if adx_val > 25:
                    buy_score += 0.10
            elif htf_trend == "BEAR":
                sell_score += 0.35
                if current_rsi > 60:
                    sell_score += 0.25
                elif current_rsi > 50:
                    sell_score += 0.10
                if current_close >= (upper_band - 0.25 * band_width):
                    sell_score += 0.25
                if is_bear_pattern:
                    sell_score += 0.15
                if adx_val > 25:
                    sell_score += 0.10

            CONFIDENCE_THRESHOLD = 0.65
            decision = "HOLD"
            conf = 0.0

            if buy_score >= CONFIDENCE_THRESHOLD:
                decision = "BUY"
                conf = buy_score
            elif sell_score >= CONFIDENCE_THRESHOLD:
                decision = "SELL"
                conf = sell_score

            if decision == "HOLD":
                continue

            # Check directional cooldown
            if self.mode == "enhanced":
                is_dir_cooled, _, _ = rm.is_instrument_cooled_down(instrument, direction=decision, current_time=curr_time_sec)
                if is_dir_cooled:
                    continue

            # M1 Momentum Veto (Enhanced Mode)
            if self.mode == "enhanced":
                curr_body = current_bar["close"] - current_bar["open"]
                curr_range = max(current_bar["high"] - current_bar["low"], 1e-6)
                m1_ema_slope = (df_m1["m1_ema9"].iloc[i] - df_m1["m1_ema9"].iloc[i - 2]) / pip_unit if i >= 2 else 0.0

                if decision == "SELL":
                    strong_bull_bar = (curr_body > 0) and (current_bar["close"] >= current_bar["high"] - 0.25 * curr_range) and (curr_range > 0.4 * current_atr)
                    surging_up = m1_ema_slope > 1.2
                    no_bear_rev = (current_bar["close"] > prev_bar["close"]) and not is_bear_pattern
                    if strong_bull_bar or surging_up or no_bear_rev:
                        continue
                elif decision == "BUY":
                    strong_bear_bar = (curr_body < 0) and (current_bar["close"] <= current_bar["low"] + 0.25 * curr_range) and (curr_range > 0.4 * current_atr)
                    surging_down = m1_ema_slope < -1.2
                    no_bull_rev = (current_bar["close"] < prev_bar["close"]) and not is_bull_pattern
                    if strong_bear_bar or surging_down or no_bull_rev:
                        continue

            # Calculate SL / TP
            MIN_SL = 5.0 * pip_unit
            MAX_SL = 10.0 * pip_unit
            sl_dist = max(MIN_SL, min(MAX_SL, current_atr * 1.2))

            rr_multiplier = 2.0 if conf >= 0.85 else (1.75 if conf >= 0.75 else 1.5)
            tp_dist = sl_dist * rr_multiplier

            tp_pips = tp_dist / pip_unit
            atr_pips = current_atr / pip_unit

            # -------------------------------------------------------------
            # 4. Spread Gate Evaluation
            # -------------------------------------------------------------
            if self.mode == "enhanced":
                pair_floor = DEFAULT_PAIR_SPREADS.get(instrument, 1.4)
                tp_limit = tp_pips * 0.20
                max_spread = min(2.8, max(pair_floor, tp_limit))
            else:
                # Baseline 0.8p ATR clamp
                max_spread = min(2.5, max(0.8, atr_pips * 0.3))

            if simulated_spread > max_spread:
                continue

            # -------------------------------------------------------------
            # 5. Open Position (Deducting Spread from Entry)
            # -------------------------------------------------------------
            precision = price_precision(instrument)
            half_spread = (simulated_spread * pip_unit) / 2.0

            if decision == "BUY":
                fill_price = current_close + half_spread
                sl_price = fill_price - sl_dist
                tp_price = fill_price + tp_dist
            else:
                fill_price = current_close - half_spread
                sl_price = fill_price + sl_dist
                tp_price = fill_price - tp_dist

            active_trade = {
                "instrument": instrument,
                "direction": decision,
                "units": 10000,
                "opened_at": curr_time,
                "entry_price": round(fill_price, precision),
                "current_sl": round(sl_price, precision),
                "tp_price": round(tp_price, precision),
                "sl_dist_pips": round(sl_dist / pip_unit, 1),
                "tp_dist_pips": round(tp_pips, 1),
                "spread_pips": simulated_spread,
                "confidence": round(conf, 2)
            }

        return self._generate_report(instrument, trades, balance)

    def _generate_report(self, instrument, trades, final_balance):
        total_trades = len(trades)
        if total_trades == 0:
            return {
                "instrument": instrument,
                "mode": self.mode,
                "total_trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
                "net_pnl": 0.0,
                "return_pct": 0.0,
                "profit_factor": 0.0,
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "trades": []
            }

        wins = [t for t in trades if t["outcome"] == "WIN"]
        losses = [t for t in trades if t["outcome"] == "LOSS"]

        gross_profit = sum(t["realized_pl"] for t in wins)
        gross_loss = abs(sum(t["realized_pl"] for t in losses))

        profit_factor = round(gross_profit / gross_loss, 2) if gross_loss > 0 else (99.0 if gross_profit > 0 else 0.0)
        win_rate = round(len(wins) / total_trades * 100.0, 1)
        net_pnl = round(final_balance - self.initial_balance, 2)
        return_pct = round((net_pnl / self.initial_balance) * 100.0, 2)

        # Equity Curve and Max Drawdown Calculation
        equity = self.initial_balance
        peak = self.initial_balance
        max_dd = 0.0
        max_dd_pct = 0.0

        for t in trades:
            equity += t["realized_pl"]
            if equity > peak:
                peak = equity
            dd = peak - equity
            dd_pct = (dd / peak) * 100.0 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd
            if dd_pct > max_dd_pct:
                max_dd_pct = dd_pct

        avg_win = round(gross_profit / len(wins), 2) if wins else 0.0
        avg_loss = round(gross_loss / len(losses), 2) if losses else 0.0
        expectancy = round((win_rate / 100.0 * avg_win) - ((100.0 - win_rate) / 100.0 * avg_loss), 2)
        avg_duration = round(sum(t["duration_min"] for t in trades) / total_trades, 1)

        return {
            "instrument": instrument,
            "mode": self.mode,
            "total_trades": total_trades,
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": win_rate,
            "gross_profit": round(gross_profit, 2),
            "gross_loss": round(gross_loss, 2),
            "net_pnl": net_pnl,
            "return_pct": return_pct,
            "profit_factor": profit_factor,
            "max_drawdown": round(max_dd, 2),
            "max_drawdown_pct": round(max_dd_pct, 2),
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "expectancy": expectancy,
            "avg_duration_min": avg_duration,
            "trades": trades
        }


def print_comparison_table(results_baseline, results_enhanced):
    """Print clean, professional comparative results."""
    print("\n" + "=" * 80)
    print(f"{'METRIC':<28} | {'BASELINE (Legacy)':<22} | {'ENHANCED (New Architecture)':<24}")
    print("-" * 80)

    b = results_baseline
    e = results_enhanced

    rows = [
        ("Total Trades", f"{b['total_trades']}", f"{e['total_trades']}"),
        ("Win Rate (%)", f"{b['win_rate']}% ({b['wins']}W / {b['losses']}L)", f"{e['win_rate']}% ({e['wins']}W / {e['losses']}L)"),
        ("Net Realized PnL ($)", f"${b['net_pnl']:+.2f}", f"${e['net_pnl']:+.2f}"),
        ("Total Return (%)", f"{b['return_pct']:+.2f}%", f"{e['return_pct']:+.2f}%"),
        ("Profit Factor", f"{b['profit_factor']}", f"{e['profit_factor']}"),
        ("Max Drawdown ($)", f"${b['max_drawdown']:.2f} ({b['max_drawdown_pct']:.1f}%)", f"${e['max_drawdown']:.2f} ({e['max_drawdown_pct']:.1f}%)"),
        ("Average Win / Loss", f"+${b.get('avg_win', 0):.2f} / -${b.get('avg_loss', 0):.2f}", f"+${e.get('avg_win', 0):.2f} / -${e.get('avg_loss', 0):.2f}"),
        ("Expectancy per Trade", f"${b.get('expectancy', 0):+.2f}", f"${e.get('expectancy', 0):+.2f}"),
        ("Avg Duration (min)", f"{b.get('avg_duration_min', 0)} mins", f"{e.get('avg_duration_min', 0)} mins"),
    ]

    for label, val_b, val_e in rows:
        print(f"{label:<28} | {val_b:<22} | {val_e:<24}")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Forex Scalper High-Fidelity Historical Backtester")
    parser.add_argument("--pair", type=str, default="USD_JPY", help="Currency pair (e.g. USD_JPY, EUR_USD)")
    parser.add_argument("--count", type=int, default=5000, help="Number of candles to test (up to 5000)")
    parser.add_argument("--refresh", action="store_true", help="Force re-download candle data from OANDA")
    parser.add_argument("--compare", action="store_true", default=True, help="Compare Baseline vs Enhanced strategy")
    parser.add_argument("--spread", type=float, default=None, help="Override spread in pips (e.g. 1.5)")

    args = parser.parse_args()

    downloader = HistoricalDataDownloader()

    print(f"\n========================================================")
    print(f" Backtesting High-Precision Forex Scalper: {args.pair}")
    print(f" Candle Count: {args.count} | Spread: {args.spread or DEFAULT_PAIR_SPREADS.get(args.pair, 1.5)} pips")
    print(f"========================================================")

    t0 = time.time()
    df_m1 = downloader.get_candles(args.pair, granularity="M1", count=args.count, refresh=args.refresh)
    df_m5 = downloader.get_candles(args.pair, granularity="M5", count=args.count // 5 + 100, refresh=args.refresh)
    df_m15 = downloader.get_candles(args.pair, granularity="M15", count=args.count // 15 + 100, refresh=args.refresh)

    if df_m1.empty or df_m5.empty or df_m15.empty:
        print("[!] Failed to load required multi-timeframe candles. Aborting.")
        sys.exit(1)

    time_span_days = round((df_m1['time'].iloc[-1] - df_m1['time'].iloc[0]).total_seconds() / 86400, 1)
    print(f"[*] Loaded {len(df_m1)} M1 candles spanning {time_span_days} days ({df_m1['time'].iloc[0].strftime('%Y-%m-%d')} to {df_m1['time'].iloc[-1].strftime('%Y-%m-%d')}).")

    # Run simulations
    sim_enhanced = BacktestSimulator(mode="enhanced")
    res_enhanced = sim_enhanced.run(args.pair, df_m1, df_m5, df_m15, spread_pips=args.spread)

    if args.compare:
        sim_baseline = BacktestSimulator(mode="baseline")
        res_baseline = sim_baseline.run(args.pair, df_m1, df_m5, df_m15, spread_pips=args.spread)
        print_comparison_table(res_baseline, res_enhanced)
    else:
        print("\n--- Backtest Report (Enhanced Architecture) ---")
        print(json.dumps(res_enhanced, indent=2, default=str))

    elapsed = time.time() - t0
    print(f"[*] Backtest completed in {elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()
