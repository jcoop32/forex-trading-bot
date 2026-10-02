import pandas as pd
import numpy as np
import logging
from utils import pip_unit as get_pip_unit

class TechnicalScalpStrategy:
    """
    High-precision multi-timeframe technical scalper.
    Combines M15/M5 EMA Trend Filter with M1 RSI & Bollinger Band Pullback triggers.
    """
    def __init__(self):
        self.logger = logging.getLogger("trading_bot")

    def parse_candles(self, candles):
        """Convert OANDA candles list into a clean pandas DataFrame."""
        if not candles:
            return pd.DataFrame()
        
        data = []
        for c in candles:
            # mid price contains o, h, l, c
            data.append({
                'time': pd.to_datetime(c['time']),
                'open': float(c['mid']['o']),
                'high': float(c['mid']['h']),
                'low': float(c['mid']['l']),
                'close': float(c['mid']['c']),
                'volume': int(c['volume'])
            })
        df = pd.DataFrame(data)
        df.sort_values('time', inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def calculate_rsi(self, series, period=14):
        """Calculate Relative Strength Index (RSI)."""
        delta = series.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-9)
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df, period=14):
        """Calculate Average True Range (ATR)."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_bollinger_bands(self, series, period=20, std_dev=2.0):
        """Calculate Upper and Lower Bollinger Bands."""
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    def calculate_adx(self, df, period=14):
        """
        Calculate the Average Directional Index (ADX) to measure trend strength.
        Returns the latest ADX value, or 0.0 if insufficient data.
        """
        if len(df) < period * 2:
            return 0.0

        high = df['high']
        low = df['low']
        close = df['close']

        # Directional Movement
        plus_dm = high.diff()
        minus_dm = low.diff().apply(lambda x: -x)

        plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
        minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

        atr = self.calculate_atr(df, period)

        # Smoothed DI
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / (atr + 1e-9))
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / (atr + 1e-9))

        # DX and ADX
        dx = 100 * (np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9))
        adx = dx.rolling(window=period).mean()

        latest_adx = adx.iloc[-1]
        return float(latest_adx) if not pd.isna(latest_adx) else 0.0

    def detect_pinbar_engulfing(self, df):
        """
        Detect pinbar and engulfing candle patterns on the latest candle.
        Returns: (is_bullish_pattern, is_bearish_pattern)
        """
        if len(df) < 2:
            return False, False

        curr = df.iloc[-1]
        prev = df.iloc[-2]

        curr_body = abs(curr['close'] - curr['open'])
        curr_upper = curr['high'] - max(curr['open'], curr['close'])
        curr_lower = min(curr['open'], curr['close']) - curr['low']
        is_green = curr['close'] > curr['open']

        prev_body = abs(prev['close'] - prev['open'])
        prev_green = prev['close'] > prev['open']

        # Bullish Reversal: Hammer pinbar OR Bullish Engulfing
        bullish_pin = (curr_lower > 2 * curr_body) and (curr_upper < curr_body)
        bullish_engulf = is_green and (not prev_green) and (curr_body > prev_body)
        is_bullish = bullish_pin or bullish_engulf

        # Bearish Reversal: Shooting Star pinbar OR Bearish Engulfing
        bearish_pin = (curr_upper > 2 * curr_body) and (curr_lower < curr_body)
        bearish_engulf = (not is_green) and prev_green and (curr_body > prev_body)
        is_bearish = bearish_pin or bearish_engulf

        return is_bullish, is_bearish

    def evaluate(self, candles_m1, candles_m5, candles_m15, instrument):
        """
        Evaluate market condition across M1, M5, and M15 timeframes.
        Returns:
            decision (str): "BUY", "SELL", or "HOLD"
            confidence (float): 0.0 to 1.0
            current_atr (float): M1 ATR in price units
            sl_pips (float): Recommended SL distance in price units
            tp_pips (float): Recommended TP distance in price units
        """
        df_m1 = self.parse_candles(candles_m1)
        df_m5 = self.parse_candles(candles_m5)
        df_m15 = self.parse_candles(candles_m15)

        pip_unit = get_pip_unit(instrument)

        if len(df_m1) < 25 or len(df_m5) < 25:
            return "HOLD", 0.0, 0.0005, 0.0005, 0.0005

        # 1. Higher Timeframe Trend (M15 or M5)
        df_trend = df_m15 if len(df_m15) >= 25 else df_m5
        trend_ema_fast = df_trend['close'].ewm(span=9, adjust=False).mean()
        trend_ema_slow = df_trend['close'].ewm(span=21, adjust=False).mean()

        last_trend_fast = trend_ema_fast.iloc[-1]
        last_trend_slow = trend_ema_slow.iloc[-1]

        htf_trend = "BULL" if last_trend_fast > last_trend_slow else "BEAR"

        # 2. Lower Timeframe (M1) Indicators
        df_m1['rsi'] = self.calculate_rsi(df_m1['close'])
        df_m1['atr'] = self.calculate_atr(df_m1)
        bb_upper, bb_lower = self.calculate_bollinger_bands(df_m1['close'])

        last_m1 = df_m1.iloc[-1]
        current_close = last_m1['close']
        current_rsi = last_m1['rsi']
        current_atr = last_m1['atr']

        # Fallback if ATR is NaN
        if pd.isna(current_atr) or current_atr <= 0:
            current_atr = 5.0 * pip_unit

        upper_band = bb_upper.iloc[-1]
        lower_band = bb_lower.iloc[-1]
        band_width = upper_band - lower_band

        # 3. ADX Trend Strength Gate (computed on M5 for less noise)
        adx_value = self.calculate_adx(df_m5)
        if adx_value < 15:
            self.logger.info(f"[{instrument}] ADX {adx_value:.1f} < 15 (ranging market). Forcing HOLD.")
            return "HOLD", 0.0, current_atr, 5.0 * pip_unit, 5.0 * pip_unit

        # 4. Candlestick Pattern Trigger
        is_bull_pattern, is_bear_pattern = self.detect_pinbar_engulfing(df_m1)

        # 5. Volume Confirmation (M1)
        vol_sma = df_m1['volume'].rolling(window=20).mean()
        current_volume = df_m1['volume'].iloc[-1]
        avg_volume = vol_sma.iloc[-1] if not pd.isna(vol_sma.iloc[-1]) else current_volume
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0

        # 6. Confluence Scoring
        # We need alignment between higher timeframe trend and M1 oversold/overbought pullback
        buy_score = 0.0
        sell_score = 0.0

        if htf_trend == "BULL":
            buy_score += 0.35
            # RSI Pullback in uptrend
            if current_rsi < 40:
                buy_score += 0.25
            elif current_rsi < 50:
                buy_score += 0.10
            # Bollinger Band lower bounce
            if current_close <= (lower_band + 0.25 * band_width):
                buy_score += 0.25
            # Pattern trigger
            if is_bull_pattern:
                buy_score += 0.15
            # ADX trending bonus
            if adx_value > 25:
                buy_score += 0.10

        elif htf_trend == "BEAR":
            sell_score += 0.35
            # RSI Pullback in downtrend
            if current_rsi > 60:
                sell_score += 0.25
            elif current_rsi > 50:
                sell_score += 0.10
            # Bollinger Band upper rejection
            if current_close >= (upper_band - 0.25 * band_width):
                sell_score += 0.25
            # Pattern trigger
            if is_bear_pattern:
                sell_score += 0.15
            # ADX trending bonus
            if adx_value > 25:
                sell_score += 0.10

        # Volume modifier (applied to whichever direction is active)
        if volume_ratio > 1.5:
            buy_score += 0.10 if htf_trend == "BULL" else 0.0
            sell_score += 0.10 if htf_trend == "BEAR" else 0.0
        elif volume_ratio < 0.5:
            buy_score -= 0.10 if htf_trend == "BULL" else 0.0
            sell_score -= 0.10 if htf_trend == "BEAR" else 0.0

        # 5. Dynamic SL/TP Distances (Clamped to 5.0 - 10.0 pips for scalping)
        MIN_SL_PIPS = 5.0 * pip_unit
        MAX_SL_PIPS = 10.0 * pip_unit

        raw_sl = current_atr * 1.2
        sl_pips = max(MIN_SL_PIPS, min(MAX_SL_PIPS, raw_sl))

        # Execution Threshold
        CONFIDENCE_THRESHOLD = 0.65

        # Dynamic R:R scaling based on confluence strength
        top_score = max(buy_score, sell_score)
        if top_score >= 0.85:
            rr_multiplier = 2.0
        elif top_score >= 0.75:
            rr_multiplier = 1.75
        else:
            rr_multiplier = 1.5

        tp_pips = sl_pips * rr_multiplier

        # 6. Lower-Timeframe (M1) Momentum Veto
        # Prevent shorting into an active upward surge, or buying into an active downward dump.
        curr_bar = df_m1.iloc[-1]
        prev_bar = df_m1.iloc[-2]
        curr_body = curr_bar['close'] - curr_bar['open']
        curr_range = max(curr_bar['high'] - curr_bar['low'], 1e-6)

        m1_ema9 = df_m1['close'].ewm(span=9, adjust=False).mean()
        m1_ema_slope = (m1_ema9.iloc[-1] - m1_ema9.iloc[-3]) / pip_unit if len(m1_ema9) >= 3 else 0.0

        if sell_score >= CONFIDENCE_THRESHOLD:
            # Veto SELL if strong green bar closing at highs or sharp upward EMA slope
            strong_bull_bar = (curr_body > 0) and (curr_bar['close'] >= curr_bar['high'] - 0.25 * curr_range) and (curr_range > 0.4 * current_atr)
            surging_up = m1_ema_slope > 1.2
            no_bearish_reversal = (curr_bar['close'] > prev_bar['close']) and not is_bear_pattern

            if strong_bull_bar or surging_up or no_bearish_reversal:
                self.logger.info(
                    f"[{instrument}] M1 Momentum VETO on SELL (Slope: {m1_ema_slope:+.1f}p, Bull bar: {strong_bull_bar}, No rev: {no_bearish_reversal}). Forcing HOLD."
                )
                return "HOLD", sell_score, current_atr, sl_pips, tp_pips

            self.logger.info(f"[{instrument}] Strong SELL Signal (Conf: {sell_score:.2f}, R:R 1:{rr_multiplier}) | RSI: {current_rsi:.1f} | Trend: BEAR")
            return "SELL", sell_score, current_atr, sl_pips, tp_pips

        elif buy_score >= CONFIDENCE_THRESHOLD:
            # Veto BUY if strong red bar closing at lows or sharp downward EMA slope
            strong_bear_bar = (curr_body < 0) and (curr_bar['close'] <= curr_bar['low'] + 0.25 * curr_range) and (curr_range > 0.4 * current_atr)
            surging_down = m1_ema_slope < -1.2
            no_bullish_reversal = (curr_bar['close'] < prev_bar['close']) and not is_bull_pattern

            if strong_bear_bar or surging_down or no_bullish_reversal:
                self.logger.info(
                    f"[{instrument}] M1 Momentum VETO on BUY (Slope: {m1_ema_slope:+.1f}p, Bear bar: {strong_bear_bar}, No rev: {no_bullish_reversal}). Forcing HOLD."
                )
                return "HOLD", buy_score, current_atr, sl_pips, tp_pips

            self.logger.info(f"[{instrument}] Strong BUY Signal (Conf: {buy_score:.2f}, R:R 1:{rr_multiplier}) | RSI: {current_rsi:.1f} | Trend: BULL")
            return "BUY", buy_score, current_atr, sl_pips, tp_pips
        else:
            return "HOLD", top_score, current_atr, sl_pips, tp_pips

