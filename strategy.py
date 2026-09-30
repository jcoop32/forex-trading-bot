import pandas as pd
import numpy as np
import logging

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

        pip_unit = 0.01 if "JPY" in instrument else 0.0001

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

        # 3. Candlestick Pattern Trigger
        is_bull_pattern, is_bear_pattern = self.detect_pinbar_engulfing(df_m1)

        # 4. Confluence Scoring
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

        # 5. Dynamic SL/TP Distances (Clamped to 5.0 - 8.0 pips for scalping)
        MIN_SL_PIPS = 5.0 * pip_unit
        MAX_SL_PIPS = 8.0 * pip_unit

        raw_sl = current_atr * 1.2
        sl_pips = max(MIN_SL_PIPS, min(MAX_SL_PIPS, raw_sl))
        # 1.0 to 1.2 Risk:Reward for fast, high-probability scalping
        tp_pips = sl_pips * 1.1

        # Execution Threshold
        CONFIDENCE_THRESHOLD = 0.65

        if buy_score >= CONFIDENCE_THRESHOLD:
            self.logger.info(f"[{instrument}] Strong BUY Signal (Conf: {buy_score:.2f}) | RSI: {current_rsi:.1f} | Trend: BULL")
            return "BUY", buy_score, current_atr, sl_pips, tp_pips
        elif sell_score >= CONFIDENCE_THRESHOLD:
            self.logger.info(f"[{instrument}] Strong SELL Signal (Conf: {sell_score:.2f}) | RSI: {current_rsi:.1f} | Trend: BEAR")
            return "SELL", sell_score, current_atr, sl_pips, tp_pips
        else:
            return "HOLD", max(buy_score, sell_score), current_atr, sl_pips, tp_pips
