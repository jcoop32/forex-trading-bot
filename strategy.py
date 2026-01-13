import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import logging

class MLStrategy:
    def __init__(self):
        # Dictionary to store models for each instrument: {"EUR_USD": model, ...}
        self.models = {}
        self.logger = logging.getLogger("trading_bot")

    def parse_candles(self, candles):
        """Helper to parse OANDA candles into a DataFrame."""
        if not candles:
            return pd.DataFrame()
        
        data = []
        for c in candles:
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
        return df

    def calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df, period=14):
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        return true_range.rolling(window=period).mean()

    def calculate_bollinger_bands(self, series, period=20, std_dev=2):
        sma = series.rolling(window=period).mean()
        std = series.rolling(window=period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower

    def prepare_single_frame(self, df, prefix=""):
        """Generate features for a single timeframe."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Basic Features
        df[f'{prefix}returns'] = df['close'].pct_change()
        df[f'{prefix}range'] = df['high'] - df['low']
        df[f'{prefix}volatility'] = df[f'{prefix}range'] / df['open']
        
        # RSI
        df[f'{prefix}rsi'] = self.calculate_rsi(df['close'])
        
        # ATR (Absolute pips? No, price diff. Useful for stops)
        # We need this for the Risk Manager mainly, effectively standardizing volatility
        # but also as a feature (normalized ATR)
        df[f'{prefix}atr'] = self.calculate_atr(df)
        df[f'{prefix}atr_norm'] = df[f'{prefix}atr'] / df['close'] # Normalized for ML
        
        # Bollinger Bands
        bb_up, bb_low = self.calculate_bollinger_bands(df['close'])
        # Distance from bands (0 to 1 scaling approx? or just diff)
        df[f'{prefix}bb_width'] = (bb_up - bb_low) / df['close']
        df[f'{prefix}bb_pos'] = (df['close'] - bb_low) / (bb_up - bb_low) # 0 = low, 1 = up
        
        # Lagged features
        df[f'{prefix}returns_lag1'] = df[f'{prefix}returns'].shift(1)
        df[f'{prefix}volatility_lag1'] = df[f'{prefix}volatility'].shift(1)
        
        # Trend
        df[f'{prefix}sma_20'] = df['close'].rolling(window=20).mean()
        df[f'{prefix}trend_signal'] = np.where(df['close'] > df[f'{prefix}sma_20'], 1, -1)
        
        return df

    def prepare_data(self, candles_m1, candles_m5, candles_m15):
        """
        Convert M1, M5, M15 candles to DataFrame and generate merged features.
        """
        df_m1 = self.parse_candles(candles_m1)
        df_m5 = self.parse_candles(candles_m5)
        df_m15 = self.parse_candles(candles_m15)
        
        if df_m1.empty or df_m5.empty or df_m15.empty:
            return pd.DataFrame()
            
        # Feature Engineering by Timeframe
        df_m1 = self.prepare_single_frame(df_m1, prefix="")
        df_m5 = self.prepare_single_frame(df_m5, prefix="m5_")
        df_m15 = self.prepare_single_frame(df_m15, prefix="m15_")
        
        # Select columns to keep for higher timeframes
        cols_m5 = ['time', 'm5_returns', 'm5_rsi', 'm5_atr_norm', 'm5_bb_pos', 'm5_trend_signal']
        cols_m15 = ['time', 'm15_returns', 'm15_rsi', 'm15_atr_norm', 'm15_bb_pos', 'm15_trend_signal']
        
        # Merge
        msg_df = pd.merge_asof(df_m1, df_m5[cols_m5], on='time', direction='backward')
        final_df = pd.merge_asof(msg_df, df_m15[cols_m15], on='time', direction='backward')
        
        # Target: 1 if next M1 candle closes higher, else 0
        final_df['target'] = (final_df['close'].shift(-5) > final_df['close']).astype(int)
        
        # Drop columns that have NaNs (due to rolling windows)
        final_df.dropna(inplace=True)
        
        return final_df

    def train(self, candles_m1, candles_m5, candles_m15, instrument):
        """Train the model using multi-timeframe data."""
        self.logger.info(f"[{instrument}] Preparing Training Data (M1/M5/M15)...")
        df = self.prepare_data(candles_m1, candles_m5, candles_m15)
        
        if len(df) < 100:
            self.logger.warning(f"[{instrument}] Not enough data to train strategy (Sample: {len(df)}).")
            return
            
        # Define Features
        features = [
            'returns', 'volatility', 'rsi', 'atr_norm', 'bb_pos', 
            'returns_lag1', 'volatility_lag1',
            'm5_returns', 'm5_rsi', 'm5_atr_norm', 'm5_bb_pos', 'm5_trend_signal',
            'm15_returns', 'm15_rsi', 'm15_atr_norm', 'm15_trend_signal'
        ]
        
        X = df[features]
        y = df['target']
        
        model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        model.fit(X, y)
        self.models[instrument] = model
        self.logger.info(f"[{instrument}] ML Strategy trained successfully. Rows: {len(df)}")

    def predict(self, recent_candles_m1, recent_candles_m5, recent_candles_m15, instrument):
        """
        Predict direction for the NEXT M1 candle.
        Returns: prediction (0/1), confidence (float), current_atr (float)
        """
        if instrument not in self.models:
            self.logger.warning(f"[{instrument}] Model not trained.")
            return 0, 0.0, 0.0
            
        model = self.models[instrument]
            
        # 1. Parse & Prep
        df_m1 = self.parse_candles(recent_candles_m1)
        df_m5 = self.parse_candles(recent_candles_m5)
        df_m15 = self.parse_candles(recent_candles_m15)
        
        if df_m1.empty or df_m5.empty or df_m15.empty:
             return 0, 0.0, 0.0

        # Feature Eng
        df_m1 = self.prepare_single_frame(df_m1, prefix="")
        df_m5 = self.prepare_single_frame(df_m5, prefix="m5_")
        df_m15 = self.prepare_single_frame(df_m15, prefix="m15_")
        
        # Merge
        cols_m5 = ['time', 'm5_returns', 'm5_rsi', 'm5_atr_norm', 'm5_bb_pos', 'm5_trend_signal']
        cols_m15 = ['time', 'm15_returns', 'm15_rsi', 'm15_atr_norm', 'm15_bb_pos', 'm15_trend_signal']
        
        msg_df = pd.merge_asof(df_m1, df_m5[cols_m5], on='time', direction='backward')
        final_df = pd.merge_asof(msg_df, df_m15[cols_m15], on='time', direction='backward')
        
        # Get last row (latest completed timeframe data)
        last_row = final_df.iloc[-1:]
        
        features = [
            'returns', 'volatility', 'rsi', 'atr_norm', 'bb_pos', 
            'returns_lag1', 'volatility_lag1',
            'm5_returns', 'm5_rsi', 'm5_atr_norm', 'm5_bb_pos', 'm5_trend_signal',
            'm15_returns', 'm15_rsi', 'm15_atr_norm', 'm15_trend_signal'
        ]
        
        # Check for NaNs
        if last_row[features].isnull().values.any():
            self.logger.warning(f"[{instrument}] Latest data contains NaNs (likely insufficient history for Lags/SMA). Cannot predict.")
            # Fallback
            return 0, 0.0, 0.0
            
        prediction = model.predict(last_row[features])[0]
        prob_up = model.predict_proba(last_row[features])[0][1]
        
        confidence = prob_up if prediction == 1 else (1.0 - prob_up)
        
        # Extract latest ATR for Risk Manager (M1 ATR)
        current_atr = last_row['atr'].values[0]
        
        self.logger.info(f"[{instrument}] Pred: {prediction} (Conf: {confidence:.2f}) | ATR: {current_atr:.5f} | RSI: {last_row['rsi'].values[0]:.1f}")
        
        return prediction, confidence, current_atr
