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

    def prepare_single_frame(self, df, prefix=""):
        """Generate basic features for a single timeframe."""
        if df.empty:
            return df
            
        df = df.copy()
        
        # Basic Features
        df[f'{prefix}returns'] = df['close'].pct_change()
        df[f'{prefix}range'] = df['high'] - df['low']
        df[f'{prefix}volatility'] = df[f'{prefix}range'] / df['open']
        
        # Lagged features
        df[f'{prefix}returns_lag1'] = df[f'{prefix}returns'].shift(1)
        df[f'{prefix}returns_lag2'] = df[f'{prefix}returns'].shift(2)
        df[f'{prefix}volatility_lag1'] = df[f'{prefix}volatility'].shift(1)
        
        # Simple SMA Trend (e.g. 20 period)
        df[f'{prefix}sma_20'] = df['close'].rolling(window=20).mean()
        df[f'{prefix}trend_signal'] = np.where(df['close'] > df[f'{prefix}sma_20'], 1, -1)
        
        # Drop NaN caused by shift/rolling
        # We don't drop here yet to preserve timestamp alignment potential until merge? 
        # Actually safer to drop only used columns or fill NaNs. 
        # For simplicity, we'll keep NaNs and drop after merge.
        
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
        # We want trend and volatility context
        cols_m5 = ['time', 'm5_returns', 'm5_volatility', 'm5_trend_signal']
        cols_m15 = ['time', 'm15_returns', 'm15_volatility', 'm15_trend_signal']
        
        # Merge M5 onto M1 (Backward search: for each M1, find latest M5)
        # Note: pd.merge_asof requires sorted 'on' column
        msg_df = pd.merge_asof(df_m1, df_m5[cols_m5], on='time', direction='backward')
        
        # Merge M15 onto result
        final_df = pd.merge_asof(msg_df, df_m15[cols_m15], on='time', direction='backward')
        
        # Target: 1 if next M1 candle closes higher, else 0
        final_df['target'] = (final_df['close'].shift(-1) > final_df['close']).astype(int)
        
        # Drop columns that have NaNs (start of data)
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
            'returns', 'range', 'volatility', 'returns_lag1', 'returns_lag2', 'volatility_lag1',
            'm5_returns', 'm5_volatility', 'm5_trend_signal',
            'm15_returns', 'm15_volatility', 'm15_trend_signal'
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
        """
        if instrument not in self.models:
            self.logger.warning(f"[{instrument}] Model not trained.")
            return 0, 0.0
            
        model = self.models[instrument]
            
        # We need to reconstruct the dataframe structure just like training
        # But we want the features for the *latest completed candle* to predict the *next* move.
        
        # 1. Parse & Prep
        df_m1 = self.parse_candles(recent_candles_m1)
        df_m5 = self.parse_candles(recent_candles_m5)
        df_m15 = self.parse_candles(recent_candles_m15)
        
        if df_m1.empty or df_m5.empty or df_m15.empty:
             return 0, 0.0

        # Feature Eng
        df_m1 = self.prepare_single_frame(df_m1, prefix="")
        df_m5 = self.prepare_single_frame(df_m5, prefix="m5_")
        df_m15 = self.prepare_single_frame(df_m15, prefix="m15_")
        
        # Merge
        cols_m5 = ['time', 'm5_returns', 'm5_volatility', 'm5_trend_signal']
        cols_m15 = ['time', 'm15_returns', 'm15_volatility', 'm15_trend_signal']
        
        msg_df = pd.merge_asof(df_m1, df_m5[cols_m5], on='time', direction='backward')
        final_df = pd.merge_asof(msg_df, df_m15[cols_m15], on='time', direction='backward')
        
        # Get last row (latest completed timeframe data)
        last_row = final_df.iloc[-1:]
        
        features = [
            'returns', 'range', 'volatility', 'returns_lag1', 'returns_lag2', 'volatility_lag1',
            'm5_returns', 'm5_volatility', 'm5_trend_signal',
            'm15_returns', 'm15_volatility', 'm15_trend_signal'
        ]
        
        # Check for NaNs
        if last_row[features].isnull().values.any():
            self.logger.warning(f"[{instrument}] Latest data contains NaNs (likely insufficient history for Lags/SMA). Cannot predict.")
            # Fallback: could return 0 or try to use previous...
            return 0, 0.0
            
        prediction = model.predict(last_row[features])[0]
        prob_up = model.predict_proba(last_row[features])[0][1]
        
        confidence = prob_up if prediction == 1 else (1.0 - prob_up)
        
        self.logger.info(f"[{instrument}] Pred: {prediction} (Conf: {confidence:.2f}) | Trend M5:{last_row['m5_trend_signal'].values[0]} M15:{last_row['m15_trend_signal'].values[0]}")
        
        return prediction, confidence
