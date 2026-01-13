import logging
import math

class RiskManager:
    def __init__(self):
        self.logger = logging.getLogger("trading_bot")
        
        # Risk Configuration
        self.BASE_RISK_PCT = 0.010  # 1.0%
        self.MAX_RISK_PCT = 0.030   # 3.0%
        
        # Boosters
        self.CONFIDENCE_BOOST_LOW = 0.005  # +0.5% if prob > 60%
        self.CONFIDENCE_BOOST_HIGH = 0.010 # +1.0% if prob > 70%
        self.SENTIMENT_BOOST = 0.005       # +0.5% if sentiment agrees

    def calculate_risk_percentage(self, ml_probability, sentiment_signal, trade_direction, news_score=0.0):
        """
        Determine dynamic risk percentage based on conviction.
        
        Args:
            ml_probability (float): Model confidence (0.0 to 1.0)
            sentiment_signal (str): 'BUY', 'SELL', or 'NEUTRAL'
            trade_direction (str): 'BUY' or 'SELL'
            news_score (float): -1.0 to 1.0 (Gemini Score)
            
        Returns:
            float: Risk percentage (e.g., 0.015 for 1.5%)
        """
        risk_pct = self.BASE_RISK_PCT
        
        # 1. ML Confidence Booster
        if ml_probability > 0.70:
            risk_pct += self.CONFIDENCE_BOOST_HIGH
            self.logger.info(f"High Confidence ({ml_probability:.2%}) -> Risk Bost +{self.CONFIDENCE_BOOST_HIGH:.1%}")
        elif ml_probability > 0.60:
            risk_pct += self.CONFIDENCE_BOOST_LOW
            self.logger.info(f"Moderate Confidence ({ml_probability:.2%}) -> Risk Bost +{self.CONFIDENCE_BOOST_LOW:.1%}")

        # 2. Sentiment Booster
        if sentiment_signal == trade_direction:
            risk_pct += self.SENTIMENT_BOOST
            self.logger.info(f"Sentiment Confluence ({sentiment_signal}) -> Risk Bost +{self.SENTIMENT_BOOST:.1%}")
            
        # 3. News Sentiment Booster (Gemini)
        # If score > 0.25 (Bullish) and we are Buying -> Boost
        # If score < -0.25 (Bearish) and we are Selling -> Boost
        
        NEWS_THRESHOLD = 0.25
        NEWS_BOOST = 0.005 # +0.5%
        
        if trade_direction == "BUY" and news_score > NEWS_THRESHOLD:
            risk_pct += NEWS_BOOST
            self.logger.info(f"News Support (Score {news_score:.2f}) -> Risk Bost +{NEWS_BOOST:.1%}")
        elif trade_direction == "SELL" and news_score < -NEWS_THRESHOLD:
            risk_pct += NEWS_BOOST
            self.logger.info(f"News Support (Score {news_score:.2f}) -> Risk Bost +{NEWS_BOOST:.1%}")
            
        # 4. Cap Risk
        final_risk = min(risk_pct, self.MAX_RISK_PCT)
        
        self.logger.info(f"Final Risk Percentage: {final_risk:.2%}")
        return final_risk

    def calculate_position_size(self, account_balance, margin_available, risk_percentage, stop_loss_pips, current_price, pair="EUR_USD"):
        """
        Calculate position size (units) based on risk amount.
        """
        risk_amount = account_balance * risk_percentage
        
        if stop_loss_pips <= 0:
            return 0
            
        # 1. Standard Case: Quote Currency is Account Currency (e.g. EUR_USD for USD account)
        if pair.endswith("_USD"):
             units = risk_amount / stop_loss_pips
             
        # 2. JPY Case: Base Currency is Account Currency (e.g. USD_JPY for USD account)
        elif pair.startswith("USD_"):
             if current_price and current_price > 0:
                 units = (risk_amount * current_price) / stop_loss_pips
             else:
                 self.logger.warning(f"Invalid price for {pair}, defaulting to safe size 0.")
                 return 0
        else:
             # Fallback/Crosses
             self.logger.warning(f"Complex pair {pair} sizing. Using standard formula (approx).")
             units = risk_amount / stop_loss_pips
        
        # --- MARGIN SAFETY CHECK ---
        # Cap units to avoid insufficient margin. 
        # Use margin_available instead of full balance.
        # Apply 95% buffer to be safe against spread/price fluctuations.
        
        MAX_LEVERAGE = 20
        # Effectively, how many dollars worth of position can we buy with remaining margin?
        # Max Nominal = MarginAvailable * Leverage * 0.95
        max_nominal_value = margin_available * MAX_LEVERAGE * 0.95
        
        # Calculate current nominal value of the proposed units
        current_nominal_value = 0.0
        if pair.endswith("_USD"): # Base is NOT USD, e.g. EUR_USD
            current_nominal_value = units * current_price
        elif pair.startswith("USD_"): # Base IS USD
            current_nominal_value = units
        else:
            current_nominal_value = units * current_price # Approx
            
        if current_nominal_value > max_nominal_value:
            self.logger.warning(f"Calculated units {int(units)} exceeds available margin cap (Limit: {int(max_nominal_value)}). Capping.")
            
            # Reverse calc units from max_nominal_value
            if pair.endswith("_USD"):
                units = max_nominal_value / current_price
            else:
                units = max_nominal_value
        
        units = int(units)
        self.logger.info(f"Risking ${risk_amount:.2f} (Bal: ${account_balance:.2f}, Free: ${margin_available:.2f}) -> {units} Units")
        return units
