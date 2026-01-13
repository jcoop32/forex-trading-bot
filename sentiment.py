import logging
import oandapyV20.endpoints.instruments as instruments

class SentimentAnalyzer:
    def __init__(self, api_connection):
        self.api = api_connection.api
        self.logger = logging.getLogger("trading_bot")

    def get_market_sentiment(self, instrument):
        """
        Analyzes the Position Book to determine market sentiment.
        Returns: 'BUY', 'SELL', or 'NEUTRAL'
        
        Logic:
        - Fetch Position Book.
        - Sum longCountPercent and shortCountPercent.
        - If Long % > 80% -> Crowd is Long -> Contrarian SELL.
        - If Short % > 80% -> Crowd is Short -> Contrarian BUY.
        """
        r = instruments.InstrumentsPositionBook(instrument=instrument)
        try:
            self.api.request(r)
            bucket_data = r.response.get("positionBook", {}).get("buckets", [])
            
            total_long_pct = sum(float(b.get('longCountPercent', 0)) for b in bucket_data)
            total_short_pct = sum(float(b.get('shortCountPercent', 0)) for b in bucket_data)
            
            # Normalize just in case they don't sum to exactly 100 due to API precision
            total = total_long_pct + total_short_pct
            if total == 0:
                return "NEUTRAL"
                
            long_ratio = total_long_pct / total
            short_ratio = total_short_pct / total
            
            self.logger.info(f"Sentiment for {instrument}: Long {long_ratio:.2%}, Short {short_ratio:.2%}")

            if long_ratio > 0.80:
                return "SELL"
            elif short_ratio > 0.80:
                return "BUY"
            else:
                return "NEUTRAL"
                
        except Exception as e:
            self.logger.error(f"Error fetching sentiment: {e}")
            return "NEUTRAL"
