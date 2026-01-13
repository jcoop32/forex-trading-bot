import logging
import os
import feedparser
import google.generativeai as genai
from datetime import datetime, timedelta

class NewsAgent:
    def __init__(self):
        self.logger = logging.getLogger("trading_bot")
        self.api_key = os.getenv("GEMINI_API_KEY")
        self.is_active = False

        if self.api_key:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                self.is_active = True
                self.logger.info("NewsAgent: Active (Gemini Configured)")
            except Exception as e:
                self.logger.error(f"NewsAgent: Failed to configure Gemini: {e}")
        else:
            self.logger.warning("NewsAgent: Inactive (Missing GEMINI_API_KEY)")

        self.last_fetch_time = None
        self.cached_score = 0.0
        self.CACHE_DURATION_MINUTES = 15

    def fetch_news(self, instrument="EUR_USD"):
        """
        Fetch news from Google News RSS.
        """
        # Google News RSS for standardized query
        # "EUR USD" -> "EUR%20USD"
        query = instrument.replace("_", "%20") + "%20Forex"
        url = f"https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
        
        try:
            feed = feedparser.parse(url)
            # Get top 5 headlines
            headlines = [entry.title for entry in feed.entries[:5]]
            return headlines
        except Exception as e:
            self.logger.error(f"NewsAgent: RSS Fetch Error: {e}")
            return []

    def get_sentiment_score(self, instrument="EUR_USD"):
        """
        Get sentiment score from -1.0 to 1.0.
        Uses caching to avoid API spam.
        """
        if not self.is_active:
            return 0.0

        now = datetime.now()
        if self.last_fetch_time and (now - self.last_fetch_time) < timedelta(minutes=self.CACHE_DURATION_MINUTES):
            return self.cached_score

        headlines = self.fetch_news(instrument)
        if not headlines:
            return 0.0

        score = self._analyze_with_gemini(headlines, instrument)
        
        self.last_fetch_time = now
        self.cached_score = score
        return score

    def _analyze_with_gemini(self, headlines, instrument):
        """
        Send headlines to Gemini for scoring.
        """
        prompt = f"""
        Analyze the sentiment of the following news headlines for the Forex pair {instrument}.
        Headlines:
        {chr(10).join(['- ' + h for h in headlines])}

        Determine if the news is Bullish (positive for the pair), Bearish (negative), or Neutral.
        Return ONLY a single float score between -1.0 (Extremely Bearish) and 1.0 (Extremely Bullish).
        0.0 is Neutral.
        Example output:
        0.45
        """
        
        import re
        try:
            response = self.model.generate_content(prompt)
            text = response.text.strip()
            
            # Extract the last floating point number from the text
            # This handles cases where the model explains itself and puts the score at the end.
            # Regex looks for a number (int or float, possibly negative)
            matches = re.findall(r"[-+]?\d*\.\d+|[-+]?\d+", text)
            
            if matches:
                 # Take the last number found, assuming it's the final score
                score = float(matches[-1])
            else:
                self.logger.warning(f"NewsAgent: Could not extract score from: {text[:100]}...")
                score = 0.0

            # Clamp just in case
            score = max(-1.0, min(1.0, score))
            
            self.logger.info(f"NewsAgent: Gemini Score: {score:.2f} (from {len(headlines)} headlines)")
            return score
        except Exception as e:
            self.logger.error(f"NewsAgent: Gemini Analysis Error: {e}")
            return 0.0
