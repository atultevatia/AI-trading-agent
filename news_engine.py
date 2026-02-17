import feedparser
import time
from urllib.parse import quote
from datetime import datetime, timedelta

class NewsCache:
    def __init__(self, ttl_seconds=3600):
        self.cache = {}
        self.ttl = ttl_seconds

    def get(self, key):
        if key in self.cache:
            entry, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return entry
            else:
                del self.cache[key]
        return None

    def set(self, key, value):
        self.cache[key] = (value, time.time())

class NewsEngine:
    def __init__(self):
        self.cache = NewsCache()

    def get_stock_news(self, ticker: str):
        # Check cache
        cached_news = self.cache.get(ticker)
        if cached_news:
            return cached_news

        try:
            # Clean ticker for NSE (e.g., RELIANCE.NS -> RELIANCE)
            query_ticker = ticker.split('.')[0]
            # Use Google News RSS
            rss_url = f"https://news.google.com/rss/search?q={quote(query_ticker)}+stock+news+India&hl=en-IN&gl=IN&ceid=IN:en"
            
            feed = feedparser.parse(rss_url)
            headlines = []
            seen = set()

            for entry in feed.entries[:10]: # Top 10 headlines
                title = entry.title
                # Deduplicate and basic cleaning
                if title not in seen:
                    headlines.append({
                        "title": title,
                        "link": entry.link,
                        "published": entry.published
                    })
                    seen.add(title)

            # Fallback if no news found specifically with "stock news"
            if not headlines:
                 rss_url = f"https://news.google.com/rss/search?q={quote(query_ticker)}&hl=en-IN&gl=IN&ceid=IN:en"
                 feed = feedparser.parse(rss_url)
                 for entry in feed.entries[:5]:
                    if entry.title not in seen:
                        headlines.append({
                            "title": entry.title,
                            "link": entry.link,
                            "published": entry.published
                        })
                        seen.add(entry.title)

            self.cache.set(ticker, headlines)
            return headlines

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []

# Singleton instance
news_engine = NewsEngine()
