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
            query_ticker = ticker.split('.')[0]
            headlines = []
            seen = set()

            # Sources with relative authenticity scores
            sources = [
                {"name": "NSE Announcements", "site": "nseindia.com", "score": 1.0},
                {"name": "Reuters India", "site": "reuters.com", "score": 0.9},
                {"name": "Moneycontrol", "site": "moneycontrol.com", "score": 0.85},
                {"name": "Google Finance", "site": "", "score": 0.6} # Generic fallback
            ]

            for src in sources:
                if src['site']:
                    q = f"site:{src['site']} {query_ticker}"
                else:
                    q = f"{query_ticker} stock news India"
                
                rss_url = f"https://news.google.com/rss/search?q={quote(q)}&hl=en-IN&gl=IN&ceid=IN:en"
                feed = feedparser.parse(rss_url)
                
                for entry in feed.entries[:5]: # Take top 5 from each source
                    title = entry.title
                    if title not in seen:
                        headlines.append({
                            "title": title,
                            "link": entry.link,
                            "published": entry.published,
                            "source": src['name'],
                            "authenticity": src['score']
                        })
                        seen.add(title)

            # Sort by authenticity and recency (basic approximation)
            headlines.sort(key=lambda x: x['authenticity'], reverse=True)
            
            self.cache.set(ticker, headlines)
            return headlines

        except Exception as e:
            print(f"Error fetching news for {ticker}: {e}")
            return []

# Singleton instance
news_engine = NewsEngine()
