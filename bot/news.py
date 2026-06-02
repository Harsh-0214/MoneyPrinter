"""News fetching and sentiment scoring via NewsAPI, RSS feeds, and SEC EDGAR."""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import requests
from textblob import TextBlob

logger = logging.getLogger(__name__)

RSS_FEEDS = [
    "https://feeds.reuters.com/reuters/businessNews",
    "https://finance.yahoo.com/rss/",
    "https://www.benzinga.com/feed",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://seekingalpha.com/feed.xml",
]

SEC_8K_FEED = "https://www.sec.gov/cgi-bin/browse-edgar?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=&output=atom"


def _sentiment(text: str) -> float:
    """TextBlob polarity: -1.0 to +1.0."""
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


def _fetch_newsapi(ticker: str, company_name: str, api_key: str) -> list[dict]:
    """Fetch headlines from NewsAPI.org."""
    if not api_key:
        return []
    headlines = []
    queries = [f"{ticker} stock", company_name]
    for q in queries:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": q,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 10,
                "from": (datetime.now() - timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%S"),
                "apiKey": api_key,
            }
            resp = requests.get(url, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                for art in data.get("articles", []):
                    title = art.get("title") or ""
                    desc  = art.get("description") or ""
                    text  = f"{title}. {desc}".strip()
                    if text and text != ".":
                        headlines.append({"text": text, "source": "newsapi"})
        except Exception as e:
            logger.warning(f"[news] NewsAPI failed for {ticker}: {e}")
    return headlines[:10]


def _fetch_rss(ticker: str, company_name: str) -> list[dict]:
    """Fetch headlines from RSS feeds, filter for relevance."""
    headlines = []
    terms = {ticker.lower(), company_name.lower().split()[0]}
    cutoff = datetime.now() - timedelta(hours=24)

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:30]:
                title = getattr(entry, "title", "") or ""
                summary = getattr(entry, "summary", "") or ""
                text = f"{title}. {summary}"
                if not any(t in text.lower() for t in terms):
                    continue
                pub = getattr(entry, "published_parsed", None)
                if pub:
                    pub_dt = datetime(*pub[:6])
                    if pub_dt < cutoff:
                        continue
                headlines.append({"text": text[:500], "source": "rss"})
        except Exception as e:
            logger.debug(f"[news] RSS feed failed ({feed_url}): {e}")

    return headlines[:10]


def _check_sec_8k(ticker: str, company_name: str) -> bool:
    """Check SEC EDGAR for recent 8-K filings for the company."""
    try:
        feed = feedparser.parse(SEC_8K_FEED)
        company_lower = company_name.lower().split()[0]
        ticker_lower  = ticker.lower()
        for entry in feed.entries[:40]:
            title = (getattr(entry, "title", "") or "").lower()
            if ticker_lower in title or company_lower in title:
                return True
    except Exception as e:
        logger.debug(f"[news] SEC 8-K check failed: {e}")
    return False


def _check_earnings_proximity(ticker: str) -> bool:
    """Check if earnings are within 3 days using yfinance calendar."""
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            return False
        # cal can be a dict or DataFrame depending on yfinance version
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if not dates:
                return False
            if hasattr(dates, '__iter__') and not isinstance(dates, str):
                for d in dates:
                    if hasattr(d, 'date'):
                        delta = (d.date() - datetime.now().date()).days
                        if 0 <= delta <= 3:
                            return True
            return False
        elif hasattr(cal, 'iloc'):
            for col in cal.columns:
                for val in cal[col]:
                    try:
                        if hasattr(val, 'date'):
                            delta = (val.date() - datetime.now().date()).days
                            if 0 <= delta <= 3:
                                return True
                    except Exception:
                        pass
    except Exception as e:
        logger.debug(f"[news] earnings check failed for {ticker}: {e}")
    return False


def get_news_sentiment(ticker: str, company_name: str, api_key: Optional[str] = None) -> dict:
    """
    Fetch news for a ticker and return sentiment analysis.

    Returns:
        avg_polarity, top_headlines, sec_8k_flag, earnings_risk
    """
    if api_key is None:
        api_key = os.getenv("NEWS_API_KEY", "")

    all_headlines = []
    all_headlines.extend(_fetch_newsapi(ticker, company_name, api_key))
    all_headlines.extend(_fetch_rss(ticker, company_name))

    # deduplicate by text prefix
    seen = set()
    unique = []
    for h in all_headlines:
        key = h["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(h)

    # Score each headline
    scored = []
    for h in unique[:10]:
        pol = _sentiment(h["text"])
        scored.append({**h, "polarity": pol})

    if scored:
        avg_polarity = sum(s["polarity"] for s in scored) / len(scored)
    else:
        avg_polarity = 0.0

    top5 = sorted(scored, key=lambda x: abs(x["polarity"]), reverse=True)[:5]

    sec_flag      = _check_sec_8k(ticker, company_name)
    earnings_risk = _check_earnings_proximity(ticker)

    return {
        "ticker": ticker,
        "avg_polarity": round(avg_polarity, 4),
        "headline_count": len(scored),
        "top_headlines": [{"text": h["text"][:200], "polarity": h["polarity"]} for h in top5],
        "sec_8k_flag": sec_flag,
        "earnings_risk": earnings_risk,
    }


def get_news_batch(tickers: list, company_names: dict, api_key: Optional[str] = None,
                   max_workers: int = 3) -> dict:
    """Fetch news for multiple tickers. Rate-limited to avoid NewsAPI quota burn."""
    from concurrent.futures import ThreadPoolExecutor
    results = {}

    def fetch_one(ticker):
        name = company_names.get(ticker, ticker)
        return ticker, get_news_sentiment(ticker, name, api_key)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(fetch_one, t): t for t in tickers}
        for fut in futures:
            try:
                ticker, data = fut.result()
                results[ticker] = data
            except Exception as e:
                logger.warning(f"[news] batch failed for {futures[fut]}: {e}")
                results[futures[fut]] = {
                    "ticker": futures[fut],
                    "avg_polarity": 0.0,
                    "headline_count": 0,
                    "top_headlines": [],
                    "sec_8k_flag": False,
                    "earnings_risk": False,
                }
    return results
