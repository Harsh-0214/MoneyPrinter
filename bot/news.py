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


_earnings_cache: dict = {}


def _check_earnings_proximity(ticker: str) -> dict:
    """
    Check earnings proximity using yfinance calendar.
    Returns dict: {"days_to_earnings": int or None, "risk_level": "block"|"warn"|"clear"}
    - block = within 3 days (no trade)
    - warn  = within 7 days (reduce confidence by 0.20)
    - clear = no imminent earnings
    Results are cached for the process lifetime.
    """
    if ticker in _earnings_cache:
        return _earnings_cache[ticker]

    result = {"days_to_earnings": None, "risk_level": "clear"}
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            _earnings_cache[ticker] = result
            return result

        closest_days = None
        # cal can be a dict or DataFrame depending on yfinance version
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if hasattr(dates, '__iter__') and not isinstance(dates, str):
                for d in dates:
                    if hasattr(d, 'date'):
                        delta = (d.date() - datetime.now().date()).days
                        if delta >= 0:
                            if closest_days is None or delta < closest_days:
                                closest_days = delta
        elif hasattr(cal, 'iloc'):
            for col in cal.columns:
                for val in cal[col]:
                    try:
                        if hasattr(val, 'date'):
                            delta = (val.date() - datetime.now().date()).days
                            if delta >= 0:
                                if closest_days is None or delta < closest_days:
                                    closest_days = delta
                    except Exception:
                        pass

        result["days_to_earnings"] = closest_days
        if closest_days is not None:
            if closest_days <= 3:
                result["risk_level"] = "block"
            elif closest_days <= 7:
                result["risk_level"] = "warn"
    except Exception as e:
        logger.debug(f"[news] earnings check failed for {ticker}: {e}")

    _earnings_cache[ticker] = result
    return result


_MASSIVE_BULL_KW = [
    "jensen huang", "elon musk", "trillion dollar", "record revenue",
    "beats expectations", "raised guidance", "major contract", "fda approved",
    "partnership with nvidia", "ai breakthrough", "blowout quarter", "record earnings",
]
_MASSIVE_BEAR_KW = [
    "sec investigation", "class action", "ceo resigned", "revenue miss",
    "guidance cut", "data breach", "recall", "bankruptcy", "delisted",
    "doj probe", "going concern", "fraud",
]
_MODERATE_BULL_KW = [
    "upgrade", "buy rating", "price target raised", "strong demand",
    "market share gain", "beat estimates", "raised outlook",
]
_MODERATE_BEAR_KW = [
    "downgrade", "sell rating", "price target cut", "disappointing",
    "headwinds", "miss", "below expectations",
]


def keyword_amplifier(text: str) -> tuple:
    """
    Scan combined headline text for bullish/bearish keywords.
    Returns (bull_boost, bear_boost) — additive point values for scoring.
    """
    lower = text.lower()
    bull_boost = 0.0
    bear_boost = 0.0
    for kw in _MASSIVE_BULL_KW:
        if kw in lower:
            bull_boost += 25
    for kw in _MASSIVE_BEAR_KW:
        if kw in lower:
            bear_boost += 25
    for kw in _MODERATE_BULL_KW:
        if kw in lower:
            bull_boost += 12
    for kw in _MODERATE_BEAR_KW:
        if kw in lower:
            bear_boost += 12
    return bull_boost, bear_boost


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

    # Keyword amplifier on combined headline text
    combined_text = " ".join(h["text"] for h in scored)
    bull_kw_boost, bear_kw_boost = keyword_amplifier(combined_text)

    return {
        "ticker": ticker,
        "avg_polarity": round(avg_polarity, 4),
        "headline_count": len(scored),
        "top_headlines": [{"text": h["text"][:200], "polarity": h["polarity"]} for h in top5],
        "sec_8k_flag": sec_flag,
        "earnings_risk": earnings_risk,
        "bull_keyword_boost": bull_kw_boost,
        "bear_keyword_boost": bear_kw_boost,
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
