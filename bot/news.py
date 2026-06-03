"""News fetching and sentiment scoring.

Primary source: yfinance .news (works from any environment, no API key).
Secondary:      Google News RSS via feedparser.
Tertiary:       NewsAPI.org (free tier blocks server IPs — used as last resort).
"""

import logging
import os
import time
from datetime import datetime, timedelta
from typing import Optional

import feedparser
import requests
from textblob import TextBlob

logger = logging.getLogger(__name__)

SEC_8K_FEED = (
    "https://www.sec.gov/cgi-bin/browse-edgar"
    "?action=getcurrent&type=8-K&dateb=&owner=include&count=40&search_text=&output=atom"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}


# ── Sentiment ──────────────────────────────────────────────────────────────────

def _sentiment(text: str) -> float:
    try:
        return TextBlob(str(text)).sentiment.polarity
    except Exception:
        return 0.0


# ── Source 1: yfinance .news ────────────────────────────────────────────────────

def _fetch_yfinance_news(ticker: str) -> list[dict]:
    """
    Use yfinance Ticker.news — already works in GitHub Actions because
    yfinance is the same lib used for indicators.  Returns up to 10 items.
    """
    try:
        import yfinance as yf
        t = yf.Ticker(ticker)
        articles = t.news or []
        cutoff = datetime.now() - timedelta(hours=48)
        headlines = []
        for art in articles[:20]:
            title = art.get("title", "").strip()
            if not title:
                continue
            pub_ts = art.get("providerPublishTime")
            if pub_ts:
                pub_dt = datetime.fromtimestamp(pub_ts)
                if pub_dt < cutoff:
                    continue
            source = art.get("publisher", "yfinance")
            headlines.append({"text": title, "source": source})
        logger.info(f"[news] yfinance returned {len(headlines)} headlines for {ticker}")
        return headlines[:10]
    except Exception as e:
        logger.warning(f"[news] yfinance news failed for {ticker}: {e}")
        return []


# ── Source 2: Google News RSS ───────────────────────────────────────────────────

def _fetch_google_rss(ticker: str, company_name: str) -> list[dict]:
    """
    Google News RSS — no API key, works from most server environments.
    Falls back silently if blocked.
    """
    headlines = []
    queries = [
        f"https://news.google.com/rss/search?q={ticker}+stock&hl=en-US&gl=US&ceid=US:en",
        f"https://news.google.com/rss/search?q={requests.utils.quote(company_name)}+stock&hl=en-US&gl=US&ceid=US:en",
    ]
    cutoff = datetime.now() - timedelta(hours=48)

    for url in queries:
        try:
            feed = feedparser.parse(url, request_headers=_HEADERS)
            for entry in feed.entries[:15]:
                title   = getattr(entry, "title",   "").strip()
                summary = getattr(entry, "summary", "").strip()
                text    = f"{title}. {summary}".strip(". ") if summary else title
                if not text:
                    continue
                pub = getattr(entry, "published_parsed", None)
                if pub:
                    try:
                        pub_dt = datetime(*pub[:6])
                        if pub_dt < cutoff:
                            continue
                    except Exception:
                        pass
                headlines.append({"text": text[:400], "source": "google_news"})
        except Exception as e:
            logger.debug(f"[news] Google RSS failed ({url}): {e}")

    if headlines:
        logger.info(f"[news] Google RSS returned {len(headlines)} headlines for {ticker}")
    return headlines[:10]


# ── Source 3: NewsAPI (last resort) ────────────────────────────────────────────

def _fetch_newsapi(ticker: str, company_name: str, api_key: str) -> list[dict]:
    """
    NewsAPI.org — free tier blocks GitHub Actions IPs.
    Kept as last resort; logs the actual error when it fails so we can see why.
    """
    if not api_key:
        return []

    headlines = []
    queries = [f"{ticker} stock", company_name]

    for q in queries:
        try:
            url    = "https://newsapi.org/v2/everything"
            params = {
                "q":        q,
                "language": "en",
                "sortBy":   "publishedAt",
                "pageSize": 10,
                "from":     (datetime.now() - timedelta(hours=48)).strftime("%Y-%m-%dT%H:%M:%S"),
                "apiKey":   api_key,
            }
            resp = requests.get(url, params=params, timeout=10)

            if resp.status_code != 200:
                try:
                    body = resp.json()
                    code = body.get("code", "unknown")
                    msg  = body.get("message", resp.text[:200])
                except Exception:
                    code, msg = "parse_error", resp.text[:200]
                logger.warning(
                    f"[news] NewsAPI HTTP {resp.status_code} for '{q}' "
                    f"— code={code} message={msg}"
                )
                continue

            data = resp.json()
            if data.get("status") != "ok":
                logger.warning(
                    f"[news] NewsAPI non-ok status for '{q}': "
                    f"code={data.get('code')} message={data.get('message')}"
                )
                continue

            count = len(data.get("articles", []))
            logger.info(f"[news] NewsAPI returned {count} articles for '{q}'")
            for art in data.get("articles", []):
                title = art.get("title") or ""
                desc  = art.get("description") or ""
                text  = f"{title}. {desc}".strip(". ")
                if text:
                    headlines.append({"text": text, "source": "newsapi"})

        except Exception as e:
            logger.warning(f"[news] NewsAPI exception for '{q}': {type(e).__name__}: {e}")

    return headlines[:10]


# ── SEC 8-K check ──────────────────────────────────────────────────────────────

def _check_sec_8k(ticker: str, company_name: str) -> bool:
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


# ── Earnings proximity ─────────────────────────────────────────────────────────

_earnings_cache: dict = {}


def _check_earnings_proximity(ticker: str) -> dict:
    """
    Check earnings proximity using yfinance calendar.
    Returns: {"days_to_earnings": int|None, "risk_level": "block"|"warn"|"clear"}
    """
    if ticker in _earnings_cache:
        return _earnings_cache[ticker]

    result = {"days_to_earnings": None, "risk_level": "clear"}
    try:
        import yfinance as yf
        t   = yf.Ticker(ticker)
        cal = t.calendar
        if cal is None:
            _earnings_cache[ticker] = result
            return result

        closest_days = None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date", [])
            if hasattr(dates, "__iter__") and not isinstance(dates, str):
                for d in dates:
                    if hasattr(d, "date"):
                        delta = (d.date() - datetime.now().date()).days
                        if delta >= 0:
                            if closest_days is None or delta < closest_days:
                                closest_days = delta
        elif hasattr(cal, "iloc"):
            for col in cal.columns:
                for val in cal[col]:
                    try:
                        if hasattr(val, "date"):
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


# ── Keyword amplifier ──────────────────────────────────────────────────────────

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
    lower = text.lower()
    bull_boost = bear_boost = 0.0
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


# ── Public API ─────────────────────────────────────────────────────────────────

def get_news_sentiment(ticker: str, company_name: str, api_key: Optional[str] = None) -> dict:
    """
    Fetch news for a ticker and return sentiment analysis.

    Source priority:
      1. yfinance .news  (no key, works from GitHub Actions)
      2. Google News RSS (no key, works from most servers)
      3. NewsAPI.org     (key required; free tier blocks server IPs — last resort)
    """
    if api_key is None:
        api_key = os.getenv("NEWS_API_KEY", "")

    # ── Collect from all sources, best-first ──────────────────────────────
    all_headlines: list[dict] = []

    yf_news = _fetch_yfinance_news(ticker)
    all_headlines.extend(yf_news)

    if len(all_headlines) < 5:
        goog = _fetch_google_rss(ticker, company_name)
        all_headlines.extend(goog)

    if len(all_headlines) < 3 and api_key:
        napi = _fetch_newsapi(ticker, company_name, api_key)
        all_headlines.extend(napi)

    if not all_headlines:
        logger.warning(
            f"[news] {ticker}: 0 headlines from all sources "
            f"(yfinance={len(yf_news)}, newsapi_key={'yes' if api_key else 'no'})"
        )

    # ── Deduplicate ────────────────────────────────────────────────────────
    seen, unique = set(), []
    for h in all_headlines:
        key = h["text"][:80]
        if key not in seen:
            seen.add(key)
            unique.append(h)

    # ── Score sentiment ────────────────────────────────────────────────────
    scored = []
    for h in unique[:15]:
        pol = _sentiment(h["text"])
        scored.append({**h, "polarity": pol})

    avg_polarity = (
        sum(s["polarity"] for s in scored) / len(scored) if scored else 0.0
    )
    top5 = sorted(scored, key=lambda x: abs(x["polarity"]), reverse=True)[:5]

    sec_flag      = _check_sec_8k(ticker, company_name)
    earnings_risk = _check_earnings_proximity(ticker)

    combined_text            = " ".join(h["text"] for h in scored)
    bull_kw_boost, bear_kw_boost = keyword_amplifier(combined_text)

    logger.info(
        f"[news] {ticker}: {len(scored)} headlines  "
        f"polarity={avg_polarity:+.2f}  "
        f"bull_boost={bull_kw_boost}  bear_boost={bear_kw_boost}"
    )

    return {
        "ticker":             ticker,
        "avg_polarity":       round(avg_polarity, 4),
        "headline_count":     len(scored),
        "top_headlines":      [{"text": h["text"][:200], "polarity": h["polarity"]} for h in top5],
        "sec_8k_flag":        sec_flag,
        "earnings_risk":      earnings_risk,
        "bull_keyword_boost": bull_kw_boost,
        "bear_keyword_boost": bear_kw_boost,
    }


def get_news_batch(tickers: list, company_names: dict, api_key: Optional[str] = None,
                   max_workers: int = 3) -> dict:
    """Fetch news for multiple tickers in parallel."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = {}

    def fetch_one(ticker):
        name = company_names.get(ticker, ticker)
        return ticker, get_news_sentiment(ticker, name, api_key)

    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futures):
            t = futures[fut]
            try:
                ticker, data = fut.result()
                results[ticker] = data
            except Exception as e:
                logger.warning(f"[news] batch failed for {t}: {e}")
                results[t] = {
                    "ticker":             t,
                    "avg_polarity":       0.0,
                    "headline_count":     0,
                    "top_headlines":      [],
                    "sec_8k_flag":        False,
                    "earnings_risk":      {"days_to_earnings": None, "risk_level": "clear"},
                    "bull_keyword_boost": 0,
                    "bear_keyword_boost": 0,
                }
    return results
