"""
Discovery scanner — screens a large-cap universe for active movers
and promotes them into discovered_tickers.json for the next trading sessions.

Criteria for promotion:
  - Average daily volume >= 2M shares
  - Today's volume ratio >= 1.5x (actively moving)
  - |price change vs prev close| >= 1.5%  OR  within 3% of 52-week high
  - Not already in the static watchlist
  - Claude second-opinion screen: rejects candidates without a clear
    business reason driving the activity
  - Max DISCOVERY_LIMIT tickers kept at once (ranked by Claude confidence)

Large-cap is enforced implicitly by the hand-curated UNIVERSE below — Alpaca
does not expose market cap, so there is no live market-cap filter.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from bot.data import fetch_snapshots_batch, fetch_daily_bars_batch

logger = logging.getLogger(__name__)

DISCOVERED_PATH = Path(__file__).parent.parent / "discovered_tickers.json"
DISCOVERY_LIMIT = 10   # max tickers added from discovery at one time

# ~150 large/mid-cap liquid names across sectors — hand-curated to avoid micro caps
UNIVERSE = [
    # Mega-cap tech
    "ORCL", "CRM", "ADBE", "INTC", "QCOM", "TXN", "NOW", "SNOW", "NET", "PANW",
    "CRWD", "ZS", "DDOG", "MDB", "SHOP", "UBER", "LYFT", "ABNB", "DASH", "RBLX",
    "PLTR", "PATH", "AI", "BBAI", "SOUN",
    # Semis
    "AVGO", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ARM", "ON", "SWKS", "MPWR",
    "SMCI", "NVDA", "AMD", "TSM", "ASML", "WOLF",
    # Large-cap consumer/retail
    "AMZN", "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "YUM",
    "BABA", "JD", "PDD", "ETSY", "CHWY", "W",
    # Financials
    "MS", "BLK", "SCHW", "C", "WFC", "AXP", "V", "MA", "PYPL", "SOFI",
    "HOOD", "AFRM", "NU", "LC",
    # Healthcare/biotech (liquid large-caps only)
    "JNJ", "PFE", "MRNA", "ABBV", "LLY", "UNH", "CVS", "BMY", "GILD", "BIIB",
    "REGN", "VRTX", "ISRG", "DXCM", "TDOC", "HIMS",
    # Industrials/defense
    "BA", "LMT", "RTX", "NOC", "GE", "CAT", "DE", "HON", "MMM", "UPS",
    "AXON", "KTOS", "HII",
    # Energy
    "SLB", "HAL", "MPC", "VLO", "PSX", "OXY", "DVN", "FANG",
    # Media/telecom
    "NFLX", "DIS", "CMCSA", "T", "VZ", "WBD", "SPOT", "TTD",
    # EV / clean energy
    "RIVN", "LCID", "NIO", "XPEV", "LI", "ENPH", "FSLR", "RUN", "PLUG",
    # Commodities / materials
    "FCX", "NEM", "GOLD", "AA", "CLF", "MP", "VALE",
    # Real estate / REITs (liquid)
    "AMT", "EQIX", "PLD", "O", "WELL",
    # Crypto-adjacent large-caps
    "COIN", "RIOT", "MARA", "MSTR", "CLSK",
    # High-beta momentum names
    "TSLA", "GME", "AMC", "SPCE", "JOBY", "ACHR",
    # ETFs with single-stock-like behavior (leveraged/thematic)
    "SOXL", "TQQQ", "ARKK", "LABU",
]

# Ticker -> company name for news lookups. NewsAPI matches the search term as a
# keyword, so querying the bare symbol (e.g. "AA", "PATH", "HON", "AI") returns
# articles about unrelated subjects and makes the Claude screen reject every
# candidate for lack of a catalyst. Searching the company name fixes that.
COMPANY_NAMES = {
    # Mega-cap tech
    "ORCL": "Oracle", "CRM": "Salesforce", "ADBE": "Adobe", "INTC": "Intel",
    "QCOM": "Qualcomm", "TXN": "Texas Instruments", "NOW": "ServiceNow",
    "SNOW": "Snowflake", "NET": "Cloudflare", "PANW": "Palo Alto Networks",
    "CRWD": "CrowdStrike", "ZS": "Zscaler", "DDOG": "Datadog", "MDB": "MongoDB",
    "SHOP": "Shopify", "UBER": "Uber", "LYFT": "Lyft", "ABNB": "Airbnb",
    "DASH": "DoorDash", "RBLX": "Roblox", "PLTR": "Palantir", "PATH": "UiPath",
    "AI": "C3.ai", "BBAI": "BigBear.ai", "SOUN": "SoundHound AI",
    # Semis
    "AVGO": "Broadcom", "MU": "Micron", "AMAT": "Applied Materials",
    "LRCX": "Lam Research", "KLAC": "KLA Corporation", "MRVL": "Marvell Technology",
    "ARM": "Arm Holdings", "ON": "ON Semiconductor", "SWKS": "Skyworks Solutions",
    "MPWR": "Monolithic Power Systems", "SMCI": "Super Micro Computer",
    "NVDA": "Nvidia", "AMD": "AMD", "TSM": "Taiwan Semiconductor",
    "ASML": "ASML", "WOLF": "Wolfspeed",
    # Consumer/retail
    "AMZN": "Amazon", "WMT": "Walmart", "COST": "Costco", "TGT": "Target",
    "HD": "Home Depot", "LOW": "Lowe's", "NKE": "Nike", "SBUX": "Starbucks",
    "MCD": "McDonald's", "YUM": "Yum Brands", "BABA": "Alibaba", "JD": "JD.com",
    "PDD": "PDD Holdings", "ETSY": "Etsy", "CHWY": "Chewy", "W": "Wayfair",
    # Financials
    "MS": "Morgan Stanley", "BLK": "BlackRock", "SCHW": "Charles Schwab",
    "C": "Citigroup", "WFC": "Wells Fargo", "AXP": "American Express",
    "V": "Visa", "MA": "Mastercard", "PYPL": "PayPal", "SOFI": "SoFi",
    "HOOD": "Robinhood", "AFRM": "Affirm", "NU": "Nu Holdings", "LC": "LendingClub",
    # Healthcare/biotech
    "JNJ": "Johnson & Johnson", "PFE": "Pfizer", "MRNA": "Moderna",
    "ABBV": "AbbVie", "LLY": "Eli Lilly", "UNH": "UnitedHealth",
    "CVS": "CVS Health", "BMY": "Bristol Myers Squibb", "GILD": "Gilead Sciences",
    "BIIB": "Biogen", "REGN": "Regeneron", "VRTX": "Vertex Pharmaceuticals",
    "ISRG": "Intuitive Surgical", "DXCM": "Dexcom", "TDOC": "Teladoc Health",
    "HIMS": "Hims & Hers Health",
    # Industrials/defense
    "BA": "Boeing", "LMT": "Lockheed Martin", "RTX": "RTX Corporation",
    "NOC": "Northrop Grumman", "GE": "GE Aerospace", "CAT": "Caterpillar",
    "DE": "Deere", "HON": "Honeywell", "MMM": "3M", "UPS": "UPS",
    "AXON": "Axon Enterprise", "KTOS": "Kratos Defense", "HII": "Huntington Ingalls",
    # Energy
    "SLB": "Schlumberger", "HAL": "Halliburton", "MPC": "Marathon Petroleum",
    "VLO": "Valero Energy", "PSX": "Phillips 66", "OXY": "Occidental Petroleum",
    "DVN": "Devon Energy", "FANG": "Diamondback Energy",
    # Media/telecom
    "NFLX": "Netflix", "DIS": "Disney", "CMCSA": "Comcast", "T": "AT&T",
    "VZ": "Verizon", "WBD": "Warner Bros Discovery", "SPOT": "Spotify",
    "TTD": "The Trade Desk",
    # EV / clean energy
    "RIVN": "Rivian", "LCID": "Lucid Motors", "NIO": "NIO", "XPEV": "XPeng",
    "LI": "Li Auto", "ENPH": "Enphase Energy", "FSLR": "First Solar",
    "RUN": "Sunrun", "PLUG": "Plug Power",
    # Commodities / materials
    "FCX": "Freeport-McMoRan", "NEM": "Newmont", "GOLD": "Barrick Gold",
    "AA": "Alcoa", "CLF": "Cleveland-Cliffs", "MP": "MP Materials", "VALE": "Vale",
    # Real estate / REITs
    "AMT": "American Tower", "EQIX": "Equinix", "PLD": "Prologis",
    "O": "Realty Income", "WELL": "Welltower",
    # Crypto-adjacent
    "COIN": "Coinbase", "RIOT": "Riot Platforms", "MARA": "MARA Holdings",
    "MSTR": "MicroStrategy", "CLSK": "CleanSpark",
    # High-beta momentum
    "TSLA": "Tesla", "GME": "GameStop", "AMC": "AMC Entertainment",
    "SPCE": "Virgin Galactic", "JOBY": "Joby Aviation", "ACHR": "Archer Aviation",
    # ETFs (leveraged/thematic) — keep symbol; news on these is index-driven
    "SOXL": "Semiconductor", "TQQQ": "Nasdaq 100", "ARKK": "ARK Innovation",
    "LABU": "Biotech",
    # Static watchlist names not already above
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "META": "Meta",
    "JPM": "JPMorgan", "GS": "Goldman Sachs", "BAC": "Bank of America",
    "XOM": "Exxon Mobil", "CVX": "Chevron",
}


def _load_discovered() -> dict:
    if DISCOVERED_PATH.exists():
        try:
            with open(DISCOVERED_PATH) as f:
                return json.load(f)
        except Exception:
            pass
    return {"tickers": [], "meta": {}}


def _save_discovered(data: dict) -> None:
    with open(DISCOVERED_PATH, "w") as f:
        json.dump(data, f, indent=2)


def _meta_entry(
    ticker: str,
    source: str,
    *,
    price: Optional[float] = None,
    pct_change: float = 0.0,
    vol_ratio: Optional[float] = None,
    near_52wk: bool = False,
    news_polarity: Optional[float] = None,
    claude_confidence: Optional[float] = None,
    claude_reasoning: Optional[str] = None,
    gap_catalyst: bool = False,
) -> dict:
    """Build a discovered-ticker metadata entry in the canonical shape shared by
    every writer (discovery scan + premarket gap promotion). Keeping the shape
    in one place means the dashboard/display can read any entry uniformly,
    regardless of which session produced it.

    `pct_change` is SIGNED here (premarket gap folds in as-is). `gap_catalyst`
    is True only for premarket gap-ups; the continuous session uses it to give
    those tickers priority on the first scan cycle.
    """
    return {
        "ticker":            ticker,
        "source":            source,
        "price":             round(float(price), 2) if price is not None else None,
        "pct_change":        round(float(pct_change), 2),
        "vol_ratio":         round(float(vol_ratio), 2) if vol_ratio is not None else None,
        "near_52wk":         bool(near_52wk),
        "news_polarity":     round(float(news_polarity), 2) if news_polarity is not None else None,
        "claude_confidence": round(float(claude_confidence), 2) if claude_confidence is not None else None,
        "claude_reasoning":  claude_reasoning,
        "gap_catalyst":      bool(gap_catalyst),
    }


def run_discovery(static_tickers: list[str]) -> list[str]:
    """
    Screen UNIVERSE for active movers not already in static_tickers.
    Returns the updated list of discovered tickers (persisted to JSON).
    Uses batched Alpaca calls — all snapshots in one request, all bars in one request.
    """
    static_set = set(t.upper() for t in static_tickers)
    to_screen = [t for t in UNIVERSE if t not in static_set]

    logger.info(f"[discovery] Screening {len(to_screen)} tickers (batch mode)...")

    # Single batch call for all snapshots
    snapshots = fetch_snapshots_batch(to_screen)

    # Single batch call for all daily bars (365 days covers avg_vol + 52wk high)
    bars_map = fetch_daily_bars_batch(to_screen, days=365)

    candidates = []
    for ticker in to_screen:
        try:
            snap = snapshots.get(ticker)
            if not snap or not snap.get("price"):
                continue
            price      = snap["price"]
            prev_close = snap["prev_close"]

            if not price or price < 10:
                continue

            daily = bars_map.get(ticker)
            if daily is None or len(daily) < 10:
                continue

            # Use daily bars for volume ratio — snapshot daily_bar.volume is today's
            # partial intraday volume (≈0 at 8:30 AM premarket), which would make
            # every stock fail the vol_ratio filter.
            yesterday_vol = float(daily["Volume"].iloc[-1])
            prior_avg_vol = (
                float(daily["Volume"].iloc[-64:-1].mean()) if len(daily) >= 64
                else float(daily["Volume"].iloc[:-1].mean() if len(daily) > 1
                           else daily["Volume"].mean())
            )
            wk52_high = float(daily["High"].iloc[-252:].max()) if len(daily) >= 252 else float(daily["High"].max())

            if not prior_avg_vol or prior_avg_vol < 2_000_000:
                continue

            vol_ratio  = (yesterday_vol / prior_avg_vol) if prior_avg_vol else 0
            pct_change = abs((price - prev_close) / prev_close * 100) if prev_close else 0
            near_52wk  = bool(wk52_high and price >= wk52_high * 0.97)

            if vol_ratio >= 1.5 and (pct_change >= 1.5 or near_52wk):
                candidates.append({
                    "ticker":     ticker,
                    "price":      round(float(price), 2),
                    "pct_change": round(float(pct_change), 2),
                    "vol_ratio":  round(float(vol_ratio), 2),
                    "near_52wk":  near_52wk,
                })
                logger.info(
                    f"[discovery] CANDIDATE {ticker}: ${price:.2f} "
                    f"chg={pct_change:+.1f}% vol={vol_ratio:.1f}x near52wk={near_52wk}"
                )
        except Exception as e:
            logger.debug(f"[discovery] {ticker} skipped: {e}")

    # Rank by volume ratio, pass top 2x to Claude so it has room to reject
    candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
    pre_claude = candidates[:DISCOVERY_LIMIT * 2]

    # Auto-pass very strong movers (vol >= 2x AND move >= 7%) without waiting
    # for Claude to find a fundamental narrative. At this magnitude the price
    # action itself is the signal — Claude was rejecting real moves like HOOD
    # +9.1% / 2.3x simply because the headline didn't frame growth positively.
    auto_pass = [c for c in pre_claude
                 if c["vol_ratio"] >= 2.0 and c["pct_change"] >= 7.0]
    to_screen  = [c for c in pre_claude if c not in auto_pass]
    for c in auto_pass:
        c["claude_reasoning"]  = "auto-pass: vol>=2x and move>=7% (strong price action)"
        c["claude_confidence"] = 0.70
        logger.info(
            f"[discovery] AUTO-PASS {c['ticker']}: chg={c['pct_change']:+.1f}% "
            f"vol={c['vol_ratio']:.1f}x — bypassing Claude screen"
        )

    logger.info(f"[discovery] Sending {len(to_screen)} candidates to Claude for qualitative screen...")
    claude_approved = claude_screen_discovery_candidates(to_screen)
    approved = auto_pass + claude_approved

    # Keep the names Claude was most confident in (not just the highest-volume).
    approved.sort(key=lambda c: c.get("claude_confidence") or 0.0, reverse=True)
    promoted = approved[:DISCOVERY_LIMIT]
    promoted_tickers = [c["ticker"] for c in promoted]

    meta = {
        c["ticker"]: _meta_entry(
            c["ticker"],
            source="discovery",
            price=c.get("price"),
            pct_change=c.get("pct_change", 0.0),
            vol_ratio=c.get("vol_ratio"),
            near_52wk=c.get("near_52wk", False),
            claude_confidence=c.get("claude_confidence"),
            claude_reasoning=c.get("claude_reasoning"),
        )
        for c in promoted
    }
    _save_discovered({"tickers": promoted_tickers, "meta": meta})

    logger.info(f"[discovery] {len(promoted)} tickers promoted after Claude screen: {promoted_tickers}")
    return promoted_tickers


_DISCOVERY_SYSTEM_PROMPT = """\
You are a senior equity research analyst screening stocks for a short-term \
trading watchlist. You will receive basic market data for a stock that a \
quantitative screener has flagged as an active mover today.

Your job is to decide whether this stock deserves to be added to the watchlist \
for closer monitoring and potential trading over the next 1-3 sessions.

Rules:
- DO NOT promote a stock just because it has high volume or a big price move.
  You must identify WHY it is moving and whether that reason is credible and durable.
- Acceptable reasons: earnings beat, product launch, FDA approval, major contract,
  analyst upgrade with new catalyst, sector rotation with a clear macro driver.
- Unacceptable reasons: pure momentum with no news, meme activity, unexplained spike,
  near 52-week high with no fundamental catalyst.
- If you cannot identify a clear reason for the activity, say 'reject'.
- Be conservative. A missed opportunity costs nothing. A bad trade costs money.\
"""


def _fetch_newsapi_headlines(ticker: str, api_key: str) -> str:
    """Fetch recent headlines for a ticker using NewsAPI (reliable in cloud environments).

    Searches by company name as an exact phrase rather than the bare ticker symbol,
    which otherwise matches unrelated articles (e.g. "AA", "PATH", "AI", "HON") and
    causes the Claude screen to reject genuine movers for lack of a catalyst.
    """
    if not api_key:
        return "(no headlines fetched)"
    try:
        import urllib.request
        import urllib.parse
        name = COMPANY_NAMES.get(ticker.upper(), ticker)
        # Exact-phrase match on the company name; restrict to title/description so
        # incidental body mentions don't pull in off-topic stories.
        query = urllib.parse.quote(f'"{name}"')
        url = (
            f"https://newsapi.org/v2/everything?q={query}&searchIn=title,description"
            f"&sortBy=publishedAt&pageSize=5&language=en&apiKey={api_key}"
        )
        with urllib.request.urlopen(url, timeout=8) as resp:
            data = json.loads(resp.read())
        articles = data.get("articles") or []
        hits = [a["title"][:150] for a in articles[:3] if a.get("title")]
        return "\n".join(f"  - {h}" for h in hits) if hits else "(no headlines found)"
    except Exception:
        return "(no headlines fetched)"


def _screen_one(c: dict, client, newsapi_key: str) -> Optional[dict]:
    """Screen a single candidate through Claude. Returns candidate (approved) or None (rejected)."""
    ticker     = c["ticker"]
    price      = c.get("price", 0)
    pct_change = c.get("pct_change", 0)
    vol_ratio  = c.get("vol_ratio", 0)
    near_52wk  = c.get("near_52wk", False)

    headlines_text = _fetch_newsapi_headlines(ticker, newsapi_key)

    prompt = (
        f"DISCOVERY CANDIDATE: {ticker}\n\n"
        f"Price:        ${price:.2f}\n"
        f"Change today: {pct_change:+.1f}%\n"
        f"Volume ratio: {vol_ratio:.1f}x 3-month average\n"
        f"Near 52-wk high: {near_52wk}\n\n"
        f"Recent headlines:\n{headlines_text}\n\n"
        f"Should this stock be added to the short-term trading watchlist for the next 1-3 sessions?\n\n"
        f'Answer ONLY with a JSON object: '
        f'{{"decision": "approve" or "reject", "confidence": 0.0-1.0, '
        f'"reasoning": "one sentence explaining the business reason or why there is none"}}'
    )

    try:
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=_DISCOVERY_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=15,
        )
        raw = msg.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()
        result    = json.loads(raw)
        decision  = str(result.get("decision", "approve")).lower()
        reasoning = str(result.get("reasoning", ""))
        conf      = float(result.get("confidence", 1.0))

        if decision == "approve":
            c["claude_reasoning"]  = reasoning
            c["claude_confidence"] = conf
            logger.info(f"[discovery] Claude APPROVED {ticker} (conf={conf:.2f}): {reasoning}")
            return c
        else:
            logger.info(f"[discovery] Claude REJECTED {ticker}: {reasoning}")
            return None
    except Exception as e:
        logger.warning(f"[discovery] Claude screen failed for {ticker}: {e} — keeping candidate")
        c["claude_reasoning"] = "AI screen error — kept by default"
        return c


def claude_screen_discovery_candidates(candidates: list[dict]) -> list[dict]:
    """
    Pass quantitative discovery candidates through Claude for a qualitative screen.
    Runs all Claude calls in parallel (up to 6 workers).
    Falls back to returning all candidates unchanged if the API is unavailable.
    """
    if not candidates:
        return candidates

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.info("[discovery] ANTHROPIC_API_KEY not set — skipping Claude screen")
        return candidates

    newsapi_key = os.environ.get("NEWS_API_KEY", "")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.warning(f"[discovery] Claude client init failed: {e} — skipping screen")
        return candidates

    approved_map: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            executor.submit(_screen_one, c, client, newsapi_key): c["ticker"]
            for c in candidates
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
                if result is not None:
                    approved_map[ticker] = result
            except Exception as e:
                logger.warning(f"[discovery] screen future failed for {ticker}: {e}")

    # Preserve original ranking order
    approved = [approved_map[c["ticker"]] for c in candidates if c["ticker"] in approved_map]
    logger.info(f"[discovery] Claude screen: {len(approved)}/{len(candidates)} candidates approved")
    return approved


def get_discovered_tickers() -> list[str]:
    """Load previously discovered tickers from JSON."""
    return _load_discovered().get("tickers", [])


def get_discovered_meta() -> dict:
    """Load metadata for discovered tickers."""
    return _load_discovered().get("meta", {})


def scan_rising_movers(static_tickers: list[str], top_n: int = 5) -> list[str]:
    """
    Lightweight intraday momentum screen — runs quickly during the continuous
    session to surface UNIVERSE tickers that are surging right now.

    Criteria (looser than full discovery, meant for same-day trades):
      - Up >= 1.5% on the day OR within 1% of 52-week high
      - Volume ratio >= 1.3x average
      - Price >= $5

    Uses batch Alpaca calls — all snapshots in one request.
    Returns list of ticker symbols (not persisted).
    """
    static_set = set(t.upper() for t in static_tickers)
    to_screen  = [t for t in UNIVERSE if t not in static_set]

    snapshots = fetch_snapshots_batch(to_screen)
    bars_map  = fetch_daily_bars_batch(to_screen, days=365)

    movers = []
    for ticker in to_screen:
        try:
            snap = snapshots.get(ticker)
            if not snap or not snap.get("price"):
                continue
            price      = snap["price"]
            prev_close = snap["prev_close"]

            if not price or price < 5:
                continue

            daily = bars_map.get(ticker)
            if daily is None or len(daily) < 10:
                continue

            # Use daily bars for vol_ratio (same reason as run_discovery: snapshot
            # daily_bar.volume is partial intraday and unreliable early in session)
            yesterday_vol = float(daily["Volume"].iloc[-1])
            prior_avg_vol = (
                float(daily["Volume"].iloc[-64:-1].mean()) if len(daily) >= 64
                else float(daily["Volume"].iloc[:-1].mean() if len(daily) > 1
                           else daily["Volume"].mean())
            )
            wk52_high = float(daily["High"].iloc[-252:].max()) if len(daily) >= 252 else float(daily["High"].max())

            pct_change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            vol_ratio  = (yesterday_vol / prior_avg_vol) if prior_avg_vol else 0
            near_52wk  = bool(wk52_high and price >= wk52_high * 0.99)

            if vol_ratio >= 1.3 and (pct_change >= 1.5 or near_52wk):
                movers.append((ticker, pct_change, vol_ratio))
        except Exception:
            continue

    movers.sort(key=lambda x: x[1], reverse=True)
    result = [t for t, _, _ in movers[:top_n]]
    if result:
        logger.info(f"[discovery] rising movers this cycle: {result}")
    return result
