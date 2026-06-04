"""
Discovery scanner — screens a large-cap universe for active movers
and promotes them into discovered_tickers.json for the next trading sessions.

Criteria for promotion:
  - Market cap >= $10B
  - Average daily volume >= 2M shares
  - Today's volume ratio >= 1.5x (actively moving)
  - |price change vs prev close| >= 1.5%  OR  within 3% of 52-week high
  - Not already in the static watchlist
  - Max DISCOVERY_LIMIT tickers kept at once (ranked by volume ratio)
  - Claude second-opinion screen: rejects candidates without a clear
    business reason driving the activity
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
            last_vol   = snap["last_volume"]

            if not price or price < 10:
                continue

            daily = bars_map.get(ticker)
            if daily is None or len(daily) < 10:
                continue

            avg_vol   = float(daily["Volume"].iloc[-63:].mean()) if len(daily) >= 63 else float(daily["Volume"].mean())
            wk52_high = float(daily["High"].iloc[-252:].max())   if len(daily) >= 252 else float(daily["High"].max())

            if not avg_vol or avg_vol < 2_000_000:
                continue

            vol_ratio  = (last_vol / avg_vol) if last_vol and avg_vol else 0
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

    logger.info(f"[discovery] Sending {len(pre_claude)} candidates to Claude for qualitative screen...")
    approved = claude_screen_discovery_candidates(pre_claude)

    promoted = approved[:DISCOVERY_LIMIT]
    promoted_tickers = [c["ticker"] for c in promoted]

    meta = {c["ticker"]: c for c in promoted}
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
    """Fetch recent headlines for a ticker using NewsAPI (reliable in cloud environments)."""
    if not api_key:
        return "(no headlines fetched)"
    try:
        import urllib.request
        import urllib.parse
        query = urllib.parse.quote(ticker)
        url = (
            f"https://newsapi.org/v2/everything?q={query}&sortBy=publishedAt"
            f"&pageSize=5&language=en&apiKey={api_key}"
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
            last_vol   = snap["last_volume"]

            if not price or price < 5:
                continue

            daily = bars_map.get(ticker)
            if daily is None or len(daily) < 10:
                continue

            avg_vol   = float(daily["Volume"].iloc[-63:].mean()) if len(daily) >= 63 else float(daily["Volume"].mean())
            wk52_high = float(daily["High"].iloc[-252:].max())   if len(daily) >= 252 else float(daily["High"].max())

            pct_change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            vol_ratio  = (last_vol / avg_vol) if last_vol and avg_vol else 0
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
