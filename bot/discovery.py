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
from pathlib import Path
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

DISCOVERED_PATH = Path(__file__).parent.parent / "discovered_tickers.json"
DISCOVERY_LIMIT = 10   # max tickers added from discovery at one time

# ~100 large-cap, liquid names across sectors — hand-curated to avoid micro/small caps
UNIVERSE = [
    # Mega-cap tech
    "ORCL", "CRM", "ADBE", "INTC", "QCOM", "TXN", "NOW", "SNOW", "NET", "PANW",
    "CRWD", "ZS", "DDOG", "MDB", "SHOP", "UBER", "LYFT", "ABNB", "DASH", "RBLX",
    # Semis
    "AVGO", "MU", "AMAT", "LRCX", "KLAC", "MRVL", "ARM", "ON", "SWKS", "MPWR",
    # Large-cap consumer/retail
    "AMZN", "WMT", "COST", "TGT", "HD", "LOW", "NKE", "SBUX", "MCD", "YUM",
    # Financials
    "MS", "BLK", "SCHW", "C", "WFC", "AXP", "V", "MA", "PYPL", "XYZ",
    # Healthcare/biotech (liquid large-caps only)
    "JNJ", "PFE", "MRNA", "ABBV", "LLY", "UNH", "CVS", "BMY", "GILD", "BIIB",
    # Industrials/defense
    "BA", "LMT", "RTX", "NOC", "GE", "CAT", "DE", "HON", "MMM", "UPS",
    # Energy
    "SLB", "HAL", "MPC", "VLO", "PSX",
    # Media/telecom
    "NFLX", "DIS", "CMCSA", "T", "VZ", "WBD",
    # EV / clean energy
    "RIVN", "LCID", "NIO", "XPEV", "LI", "ENPH", "FSLR",
    # Commodities / materials
    "FCX", "NEM", "GOLD", "AA", "CLF",
    # Real estate / REITs (liquid)
    "AMT", "EQIX", "PLD",
    # Crypto-adjacent large-caps
    "HOOD", "RIOT", "MARA",
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
    """
    static_set = set(t.upper() for t in static_tickers)
    candidates = []

    logger.info(f"[discovery] Screening {len(UNIVERSE)} tickers...")

    for ticker in UNIVERSE:
        if ticker in static_set:
            continue
        try:
            t = yf.Ticker(ticker)
            info = t.fast_info

            # Price and market cap guard
            price = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            mkt_cap = getattr(info, "market_cap", None)
            if not price or price < 10:
                continue
            if mkt_cap and mkt_cap < 10_000_000_000:  # < $10B
                continue

            # Volume
            avg_vol = getattr(info, "three_month_average_volume", None)
            last_vol = getattr(info, "last_volume", None)
            if not avg_vol or avg_vol < 2_000_000:
                continue

            vol_ratio = (last_vol / avg_vol) if last_vol and avg_vol else 0

            # Price change vs previous close
            prev_close = getattr(info, "previous_close", None)
            pct_change = abs((price - prev_close) / prev_close * 100) if prev_close else 0

            # 52-week proximity
            wk52_high = getattr(info, "year_high", None)
            near_52wk = wk52_high and price >= wk52_high * 0.97

            # Promotion criteria
            if vol_ratio >= 1.5 and (pct_change >= 1.5 or near_52wk):
                candidates.append({
                    "ticker":     ticker,
                    "price":      round(float(price), 2),
                    "pct_change": round(float(pct_change), 2),
                    "vol_ratio":  round(float(vol_ratio), 2),
                    "mkt_cap_b":  round(float(mkt_cap) / 1e9, 1) if mkt_cap else None,
                    "near_52wk":  bool(near_52wk),
                })
                logger.info(
                    f"[discovery] CANDIDATE {ticker}: ${price:.2f} "
                    f"chg={pct_change:+.1f}% vol={vol_ratio:.1f}x near52wk={near_52wk}"
                )

        except Exception as e:
            logger.debug(f"[discovery] {ticker} skipped: {e}")
            continue

    # Rank by volume ratio, keep top N before Claude screen
    candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
    pre_claude = candidates[:DISCOVERY_LIMIT * 2]  # pass 2x to Claude so it has room to reject

    # Claude qualitative screen
    logger.info(f"[discovery] Sending {len(pre_claude)} candidates to Claude for qualitative screen...")
    approved = claude_screen_discovery_candidates(pre_claude)

    # Take top DISCOVERY_LIMIT from Claude-approved set (still ranked by vol_ratio)
    promoted = approved[:DISCOVERY_LIMIT]
    promoted_tickers = [c["ticker"] for c in promoted]

    # Persist
    meta = {c["ticker"]: c for c in promoted}
    _save_discovered({"tickers": promoted_tickers, "meta": meta})

    logger.info(
        f"[discovery] {len(promoted)} tickers promoted after Claude screen: {promoted_tickers}"
    )
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


def claude_screen_discovery_candidates(candidates: list[dict]) -> list[dict]:
    """
    Pass quantitative discovery candidates through Claude for a qualitative screen.
    Returns the subset Claude approves, each with 'claude_reasoning' attached.
    Falls back to returning all candidates unchanged if the API is unavailable.
    """
    if not candidates:
        return candidates

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        logger.info("[discovery] ANTHROPIC_API_KEY not set — skipping Claude screen")
        return candidates

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
    except Exception as e:
        logger.warning(f"[discovery] Claude client init failed: {e} — skipping screen")
        return candidates

    approved = []
    for c in candidates:
        ticker     = c["ticker"]
        price      = c.get("price", 0)
        pct_change = c.get("pct_change", 0)
        vol_ratio  = c.get("vol_ratio", 0)
        mkt_cap_b  = c.get("mkt_cap_b")
        near_52wk  = c.get("near_52wk", False)

        # Fetch a few recent headlines to give Claude context
        headlines_text = "(no headlines fetched)"
        try:
            import feedparser
            feed = feedparser.parse("https://finance.yahoo.com/rss/")
            hits = []
            for entry in feed.entries[:40]:
                title = getattr(entry, "title", "") or ""
                if ticker.lower() in title.lower():
                    hits.append(title[:150])
                    if len(hits) >= 3:
                        break
            if hits:
                headlines_text = "\n".join(f"  - {h}" for h in hits)
        except Exception:
            pass

        prompt = f"""DISCOVERY CANDIDATE: {ticker}

Price:        ${price:.2f}
Change today: {pct_change:+.1f}%
Volume ratio: {vol_ratio:.1f}x 3-month average  (unusually high activity)
Market cap:   ${mkt_cap_b}B
Near 52-wk high: {near_52wk}

Recent headlines (Yahoo Finance RSS, may be incomplete):
{headlines_text}

Should this stock be added to the trading watchlist for the next 1-3 sessions?

Answer ONLY with a JSON object:
{{"decision": "approve" or "reject", "confidence": 0.0-1.0, "reasoning": "one sentence explaining the business reason for the activity, or why there is none"}}"""

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
            result   = json.loads(raw)
            decision = str(result.get("decision", "approve")).lower()
            reasoning = str(result.get("reasoning", ""))
            conf      = float(result.get("confidence", 1.0))

            if decision == "approve":
                c["claude_reasoning"] = reasoning
                c["claude_confidence"] = conf
                approved.append(c)
                logger.info(f"[discovery] Claude APPROVED {ticker} (conf={conf:.2f}): {reasoning}")
            else:
                logger.info(f"[discovery] Claude REJECTED {ticker}: {reasoning}")

        except Exception as e:
            logger.warning(f"[discovery] Claude screen failed for {ticker}: {e} — keeping candidate")
            c["claude_reasoning"] = "AI screen error — kept by default"
            approved.append(c)

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

    Returns list of ticker symbols (not persisted — just returned for the
    current scan cycle).
    """
    static_set = set(t.upper() for t in static_tickers)
    movers = []

    for ticker in UNIVERSE:
        if ticker in static_set:
            continue
        try:
            info = yf.Ticker(ticker).fast_info
            price      = getattr(info, "last_price", None) or getattr(info, "previous_close", None)
            prev_close = getattr(info, "previous_close", None)
            avg_vol    = getattr(info, "three_month_average_volume", None)
            last_vol   = getattr(info, "last_volume", None)
            wk52_high  = getattr(info, "year_high", None)

            if not price or price < 5:
                continue
            pct_change = ((price - prev_close) / prev_close * 100) if prev_close else 0
            vol_ratio  = (last_vol / avg_vol) if last_vol and avg_vol else 0
            near_52wk  = wk52_high and price >= wk52_high * 0.99

            if vol_ratio >= 1.3 and (pct_change >= 1.5 or near_52wk):
                movers.append((ticker, pct_change, vol_ratio))
        except Exception:
            continue

    movers.sort(key=lambda x: x[1], reverse=True)  # rank by % gain
    result = [t for t, _, _ in movers[:top_n]]
    if result:
        logger.info(f"[discovery] rising movers this cycle: {result}")
    return result
