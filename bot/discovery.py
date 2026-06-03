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
"""

import json
import logging
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

    # Rank by volume ratio, keep top N
    candidates.sort(key=lambda x: x["vol_ratio"], reverse=True)
    promoted = candidates[:DISCOVERY_LIMIT]
    promoted_tickers = [c["ticker"] for c in promoted]

    # Persist
    meta = {c["ticker"]: c for c in promoted}
    _save_discovered({"tickers": promoted_tickers, "meta": meta})

    logger.info(
        f"[discovery] {len(promoted)} tickers promoted: {promoted_tickers}"
    )
    return promoted_tickers


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
