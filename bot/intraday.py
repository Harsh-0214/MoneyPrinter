"""
Intraday gap-and-go scanner and ORB (Opening Range Breakout) execution engine.

Strategy: Pre-market gap scanner builds a quality-scored watchlist each morning.
          During the session, ORB breakouts are monitored for entry. All positions
          are closed same-day — never held overnight.

Research-verified thresholds:
  - 4% minimum gap (fills <4% = 60-89% of the time)
  - RVOL ≥3x at entry (+40-60% follow-through vs <3x)
  - 10:30 AM hard entry cutoff (win rate drops below 45% in afternoon)
  - SPY down >1% → skip all intraday trades (macro headwind kills momentum)
"""

import json
import logging
import os
from datetime import datetime, timezone
from math import floor
from pathlib import Path
from typing import Optional

import pandas as pd
import pytz

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# ── Constants ──────────────────────────────────────────────────────────────────

MIN_GAP_PCT              = 0.04    # 4% min — gaps <4% fill 60-89% (research-verified)
MAX_GAP_PCT              = 0.18    # >18% → reversal risk too high
MIN_GAP_QUALITY_SCORE    = 55      # minimum score to add to watchlist
MIN_PREMARKET_VOL_ABS    = 100_000 # absolute shares pre-market (noise filter)
MIN_PREMARKET_VOL_RATIO  = 0.03    # 3% of ADV — institutional interest threshold
MIN_RVOL_ENTRY           = 3.0     # RVOL ≥3x at entry — +40% follow-through vs <3x
MIN_PRICE                = 5.0
MAX_PRICE                = 200.0
MIN_ADV                  = 500_000 # skip thinly-traded stocks
ORB_BARS                 = 3       # 3 × 5-min = 15-minute opening range
MAX_ENTRY_HOUR_ET        = 10      # no entries after this hour…
MAX_ENTRY_MINUTE_ET      = 30      # …and this minute (10:30 AM)
HARD_CLOSE_HOUR_ET       = 15
HARD_CLOSE_MINUTE_ET     = 30      # exit everything at 3:30 PM
INTRADAY_RISK_PCT        = 0.01    # risk 1% of portfolio per intraday trade
INTRADAY_POS_CAP         = 0.08    # max 8% of portfolio per position
MAX_INTRADAY_POSITIONS   = 3
INTRADAY_KILL_SWITCH_PCT = -0.025  # stop new entries if down 2.5% intraday
BREAKEVEN_TRIGGER_PCT    = 0.03    # move stop to entry at +3%
TRAILING_TRIGGER_PCT     = 0.05    # begin trailing at +5%
TRAILING_GAP_PCT         = 0.02    # trail 2% below highest close
MOMENTUM_EXIT_BARS       = 3       # exit if below VWAP for 3 consecutive bars
ORB_VOLUME_CONFIRM_MULT  = 1.5     # breakout bar must be ≥1.5× ORB avg volume

_WATCHLIST_PATH = Path(__file__).parent.parent / "data" / "intraday_watchlist.json"

# ── Watchlist persistence ──────────────────────────────────────────────────────

def _save_gap_watchlist(candidates: list[dict]) -> None:
    _WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(_WATCHLIST_PATH, "w") as f:
            json.dump(candidates, f, indent=2, default=str)
        logger.info(f"[intraday] Saved {len(candidates)} gap candidates to watchlist")
    except Exception as e:
        logger.warning(f"[intraday] Could not save watchlist: {e}")


def load_gap_watchlist() -> list[dict]:
    """Load today's gap watchlist from disk."""
    try:
        if _WATCHLIST_PATH.exists():
            with open(_WATCHLIST_PATH) as f:
                data = json.load(f)
            logger.info(f"[intraday] Loaded {len(data)} gap candidates from watchlist")
            return data
    except Exception as e:
        logger.warning(f"[intraday] Could not load watchlist: {e}")
    return []


# ── Dynamic universe ───────────────────────────────────────────────────────────

def _is_gappable(ticker: str, snap: Optional[dict], adv: float) -> bool:
    """Basic filters before quality scoring."""
    if not snap:
        return False
    price = snap.get("price") or snap.get("daily_open") or 0
    if price < MIN_PRICE or price > MAX_PRICE:
        return False
    if adv > 0 and adv < MIN_ADV:
        return False
    return True


def _fetch_todays_earnings_reporters() -> list[str]:
    """Return tickers reporting earnings today (best-effort via yfinance)."""
    reporters = []
    try:
        import yfinance as yf
        # Check a broad watchlist for today's earnings — yfinance earnings_dates
        # attribute gives the schedule; we check if today's date is in the df index.
        today_str = datetime.now().strftime("%Y-%m-%d")
        from bot.discovery import ALL_TICKERS  # type: ignore
        sample = list(ALL_TICKERS)[:100]  # check first 100 for speed
        for t in sample:
            try:
                info = yf.Ticker(t).calendar
                if info is None:
                    continue
                earnings_date = info.get("Earnings Date", [])
                if isinstance(earnings_date, list) and earnings_date:
                    for ed in earnings_date:
                        if hasattr(ed, "strftime") and ed.strftime("%Y-%m-%d") == today_str:
                            reporters.append(t)
                            break
            except Exception:
                pass
    except Exception as e:
        logger.debug(f"[intraday] earnings reporters fetch failed: {e}")
    return reporters


def build_intraday_universe(base_tickers: list[str]) -> list[str]:
    """
    Build dynamic scan universe each morning:
      1. Base universe (existing tracked tickers)
      2. Today's earnings reporters (via yfinance calendar)
      3. Alpaca most-actives from prior session (top 50)
    Caps at ~350 tickers for snapshot performance.
    """
    tickers: set[str] = set(base_tickers)

    # 1. Today's earnings reporters
    try:
        reporters = _fetch_todays_earnings_reporters()
        tickers.update(reporters)
        if reporters:
            logger.info(f"[intraday] Added {len(reporters)} earnings reporters to universe")
    except Exception:
        pass

    # 2. Alpaca most-actives (top 50 prior-session volume leaders)
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import MostActivesRequest
        client = StockHistoricalDataClient(
            os.environ.get("ALPACA_API_KEY", ""),
            os.environ.get("ALPACA_SECRET_KEY", ""),
        )
        actives = client.get_stock_most_actives(MostActivesRequest(top=50))
        syms = [a.symbol for a in (actives.most_actives or [])]
        tickers.update(syms)
        logger.info(f"[intraday] Added {len(syms)} Alpaca most-actives to universe")
    except Exception as e:
        logger.debug(f"[intraday] most-actives fetch failed (may not be available): {e}")

    # 3. Filter obviously ungappable symbols (multi-char suffixes = preferred/warrants/etc.)
    filtered = [t for t in tickers if len(t) <= 5 and t.isalpha()]
    return filtered[:350]


# ── Catalyst scoring ───────────────────────────────────────────────────────────

_CATALYST_STRONG = ["earnings", "beat", "surprise", "blowout", "record quarter", "eps beat"]
_CATALYST_UPGRADE = ["upgrade", "raised target", "price target", "overweight", "buy rating", "outperform"]
_CATALYST_MODERATE = ["buyback", "dividend", "acquisition", "partnership", "fda", "approval", "contract"]


def score_catalyst_from_news(news: Optional[dict], ticker: str) -> tuple[int, str]:
    """
    Score catalyst strength from news sentiment dict.
    Returns (pts, catalyst_label) where pts is 0-30.

    news dict expected keys: polarity, headline, summary (from bot/news.py).
    """
    if not news:
        return 0, "no_catalyst"

    headline  = (news.get("headline") or news.get("title") or "").lower()
    summary   = (news.get("summary") or news.get("description") or "").lower()
    text      = f"{headline} {summary}"
    polarity  = float(news.get("polarity") or news.get("sentiment_score") or 0.0)

    # Keyword matching — ordered by strength
    for kw in _CATALYST_STRONG:
        if kw in text:
            return 30, "earnings_beat"

    for kw in _CATALYST_UPGRADE:
        if kw in text:
            return 20, "analyst_upgrade"

    for kw in _CATALYST_MODERATE:
        if kw in text:
            return 10, kw.replace(" ", "_")

    # Fallback: rely on sentiment polarity
    if polarity > 0.3:
        return 15, "strong_positive_news"
    if polarity > 0.1:
        return 8, "positive_news"

    return 0, "no_catalyst"


# ── Gap quality scoring ────────────────────────────────────────────────────────

def _score_gap_size(gap_pct: float) -> int:
    """Score gap size. Sweet spot 5-10%; larger gaps carry more reversal risk."""
    if gap_pct < MIN_GAP_PCT:
        return -999  # hard fail
    if 0.05 <= gap_pct <= 0.10:
        return 50   # sweet spot
    if 0.04 <= gap_pct < 0.05:
        return 25
    if 0.10 < gap_pct <= 0.15:
        return 35   # higher reversal risk
    if 0.15 < gap_pct <= MAX_GAP_PCT:
        return 15
    return -999  # >18% → skip


def _score_premarket_volume(pm_volume: float, adv: float) -> int:
    """Score pre-market volume vs 20-day ADV."""
    if adv <= 0:
        return 0
    ratio = pm_volume / adv
    if ratio >= 0.20:
        return 25
    if ratio >= 0.10:
        return 18
    if ratio >= 0.05:
        return 10
    return -20  # <5% ADV → retail noise


def _score_technical_context(ind: dict, snap: dict) -> int:
    """Score technical setup context (EMA position, RSI, overbought checks)."""
    pts = 0
    price = snap.get("price") or 0
    ema50 = ind.get("ema50") or 0
    rsi   = ind.get("rsi") or 50

    if ema50 > 0 and price > ema50:
        pts += 10  # above EMA50 = uptrend context

    if 50 <= rsi <= 70:
        pts += 5   # healthy momentum, not overbought
    elif rsi > 75:
        pts -= 15  # overbought — reversal risk at open

    # FOMO trap: if stock already moved >8% before ORB, skip
    prev_close = snap.get("prev_close") or 0
    daily_open = snap.get("daily_open") or 0
    if prev_close > 0 and daily_open > 0:
        intraday_gain = (daily_open - prev_close) / prev_close
        if intraday_gain > 0.08:
            pts -= 20  # already extended; reversal likely

    return pts


def compute_gap_quality_score(
    ticker: str,
    snap: dict,
    adv: float,
    news: Optional[dict],
    ind: dict,
    spy_change_pct: float,
) -> tuple[int, dict]:
    """
    Compute overall gap quality score (0-100).
    Returns (score, breakdown_dict).
    Hard-fails (returns -999) if any disqualifying condition is met.
    """
    prev_close = snap.get("prev_close") or 0
    daily_open = snap.get("daily_open") or snap.get("price") or 0
    pm_volume  = snap.get("last_volume") or 0  # Alpaca daily_bar vol = approx pre-market

    if prev_close <= 0 or daily_open <= 0:
        return -1, {"reason": "missing_price_data"}

    gap_pct = (daily_open - prev_close) / prev_close

    # ── Hard filters (disqualify before scoring) ───────────────────────────────
    if gap_pct < MIN_GAP_PCT or gap_pct > MAX_GAP_PCT:
        return -1, {"reason": f"gap_out_of_range:{gap_pct:.2%}"}

    if pm_volume < MIN_PREMARKET_VOL_ABS:
        return -1, {"reason": f"pm_vol_too_low:{pm_volume:.0f}"}

    if adv > 0 and pm_volume / adv < MIN_PREMARKET_VOL_RATIO:
        return -1, {"reason": f"pm_vol_ratio_too_low:{pm_volume/adv:.2%}"}

    if spy_change_pct < -0.01:
        return -1, {"reason": f"spy_down:{spy_change_pct:.2%}"}

    # ── SPY context (applied to score, not hard filter, unless <-1%) ──────────
    spy_pts = 0
    if spy_change_pct > 0.003:
        spy_pts = 5
    elif spy_change_pct < -0.005:
        spy_pts = -25  # strong headwind

    # ── Component scores ───────────────────────────────────────────────────────
    gap_pts      = _score_gap_size(gap_pct)
    vol_pts      = _score_premarket_volume(pm_volume, adv)
    tech_pts     = _score_technical_context(ind, snap)
    cat_pts, cat = score_catalyst_from_news(news, ticker)

    total = gap_pts + vol_pts + tech_pts + cat_pts + spy_pts
    breakdown = {
        "gap_pct":       round(gap_pct, 4),
        "gap_pts":       gap_pts,
        "vol_pts":       vol_pts,
        "tech_pts":      tech_pts,
        "catalyst_pts":  cat_pts,
        "catalyst":      cat,
        "spy_pts":       spy_pts,
        "total":         total,
    }
    return total, breakdown


# ── Pre-market scanner ─────────────────────────────────────────────────────────

def scan_premarket_gappers(
    tickers: list[str],
    adv_map: dict[str, float],
    snapshots: dict[str, dict],
    news_map: dict[str, dict],
    daily_ind_map: dict[str, dict],
    spy_change_pct: float,
) -> list[dict]:
    """
    Build quality-scored gap watchlist from pre-market data.

    Returns sorted list of candidates (best score first) with:
      ticker, gap_pct, pm_volume, quality_score, catalyst, breakdown, entry_hint, stop_hint
    Only includes candidates with quality_score >= MIN_GAP_QUALITY_SCORE.
    """
    candidates = []

    for ticker in tickers:
        snap = snapshots.get(ticker)
        if not snap:
            continue

        adv = adv_map.get(ticker, 0)
        if not _is_gappable(ticker, snap, adv):
            continue

        ind  = daily_ind_map.get(ticker, {})
        news = news_map.get(ticker)

        score, breakdown = compute_gap_quality_score(
            ticker, snap, adv, news, ind, spy_change_pct
        )

        if score < MIN_GAP_QUALITY_SCORE:
            logger.debug(f"[intraday] {ticker}: score={score} < {MIN_GAP_QUALITY_SCORE} — skip. {breakdown.get('reason', '')}")
            continue

        prev_close = snap.get("prev_close") or 0
        daily_open = snap.get("daily_open") or snap.get("price") or 0
        gap_pct    = breakdown.get("gap_pct", 0)
        atr        = ind.get("atr") or (daily_open * 0.02)

        # Approximate entry/stop hints (ORB will override at market open)
        entry_hint = round(daily_open * 1.002, 2)          # just above open
        stop_hint  = round(daily_open * (1 - gap_pct * 0.5), 2)  # half-gap fill

        candidates.append({
            "ticker":        ticker,
            "gap_pct":       gap_pct,
            "prev_close":    prev_close,
            "daily_open":    daily_open,
            "pm_volume":     snap.get("last_volume") or 0,
            "quality_score": score,
            "catalyst":      breakdown.get("catalyst", "no_catalyst"),
            "breakdown":     breakdown,
            "entry_hint":    entry_hint,
            "stop_hint":     stop_hint,
            "atr":           atr,
            "scanned_at":    datetime.now(ET).isoformat(),
        })
        logger.info(
            f"[intraday] GAP CANDIDATE: {ticker} "
            f"gap={gap_pct:.1%} score={score} catalyst={breakdown.get('catalyst')}"
        )

    # Sort by quality score descending
    candidates.sort(key=lambda x: x["quality_score"], reverse=True)
    logger.info(f"[intraday] Pre-market scan complete: {len(candidates)} candidates pass quality gate")

    _save_gap_watchlist(candidates)
    return candidates


# ── Opening Range Breakout ─────────────────────────────────────────────────────

def compute_orb(intraday_bars_5min: pd.DataFrame) -> dict:
    """
    Compute Opening Range from 5-min bars.

    Opening Range = High and Low of the first ORB_BARS (3) × 5-min bars = 9:30-9:44.
    Returns: {orb_high, orb_low, orb_volume_avg, orb_complete}

    orb_complete is True only once at least ORB_BARS bars exist for today.
    """
    result = {
        "orb_high":       None,
        "orb_low":        None,
        "orb_volume_avg": None,
        "orb_complete":   False,
    }

    if intraday_bars_5min is None or intraday_bars_5min.empty:
        return result

    try:
        df = intraday_bars_5min.copy()
        # Keep only today's bars
        today = pd.Timestamp.now(tz=ET).date()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)

        today_bars = df[df.index.date == today]

        # Filter to opening range: 9:30 to 9:45 (first ORB_BARS 5-min bars)
        market_open = today_bars[
            (today_bars.index.hour == 9) & (today_bars.index.minute >= 30)
            | (today_bars.index.hour == 9) & (today_bars.index.minute < 45)
        ]
        # Simpler: just first ORB_BARS rows of today
        opening = today_bars.head(ORB_BARS)

        if len(opening) < ORB_BARS:
            return result  # ORB not yet complete

        result["orb_high"]       = float(opening["High"].max())
        result["orb_low"]        = float(opening["Low"].min())
        result["orb_volume_avg"] = float(opening["Volume"].mean())
        result["orb_complete"]   = True
    except Exception as e:
        logger.warning(f"[intraday] compute_orb failed: {e}")

    return result


def check_orb_breakout(
    current_price: float,
    current_bar_volume: float,
    orb: dict,
    vwap: float,
) -> dict:
    """
    Check if current price action constitutes an ORB breakout.

    Conditions (all must be true):
      1. current_price > orb_high (price broke above opening range)
      2. current_bar_volume >= ORB_VOLUME_CONFIRM_MULT × orb_volume_avg
      3. current_price > vwap (buying strength, not below average price)

    Returns: {signal: bool, reason: str}
    """
    if not orb.get("orb_complete"):
        return {"signal": False, "reason": "orb_not_complete"}

    orb_high   = orb.get("orb_high")
    orb_vol    = orb.get("orb_volume_avg")
    if orb_high is None or orb_vol is None:
        return {"signal": False, "reason": "orb_data_missing"}

    if current_price <= orb_high:
        return {"signal": False, "reason": f"price_below_orb_high:{current_price:.2f}<={orb_high:.2f}"}

    if current_bar_volume < orb_vol * ORB_VOLUME_CONFIRM_MULT:
        return {
            "signal": False,
            "reason": f"volume_insufficient:{current_bar_volume:.0f}<{orb_vol * ORB_VOLUME_CONFIRM_MULT:.0f}",
        }

    if vwap > 0 and current_price < vwap:
        return {"signal": False, "reason": f"price_below_vwap:{current_price:.2f}<{vwap:.2f}"}

    return {
        "signal": True,
        "reason": f"orb_breakout:price={current_price:.2f}>orb={orb_high:.2f},vol_ok,above_vwap",
    }


# ── Position sizing ────────────────────────────────────────────────────────────

def calculate_intraday_position(
    portfolio_value: float,
    entry_price: float,
    stop_price: float,
    risk_pct: float = INTRADAY_RISK_PCT,
    pos_cap_pct: float = INTRADAY_POS_CAP,
) -> dict:
    """
    Size intraday position to risk exactly risk_pct of portfolio.

    stop_price must be below entry_price (long only).
    Returns: {shares, cost, actual_risk_pct, stop_price, valid}
    """
    if portfolio_value <= 0 or entry_price <= 0:
        return {"shares": 0, "cost": 0, "actual_risk_pct": 0, "stop_price": stop_price, "valid": False}

    stop_distance = entry_price - stop_price
    if stop_distance <= 0:
        logger.warning(f"[intraday] stop_price ({stop_price}) must be below entry ({entry_price})")
        return {"shares": 0, "cost": 0, "actual_risk_pct": 0, "stop_price": stop_price, "valid": False}

    dollar_risk = portfolio_value * risk_pct
    shares = floor(dollar_risk / stop_distance)

    # Cap by max position value
    max_pos_value = portfolio_value * pos_cap_pct
    max_shares    = floor(max_pos_value / entry_price)
    shares        = min(shares, max_shares)
    shares        = max(0, shares)

    cost             = round(shares * entry_price, 2)
    actual_risk_pct  = (shares * stop_distance / portfolio_value) if portfolio_value > 0 else 0

    return {
        "shares":          shares,
        "cost":            cost,
        "actual_risk_pct": round(actual_risk_pct, 4),
        "stop_price":      stop_price,
        "valid":           shares > 0,
    }


# ── Exit logic ─────────────────────────────────────────────────────────────────

def should_exit_intraday(
    position: dict,
    current_price: float,
    vwap: float,
    current_volume_ratio: float,
    current_time_et: datetime,
) -> dict:
    """
    Determine if an intraday position should be exited.

    Checks (in priority order):
      1. Hard close at 3:30 PM
      2. Price dropped below stop_loss
      3. VWAP breakdown with volume surge (momentum death)
      4. Trailing stop triggered
      5. Time stop: 10:30 AM and not yet in profit

    Returns: {exit: bool, reason: str, new_stop: float|None}
    """
    result = {"exit": False, "reason": "", "new_stop": None}

    entry_price   = float(position.get("entry_price") or 0)
    stop_loss     = float(position.get("stop_loss") or 0)
    highest_seen  = float(position.get("highest_price_seen") or entry_price)
    below_vwap_ct = int(position.get("below_vwap_count") or 0)

    # Update highest price seen
    new_highest = max(highest_seen, current_price)

    # 1. Hard close at 3:30 PM — always exit, no exceptions
    if (current_time_et.hour > HARD_CLOSE_HOUR_ET or
            (current_time_et.hour == HARD_CLOSE_HOUR_ET and
             current_time_et.minute >= HARD_CLOSE_MINUTE_ET)):
        return {"exit": True, "reason": "hard_close_3:30pm", "new_stop": None}

    # 2. Stop loss hit
    if stop_loss > 0 and current_price <= stop_loss:
        return {"exit": True, "reason": f"stop_loss_hit:{current_price:.2f}<={stop_loss:.2f}", "new_stop": None}

    # 3. VWAP breakdown with volume surge (momentum dead)
    if vwap > 0 and current_price < vwap:
        if current_volume_ratio >= 2.0:
            # High-volume rejection below VWAP = distribution → exit immediately
            return {"exit": True, "reason": f"vwap_breakdown_high_vol:price={current_price:.2f}<vwap={vwap:.2f}", "new_stop": None}
        # Track consecutive bars below VWAP
        new_below_ct = below_vwap_ct + 1
        result["below_vwap_count"] = new_below_ct
        if new_below_ct >= MOMENTUM_EXIT_BARS:
            return {"exit": True, "reason": f"below_vwap_{MOMENTUM_EXIT_BARS}bars", "new_stop": None}
    else:
        result["below_vwap_count"] = 0

    # 4. Trailing stop logic
    if entry_price > 0:
        gain_pct = (new_highest - entry_price) / entry_price

        # Move to breakeven at +3%
        if gain_pct >= BREAKEVEN_TRIGGER_PCT and stop_loss < entry_price:
            result["new_stop"] = round(entry_price, 2)

        # Begin trailing at +5%
        if gain_pct >= TRAILING_TRIGGER_PCT:
            trail_price = round(new_highest * (1.0 - TRAILING_GAP_PCT), 2)
            if trail_price > stop_loss:
                result["new_stop"] = trail_price
            if current_price <= (result.get("new_stop") or stop_loss):
                return {"exit": True, "reason": f"trailing_stop_triggered:{current_price:.2f}", "new_stop": None}

    # 5. Time stop: at 10:30 AM, exit if not yet profitable
    at_cutoff = (current_time_et.hour > MAX_ENTRY_HOUR_ET or
                 (current_time_et.hour == MAX_ENTRY_HOUR_ET and
                  current_time_et.minute >= MAX_ENTRY_MINUTE_ET))
    if at_cutoff and entry_price > 0 and current_price <= entry_price:
        return {"exit": True, "reason": "time_stop:10:30am_not_profitable", "new_stop": None}

    result["highest_price_seen"] = new_highest
    return result


# ── Intraday indicator helper ──────────────────────────────────────────────────

def compute_intraday_vwap(bars_5min: pd.DataFrame) -> Optional[float]:
    """
    Compute VWAP from today's intraday 5-min bars.
    VWAP = sum(typical_price × volume) / sum(volume)
    Only uses today's bars — resets at market open.
    Returns None if insufficient data.
    """
    if bars_5min is None or bars_5min.empty:
        return None
    try:
        df = bars_5min.copy()
        today = pd.Timestamp.now(tz=ET).date()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)

        today_bars = df[df.index.date == today]
        if today_bars.empty:
            return None

        tp  = (today_bars["High"] + today_bars["Low"] + today_bars["Close"]) / 3
        vol = today_bars["Volume"]
        vwap = float((tp * vol).sum() / vol.sum()) if vol.sum() > 0 else None
        return vwap
    except Exception as e:
        logger.debug(f"[intraday] VWAP calc failed: {e}")
        return None


def compute_rvol(bars_5min: pd.DataFrame) -> Optional[float]:
    """
    Compute Relative Volume (RVOL) = today's cumulative volume / avg same-time volume.
    Uses last 10 trading days of 5-min bars as the baseline.
    Returns None if insufficient history.
    """
    if bars_5min is None or bars_5min.empty:
        return None
    try:
        df = bars_5min.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC").tz_convert(ET)
        else:
            df.index = df.index.tz_convert(ET)

        today = pd.Timestamp.now(tz=ET).date()
        today_bars    = df[df.index.date == today]
        hist_bars     = df[df.index.date < today]

        if today_bars.empty or hist_bars.empty:
            return None

        today_vol = float(today_bars["Volume"].sum())
        # Average cumulative volume at same time-of-day across past days
        today_bar_count  = len(today_bars)
        hist_by_day      = hist_bars.groupby(hist_bars.index.date)["Volume"].apply(
            lambda v: v.iloc[:today_bar_count].sum() if len(v) >= today_bar_count else v.sum()
        )
        if hist_by_day.empty:
            return None

        avg_hist_vol = float(hist_by_day.mean())
        return round(today_vol / avg_hist_vol, 2) if avg_hist_vol > 0 else None
    except Exception as e:
        logger.debug(f"[intraday] RVOL calc failed: {e}")
        return None
