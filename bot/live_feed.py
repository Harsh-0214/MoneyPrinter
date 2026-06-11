"""Writes bot decisions to data/live_feed.json for the Vercel dashboard."""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

FEED_PATH   = Path(__file__).parent.parent / "data" / "live_feed.json"
MAX_ENTRIES = 2000


def _load() -> dict:
    if FEED_PATH.exists():
        try:
            with open(FEED_PATH) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[live_feed] load failed: {e} — starting fresh")
    return {"meta": {}, "daily_history": {}, "entries": []}


def _save(feed: dict) -> None:
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEED_PATH, "w") as f:
        json.dump(feed, f, indent=2, default=str)


def write_live_feed(decisions: list[dict], session: str) -> None:
    """
    Append all scored decisions (buys, holds, shorts) to live_feed.json.
    Pulls indicator fields from _indicators if present on the decision dict.
    Keeps last MAX_ENTRIES entries. Safe to call with an empty list.
    """
    feed = _load()
    entries: list = feed.get("entries", [])
    daily: dict   = feed.get("daily_history", {})

    now_iso     = datetime.now(timezone.utc).isoformat()
    today_str   = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # today_runs counter — reset if date rolled over
    meta        = feed.get("meta", {})
    today_runs  = meta.get("total_runs_today", 0)
    if meta.get("last_updated", "")[:10] != today_str:
        today_runs = 0
    today_runs += 1

    # Tally today's decisions for the daily_history block
    today = daily.setdefault(today_str, {
        "date": today_str,
        "total_decisions": 0,
        "buys":   0,
        "holds":  0,
        "shorts": 0,
        "claude_confirmed": 0,
        "claude_rejected":  0,
        "claude_skipped":   0,
        "sessions":         [],
    })
    if session not in today["sessions"]:
        today["sessions"].append(session)

    # Build entry rows
    for d in decisions:
        ind    = d.get("_indicators") or {}
        action = (d.get("action") or "hold").lower()

        entry = {
            "timestamp":          now_iso,
            "session":            session,
            "ticker":             d.get("ticker", ""),
            "action":             action,
            "net_score":          d.get("net_score", 0),
            "confidence":         round(float(d.get("confidence") or 0), 4),
            "strategy":           d.get("strategy", ""),
            "bull_score":         d.get("bull_score", 0),
            "bear_score":         d.get("bear_score", 0),
            "signals_triggered":  d.get("signals_triggered", []),
            "signals_against":    d.get("signals_against", []),
            "ai_confirmed":        d.get("ai_confirmed"),
            "ai_reasoning":        d.get("ai_reasoning", ""),
            "ai_entry_price":      d.get("ai_entry_price"),
            "ai_stop_loss":        d.get("ai_stop_loss"),
            "ai_take_profit":      d.get("ai_take_profit"),
            "ai_risk_reward":      d.get("ai_risk_reward"),
            "ai_entry_condition":  d.get("ai_entry_condition", ""),
            "entry_price":         d.get("entry_price"),
            "stop_loss":           d.get("stop_loss"),
            "take_profit":         d.get("take_profit"),
            "risk_reward":         d.get("risk_reward"),
            "reasoning":           d.get("reasoning", ""),
            # Live position context (if bot already holds this stock)
            "held_position":      {
                "qty":             d["_position"].get("qty"),
                "avg_entry_price": d["_position"].get("avg_entry_price"),
                "unrealized_plpc": d["_position"].get("unrealized_plpc"),
                "side":            d["_position"].get("side"),
            } if d.get("_position") else None,
            # News headlines Claude read (stored so dashboard can show them)
            "news_headlines":     (d.get("_news") or {}).get("top_headlines", []),
            "news_polarity":      (d.get("_news") or {}).get("avg_polarity"),
            "news_count":         (d.get("_news") or {}).get("headline_count", 0),
            "bull_keyword_boost": (d.get("_news") or {}).get("bull_keyword_boost", 0),
            "bear_keyword_boost": (d.get("_news") or {}).get("bear_keyword_boost", 0),
            "earnings_risk":      (d.get("_news") or {}).get("earnings_risk", {}).get("risk_level", "clear"),
            # Multi-timeframe velocity and fundamentals
            "return_1d":          d.get("return_1d"),
            "return_5d":          d.get("return_5d"),
            "return_1m":          d.get("return_1m"),
            "return_3m":          d.get("return_3m"),
            "velocity_penalty":   d.get("velocity_penalty_applied"),
            "fundamental_score":  d.get("fundamental_score"),
            "hype_penalty":       d.get("hype_penalty_applied"),
            "breakout_quality":   d.get("breakout_quality", "unknown"),
            # Multi-day setup maturity (from historical_context)
            "setup_maturity":     (d.get("_historical_context") or {}).get("maturity_label", "unknown"),
            "setup_confluence":   (d.get("_historical_context") or {}).get("days_of_confluence", 0),
            # Full indicator set
            "rsi":                ind.get("rsi"),
            "macd_hist":          ind.get("macd_hist"),
            "volume_ratio":       ind.get("volume_ratio"),
            "intraday_move_pct":  ind.get("intraday_move_pct"),
            "gap_pct":            ind.get("gap_pct"),
            "adx":                ind.get("adx"),
            "adx_di_plus":        ind.get("adx_di_plus"),
            "adx_di_minus":       ind.get("adx_di_minus"),
            "bb_pctb":            ind.get("bb_pctb"),
            "atr":                ind.get("atr"),
            "vwap":               ind.get("vwap"),
            "ema9":               ind.get("ema9"),
            "ema21":              ind.get("ema21"),
            "ema50":              ind.get("ema50"),
            "ema200":             ind.get("ema200"),
            "stoch_rsi":          ind.get("stoch_rsi"),
            "cci":                ind.get("cci"),
            "williams_r":         ind.get("williams_r"),
            "obv":                ind.get("obv"),
            "mfi":                ind.get("mfi"),
            "wk52_high":          ind.get("wk52_high"),
            "wk52_low":           ind.get("wk52_low"),
            "pct_from_52wk_high": ind.get("pct_from_52wk_high"),
            "pct_from_52wk_low":  ind.get("pct_from_52wk_low"),
            "pct_from_ema200":    ind.get("pct_from_ema200"),
            "current_price":      ind.get("current_price"),
            "spy_regime":         ind.get("spy_regime"),
            "vix":                ind.get("vix"),
            "ema_align":          (
                "full_bull"    if (ind.get("ema9") or 0) > (ind.get("ema21") or 0) > (ind.get("ema50") or 0) > (ind.get("ema200") or 0)
                else "partial_bull" if (ind.get("ema9") or 0) > (ind.get("ema21") or 0) > (ind.get("ema50") or 0)
                else "bear"    if (ind.get("ema9") or 0) < (ind.get("ema21") or 0)
                else "mixed"
            ),
        }
        entries.append(entry)

        # Tally
        today["total_decisions"] += 1
        if action == "buy":     today["buys"]   += 1
        elif action == "short": today["shorts"] += 1
        elif action == "sell":  today["shorts"] += 1  # exits counted alongside shorts
        else:                   today["holds"]  += 1
        ai = d.get("ai_confirmed")
        if ai is True:   today["claude_confirmed"] += 1
        elif ai is False: today["claude_rejected"]  += 1
        else:             today["claude_skipped"]   += 1

    # Trim entries
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]

    # Keep only last 30 days in daily_history
    all_days = sorted(daily.keys(), reverse=True)
    daily = {d: daily[d] for d in all_days[:30]}

    feed = {
        "meta": {
            "last_updated":     now_iso,
            "last_session":     session,
            "total_runs_today": today_runs,
        },
        "daily_history": daily,
        "entries":        entries,
    }
    _save(feed)
    logger.info(f"[live_feed] {len(decisions)} decision(s) written — {len(entries)} total")


def write_eod_summary(date_str: str, pnl: float, trades: int,
                      win_rate: float, portfolio_value: float) -> None:
    """Patch today's daily_history entry with EOD P&L data."""
    feed = _load()
    daily = feed.get("daily_history", {})
    today = daily.setdefault(date_str, {"date": date_str})
    today.update({
        "pnl_dollar":       round(pnl, 2),
        "trades_executed":  trades,
        "win_rate":         round(win_rate, 4),
        "portfolio_value":  round(portfolio_value, 2),
    })
    feed["daily_history"] = daily
    _save(feed)
    logger.info(f"[live_feed] EOD summary patched for {date_str}")


def write_portfolio_snapshot(alpaca_client) -> None:
    """
    Write current account + positions into feed["portfolio"] so the dashboard
    can show holdings. Positions merge live Alpaca data (ground truth) with
    DB stop/target/strategy context.
    """
    if alpaca_client is None:
        return
    try:
        from bot.trader import get_account, get_positions
        from bot.logger import get_open_trades

        account = get_account(alpaca_client)
        live    = get_positions(alpaca_client)
        db_open = {t.get("ticker"): t for t in get_open_trades()}

        positions = []
        for p in live:
            sym = p["symbol"]
            db  = db_open.get(sym, {})
            qty = float(p.get("qty") or 0)
            cur = p.get("current_price")
            positions.append({
                "ticker":           sym,
                "qty":              qty,
                "side":             str(p.get("side", "")).split(".")[-1].lower(),
                "avg_entry":        p.get("avg_entry_price"),
                "current":          cur,
                "market_value":     round(qty * cur, 2) if cur else None,
                "unrealized_pl":    p.get("unrealized_pl"),
                "unrealized_plpc":  p.get("unrealized_plpc"),
                "stop_loss":        db.get("stop_loss"),
                "take_profit":      db.get("take_profit"),
                "strategy":         db.get("strategy"),
                "time_horizon":     db.get("time_horizon"),
                "entered":          db.get("timestamp"),
            })

        feed = _load()
        feed["portfolio"] = {
            "updated":      datetime.now(timezone.utc).isoformat(),
            "equity":       account.get("equity"),
            "cash":         account.get("cash"),
            "buying_power": account.get("buying_power"),
            "positions":    positions,
        }
        _save(feed)
        logger.info(f"[live_feed] portfolio snapshot written — {len(positions)} positions")
    except Exception as e:
        logger.warning(f"[live_feed] portfolio snapshot failed: {e}")


def write_trades_snapshot(limit: int = 200) -> None:
    """Write recent trade rows (open + closed) into feed["trades"]."""
    try:
        from bot.logger import get_recent_trades
        rows = get_recent_trades(limit)
        slim = []
        for t in rows:
            slim.append({
                "id":             t.get("id"),
                "timestamp":      t.get("timestamp"),
                "exit_timestamp": t.get("exit_timestamp"),
                "session":        t.get("session"),
                "ticker":         t.get("ticker"),
                "action":         t.get("action"),
                "strategy":       t.get("strategy"),
                "time_horizon":   t.get("time_horizon"),
                "quantity":       t.get("quantity"),
                "entry_price":    t.get("entry_price"),
                "exit_price":     t.get("exit_price"),
                "stop_loss":      t.get("stop_loss"),
                "take_profit":    t.get("take_profit"),
                "status":         t.get("status"),
                "pnl_dollar":     t.get("pnl_dollar"),
                "pnl_pct":        t.get("pnl_pct"),
                "confidence":     t.get("confidence"),
                "net_score":      t.get("net_score"),
                "reasoning":      (t.get("reasoning") or "")[:300],
            })
        feed = _load()
        feed["trades"] = slim
        _save(feed)
        logger.info(f"[live_feed] trades snapshot written — {len(slim)} rows")
    except Exception as e:
        logger.warning(f"[live_feed] trades snapshot failed: {e}")
