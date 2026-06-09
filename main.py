"""
Main entry point for the autonomous trading bot.
Routes to session-specific logic based on --session argument.
"""

import argparse
import json
import logging
import os
import sys
import time as _time
import warnings
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# Suppress FutureWarning from ta library's PSAR (Series.__setitem__ deprecation)
# until the upstream library updates to pandas-compatible indexing.
warnings.filterwarnings("ignore", category=FutureWarning, module="ta")

import pandas_market_calendars as mcal
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler

load_dotenv()

# ── Logging setup ──────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
log_file = LOG_DIR / f"bot_{datetime.utcnow().strftime('%Y%m%d')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s - %(message)s",
    handlers=[
        RichHandler(rich_tracebacks=True, show_path=False),
        logging.FileHandler(log_file),
    ],
)
logger = logging.getLogger("main")
console = Console()

# ── Load watchlist ──────────────────────────────────────────────────────────
WATCHLIST_PATH = Path(__file__).parent / "watchlist.json"
with open(WATCHLIST_PATH) as f:
    WATCHLIST = json.load(f)

STATIC_TICKERS = (
    WATCHLIST["trade"]["tech"]
    + WATCHLIST["trade"]["momentum"]
    + WATCHLIST["trade"]["financials"]
    + WATCHLIST["trade"]["energy"]
)
MACRO_TICKERS = WATCHLIST["macro_context_only"]
COMPANY_NAMES = WATCHLIST["company_names"]
DRY_RUN       = os.getenv("DRY_RUN", "true").lower() == "true"
USE_CLAUDE    = os.getenv("USE_CLAUDE", "false").lower() == "true"

# ── Risk control constants ──────────────────────────────────────────────────
MAX_DAILY_LOSS_PCT        = 0.03   # halt trading if session P&L drops 3% below open equity
MAX_PORTFOLIO_EXPOSURE_PCT = 0.25  # max 25% of portfolio in any single new position
MAX_TOTAL_EXPOSURE_PCT     = 0.60  # max 60% of portfolio deployed at once

# ── Session-level state ─────────────────────────────────────────────────────
_session_start_equity: Optional[float] = None
_daily_loss_halt: bool = False


def get_all_trade_tickers() -> list[str]:
    """Return static watchlist merged with any tickers promoted by discovery."""
    from bot.discovery import get_discovered_tickers
    discovered = get_discovered_tickers()
    combined = list(STATIC_TICKERS)
    for t in discovered:
        if t not in combined:
            combined.append(t)
    return combined

# Max new entries per session — higher turnover acceptable for short-term style
MAX_TRADES_PER_SESSION = 5

SECTOR_GROUPS = {
    "ai_chips":   ["NVDA", "AMD", "MRVL", "SMCI", "AVGO", "ARM"],
    "big_tech":   ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
    "crypto":     ["COIN", "MSTR", "SOFI", "RIOT", "MARA"],
    "energy":     ["XOM", "CVX", "SLB", "OXY"],
    "financials": ["JPM", "GS", "BAC", "MS"],
    "healthcare": ["LLY", "UNH", "ABBV", "MRNA"],
}
MAX_POSITIONS_PER_SECTOR = 2


def _sector_of(ticker: str) -> Optional[str]:
    for sector, tickers in SECTOR_GROUPS.items():
        if ticker.upper() in tickers:
            return sector
    return None


def _check_sector_cap(ticker: str, alpaca_client=None) -> Optional[str]:
    """Returns blocking reason string if sector cap exceeded, else None."""
    sector = _sector_of(ticker)
    if sector is None:
        return None

    from bot.logger import get_open_trades
    from bot.trader import get_positions

    open_tickers = set()
    for t in get_open_trades():
        if t.get("status") in ("open", "dry_run"):
            open_tickers.add(t.get("ticker", ""))

    if alpaca_client:
        try:
            for p in get_positions(alpaca_client):
                # Deeply underwater positions (-10% or worse) don't count toward
                # the sector cap — they're not healthy positions worth protecting.
                plpc = p.get("unrealized_plpc")
                if plpc is not None and plpc < -0.10:
                    logger.debug(
                        f"[sector_cap] {p['symbol']} excluded from cap "
                        f"(unrealized P&L: {plpc:.1%})"
                    )
                    continue
                open_tickers.add(p["symbol"])
        except Exception:
            pass

    sector_positions = [t for t in open_tickers if _sector_of(t) == sector and t != ticker]
    if len(sector_positions) >= MAX_POSITIONS_PER_SECTOR:
        return (f"sector_cap:{sector} already has {len(sector_positions)} positions "
                f"({', '.join(sector_positions)})")
    return None


def _quick_news_recheck(ticker: str) -> bool:
    """Returns False if a breaking negative headline found in last 5 minutes."""
    try:
        import feedparser
        feed = feedparser.parse("https://finance.yahoo.com/rss/")
        for entry in feed.entries[:20]:
            title = (getattr(entry, "title", "") or "").lower()
            if ticker.lower() in title:
                negative_words = ["crash", "halt", "sec", "fraud", "bankrupt", "recall", "lawsuit", "downgrade"]
                if any(w in title for w in negative_words):
                    logger.warning(f"[main] {ticker} breaking negative headline before order: {title[:100]}")
                    return False
    except Exception:
        pass
    return True


def _rescore_open_positions(tickers, alpaca_client, data_client, dry_run: bool) -> None:
    """Re-score all open buy positions and exit early if thesis is broken (net < 20)."""
    import time as _time
    from bot.indicators import get_indicators
    from bot.scorer import score_ticker
    from bot.logger import get_open_trades, update_trade_exit
    from bot.trader import close_position, get_account

    try:
        open_trades = get_open_trades()
        macro = get_macro_context()
        for trade in open_trades:
            if trade.get("action") != "buy":
                continue
            ticker = trade.get("ticker")
            if not ticker:
                continue
            try:
                ind = get_indicators(ticker)
                if ind.get("error"):
                    continue
                score = score_ticker(ticker, ind, {}, macro)
                net = score.get("net_score", 0)
                if net < 20:
                    logger.warning(f"[main] {ticker} thesis broken (net={net}) — exiting early")
                    close_position(alpaca_client, ticker, dry_run)
                    trade_id = trade.get("id")
                    if trade_id:
                        ep = float(trade.get("entry_price") or 0)
                        cp = float(ind.get("current_price") or ep)
                        pnl_pct = (cp - ep) / ep * 100 if ep > 0 else 0.0
                        pnl_dollar = (cp - ep) * float(trade.get("quantity") or 0)
                        update_trade_exit(trade_id, cp, "thesis_broken", pnl_dollar, pnl_pct)
            except Exception as e:
                logger.warning(f"[main] rescore error for {ticker}: {e}")
    except Exception as e:
        logger.warning(f"[main] _rescore_open_positions failed: {e}")


def _check_earnings_proximity(open_trades) -> None:
    """Tighten stop for open positions with earnings within 2 calendar days."""
    import yfinance as yf
    from bot.logger import update_trade_trailing

    try:
        for trade in open_trades:
            ticker = trade.get("ticker")
            if not ticker:
                continue
            try:
                cal = yf.Ticker(ticker).calendar
                if cal is None:
                    continue
                # calendar may be a DataFrame or dict
                earnings_date = None
                if hasattr(cal, "get"):
                    earnings_date = cal.get("Earnings Date")
                elif hasattr(cal, "columns") and "Earnings Date" in cal.columns:
                    earnings_date = cal["Earnings Date"].iloc[0] if len(cal) > 0 else None
                elif hasattr(cal, "T"):
                    t = cal.T
                    if "Earnings Date" in t.columns:
                        earnings_date = t["Earnings Date"].iloc[0] if len(t) > 0 else None

                if earnings_date is None:
                    continue

                from datetime import date
                if hasattr(earnings_date, "date"):
                    ed = earnings_date.date()
                else:
                    ed = earnings_date

                days_until = (ed - date.today()).days
                if 0 <= days_until <= 2:
                    ep = float(trade.get("entry_price") or 0)
                    if ep > 0:
                        tight_stop = round(ep * 0.99, 2)
                        logger.warning(f"[main] {ticker} earnings in {days_until} days — tightening stop to {tight_stop}")
                        trade_id = trade.get("id")
                        if trade_id:
                            update_trade_trailing(trade_id, ep, tight_stop)
            except Exception as e:
                logger.debug(f"[main] earnings proximity check failed for {ticker}: {e}")
    except Exception as e:
        logger.warning(f"[main] _check_earnings_proximity failed: {e}")


def is_market_open_today() -> bool:
    nyse = mcal.get_calendar("NYSE")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty


def _has_open_position(ticker: str, alpaca_client=None) -> bool:
    """Return True if ticker has an open position (DB or live Alpaca) or was traded today."""
    from bot.logger import get_open_trades, get_trades_today

    # Source 1: SQLite open trades
    for t in get_open_trades():
        if t.get("ticker") == ticker and t.get("status") in ("open", "dry_run"):
            return True

    # Source 2: traded today (any action)
    for t in get_trades_today():
        if t.get("ticker") == ticker:
            return True

    # Source 3: Alpaca live positions (ground truth)
    if alpaca_client:
        try:
            from bot.trader import get_positions
            live = {p["symbol"] for p in get_positions(alpaca_client)}
            if ticker in live:
                return True
        except Exception:
            pass

    return False


def get_macro_context() -> dict:
    """Compute macro context: VIX, SPY regime, bearish_market flag, position size multiplier."""
    from bot.data import fetch_vix
    from bot.indicators import get_indicators
    from bot.risk import get_vix_multiplier

    macro = {
        "vix": 20.0,
        "spy_regime": "bull",
        "bearish_market": False,
        "vix_multiplier": 1.0,
        "spy_ema50": None,
        "spy_ema200": None,
        "spy_price": None,
    }

    # VIX
    try:
        vix_data = fetch_vix(days=2)
        if vix_data is not None and not vix_data.empty:
            macro["vix"] = float(vix_data["Close"].iloc[-1])
    except Exception as e:
        logger.warning(f"VIX fetch failed: {e}")

    # SPY regime
    try:
        spy_ind = get_indicators("SPY")
        ema50  = spy_ind.get("ema50")
        ema200 = spy_ind.get("ema200")
        price  = spy_ind.get("current_price")
        macro["spy_ema50"]  = ema50
        macro["spy_ema200"] = ema200
        macro["spy_price"]  = price

        if price and ema50 and ema200:
            if price > ema50 and price > ema200:
                macro["spy_regime"]      = "bull"
                macro["bearish_market"]  = False
            elif price < ema50 and price > ema200:
                macro["spy_regime"]      = "caution"
                macro["bearish_market"]  = True   # below EMA50 = bearish_market
            else:
                macro["spy_regime"]      = "bear"
                macro["bearish_market"]  = True
    except Exception as e:
        logger.warning(f"SPY regime check failed: {e}")

    macro["vix_multiplier"] = get_vix_multiplier(macro["vix"])
    logger.info(
        f"[macro] VIX={macro['vix']:.1f} regime={macro['spy_regime']} "
        f"bearish_market={macro['bearish_market']} "
        f"size_mult={macro['vix_multiplier']:.2f}"
    )
    return macro


def run_full_scan(session: str, macro_context: dict,
                  alpaca_client=None, data_client=None,
                  extra_tickers: list | None = None) -> list[dict]:
    """
    Score all tickers and return actionable signal list.

    Applies SPY trend filter: if bearish_market=True, all buy signals are
    dropped and only shorts with confidence > 0.80 pass through.
    """
    from bot.indicators import get_indicators_batch
    from bot.news        import get_news_batch
    from bot.scorer      import score_ticker
    from bot.strategies  import classify_strategy

    NEWS_API_KEY    = os.getenv("NEWS_API_KEY", "")
    bearish_market  = macro_context.get("bearish_market", False)

    if bearish_market:
        console.print("[bold yellow]Bearish market (SPY below EMA50) — BUY signals suppressed[/bold yellow]")

    # ── Fetch live Alpaca positions once ──────────────────────────────────────
    live_positions: dict = {}   # {ticker: position_dict}
    if alpaca_client:
        try:
            from bot.trader import get_positions as _get_live_pos
            for p in _get_live_pos(alpaca_client):
                live_positions[p["symbol"]] = p
            if live_positions:
                console.print(
                    f"[bold yellow]Open positions: {', '.join(live_positions.keys())}[/bold yellow]"
                )
        except Exception as _e:
            logger.warning(f"[scan] Could not fetch live positions: {_e}")

    all_tickers = get_all_trade_tickers()
    if extra_tickers:
        for t in extra_tickers:
            if t not in all_tickers:
                all_tickers.append(t)

    # Always include tickers we're currently holding so we evaluate exit/add
    for held_ticker in live_positions:
        if held_ticker not in all_tickers:
            all_tickers.append(held_ticker)

    console.print(f"[bold cyan]Scanning {len(all_tickers)} tickers...[/bold cyan]")
    indicators_map = get_indicators_batch(all_tickers, max_workers=2)
    news_map       = get_news_batch(all_tickers, COMPANY_NAMES, api_key=NEWS_API_KEY, max_workers=3)

    from bot.historical_context import get_historical_context_batch
    hist_ctx_map = get_historical_context_batch(all_tickers, data_client)

    signals     = []
    signals_all = []   # all scored tickers, fed to AI batch
    bull_count  = 0
    bear_count  = 0

    for ticker in all_tickers:
        ind  = indicators_map.get(ticker, {})
        news = news_map.get(ticker, {})

        if ind.get("error"):
            logger.warning(f"Skipping {ticker}: {ind['error']}")
            continue

        hist_ctx = hist_ctx_map.get(ticker, {})
        try:
            score = score_ticker(ticker, ind, news, macro_context,
                                 historical_context=hist_ctx)
            score = classify_strategy(score, ind)
            score["_historical_context"] = hist_ctx
        except Exception as e:
            logger.warning(f"Scoring failed for {ticker}: {e}")
            continue

        action     = score.get("action", "hold")
        net        = score.get("net_score", 0)
        confidence = score.get("confidence", 0.0)

        # ── SPY trend filter ───────────────────────────────────────────────
        if bearish_market and action == "buy":
            logger.info(f"[{ticker}] buy suppressed — bearish market")
            action = "hold"
            score["action"] = "hold"

        if bearish_market and action in ("short", "sell") and confidence < 0.80:
            logger.info(f"[{ticker}] short suppressed in bearish market — confidence {confidence:.2f} < 0.80")
            action = "hold"
            score["action"] = "hold"

        logger.info(
            f"[{ticker}] action={action} net={net} bull={score.get('bull_score')} "
            f"bear={score.get('bear_score')} conf={confidence:.2f} "
            f"strategy={score.get('strategy')} horizon={score.get('time_horizon','?')} "
            f"src={ind.get('price_source','?')}"
        )

        # Stamp when this signal was scored so execute_signals can detect staleness
        score["scored_at"]   = _time.time()
        score["_indicators"] = ind
        score["_news"]       = news
        if ticker in live_positions:
            score["_position"] = live_positions[ticker]
        signals_all.append(score)

    # ── Claude second-opinion pass (optional) ─────────────────────────────
    if USE_CLAUDE:
        from bot.ai_filter import run_ai_filter_batch
        pairs = [(s, s.get("_indicators", {})) for s in signals_all]
        signals_all = run_ai_filter_batch(pairs)

    # Re-tally after AI may have upgraded/downgraded actions (or scorer-only)
    for score in signals_all:
        action     = score.get("action", "hold")
        confidence = score.get("confidence", 0.0)

        # Apply strategy/confidence gates
        if action == "buy" and (confidence < 0.65 or score.get("strategy") == "mixed"):
            action = "hold"
            score["action"] = "hold"
        elif action in ("short", "sell") and confidence < 0.70:
            action = "hold"
            score["action"] = "hold"

        if action != "hold":
            signals.append(score)
            if action == "buy":
                bull_count += 1
            else:
                bear_count += 1

    from bot.logger    import log_scan
    from bot.live_feed import write_live_feed
    log_scan(
        session=session,
        tickers_scanned=len(get_all_trade_tickers()),
        signals_generated=len(signals),
        trades_executed=0,
        total_bull=bull_count,
        total_bear=bear_count,
    )
    # Write ALL scored decisions (including holds) to the live feed for the dashboard
    write_live_feed(signals_all, session)
    return signals


CORRELATION_GROUPS = {
    "AI_CHIPS":        {"NVDA", "AMD", "MRVL", "SMCI", "AVGO"},
    "BIG_TECH":        {"AAPL", "MSFT", "GOOGL", "META", "AMZN"},
    "CRYPTO_ADJACENT": {"COIN", "MSTR", "SOFI"},
    "ENERGY":          {"XOM", "CVX"},
}


def _correlation_group(ticker: str):
    for group_name, members in CORRELATION_GROUPS.items():
        if ticker in members:
            return group_name, members
    return None


def execute_signals(signals: list, alpaca_client, data_client,
                    macro_context: dict, session: str,
                    max_trades: int = MAX_TRADES_PER_SESSION) -> int:
    """
    Submit orders for the top-N signals by confidence.
    Skips duplicates (ticker already has an open position in the DB).
    """
    import time as _time
    global _session_start_equity, _daily_loss_halt

    from bot.logger import log_trade, log_rejection
    from bot.risk   import calculate_position, is_kill_switch_active, init_daily_state
    from bot.trader import (
        get_account, submit_order, compute_limit_price,
        get_latest_quote, check_order_filled,
    )

    if is_kill_switch_active():
        logger.warning("[execute] Kill switch active - no orders will be placed")
        # Log kill_switch rejections for all signals
        try:
            for sig in signals:
                if sig.get("action") in ("buy", "short"):
                    log_rejection(
                        session=session,
                        ticker=sig["ticker"],
                        net_score=sig.get("net_score", 0),
                        confidence=sig.get("confidence", 0.0),
                        action=sig.get("action", "hold"),
                        rejection_reason="kill_switch",
                        bull_score=sig.get("bull_score", 0),
                        bear_score=sig.get("bear_score", 0),
                        strategy=sig.get("strategy", ""),
                    )
        except Exception:
            pass
        return 0

    account         = get_account(alpaca_client)
    portfolio_value = account.get("portfolio_value", 100_000)
    current_equity  = account.get("equity", portfolio_value)
    init_daily_state(portfolio_value)

    # ── Daily loss circuit breaker ─────────────────────────────────────────
    if _session_start_equity is None:
        _session_start_equity = current_equity
        logger.info(f"[main] Session start equity recorded: ${_session_start_equity:,.2f}")

    if (_session_start_equity and _session_start_equity > 0
            and (current_equity - _session_start_equity) / _session_start_equity < -MAX_DAILY_LOSS_PCT):
        _daily_loss_halt = True
        logger.warning("[main] DAILY LOSS LIMIT HIT — halting new entries for this session")

    # Sort by confidence descending.
    # Buffer beyond max_trades so duplicate/held positions that get skipped in the
    # loop don't consume all slots — without this, 3 open positions in the top 5
    # leaves only 2 slots for genuinely new entries.
    ranked  = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)
    ranked  = ranked[:max_trades + 6]

    executed = 0
    for sig in ranked:
        if is_kill_switch_active():
            log_rejection(
                session=session,
                ticker=sig["ticker"],
                net_score=sig.get("net_score", 0),
                confidence=sig.get("confidence", 0.0),
                action=sig.get("action", "hold"),
                rejection_reason="kill_switch",
                bull_score=sig.get("bull_score", 0),
                bear_score=sig.get("bear_score", 0),
                strategy=sig.get("strategy", ""),
            )
            break

        if executed >= max_trades:
            log_rejection(
                session=session,
                ticker=sig["ticker"],
                net_score=sig.get("net_score", 0),
                confidence=sig.get("confidence", 0.0),
                action=sig.get("action", "hold"),
                rejection_reason="max_trades",
                bull_score=sig.get("bull_score", 0),
                bear_score=sig.get("bear_score", 0),
                strategy=sig.get("strategy", ""),
            )
            continue

        ticker      = sig["ticker"]
        action      = sig["action"]
        confidence  = sig["confidence"]
        atr         = sig.get("atr") or (sig.get("entry_price", 100) * 0.02)
        entry_price = sig.get("entry_price") or 0
        strategy    = sig.get("strategy", "mixed")
        high_vol    = sig.get("high_vol_flag", False)

        if entry_price == 0:
            continue

        # ── Daily loss halt: skip new buys/shorts ─────────────────────────
        if _daily_loss_halt and action in ("buy", "short"):
            log_rejection(
                session=session,
                ticker=ticker,
                net_score=sig.get("net_score", 0),
                confidence=confidence,
                action=action,
                rejection_reason="daily_loss_halt",
                bull_score=sig.get("bull_score", 0),
                bear_score=sig.get("bear_score", 0),
                strategy=strategy,
            )
            continue

        # Short selling is disabled pending dedicated short indicator set
        if action in ("short", "sell") and not _has_open_position(ticker, alpaca_client):
            continue  # not an open position exit — skip

        # ── Sell action: exit existing position via Alpaca ────────────────
        if action == "sell":
            try:
                from bot.trader import get_positions as _gp
                held = {p["symbol"]: p for p in _gp(alpaca_client)}
                if ticker not in held:
                    logger.info(f"[execute] SELL {ticker}: no open position found, skipping")
                    continue
                pos_qty = int(float(held[ticker]["qty"]))
                if pos_qty <= 0:
                    continue
                quote        = get_latest_quote(data_client, ticker)
                limit_price  = compute_limit_price("sell", quote, entry_price)
                order_id = submit_order(
                    client=alpaca_client,
                    ticker=ticker,
                    side="sell",
                    qty=pos_qty,
                    limit_price=limit_price,
                    dry_run=DRY_RUN,
                )
                console.print(
                    f"[red]✓ SELL (EXIT) {pos_qty}x {ticker} @ ${entry_price:.2f} "
                    f"(limit ${limit_price:.2f}) | AI recommended exit[/red]"
                )
                executed += 1
            except Exception as e:
                logger.error(f"[execute] SELL order failed for {ticker}: {e}")
            continue

        # ── Duplicate position guard (for new buys/shorts) ────────────────
        # Exception: high-confidence buy on existing position → attempt scale-in
        if action not in ("sell",) and _has_open_position(ticker, alpaca_client):
            if action == "buy" and confidence > 0.75:
                # Attempt scale-in instead of a full new entry
                from bot.risk import calculate_scale_in as _scale_in
                from bot.portfolio import get_open_positions as _get_op
                from bot.logger import get_open_trades as _got_scale
                try:
                    open_pos_map = {p["ticker"]: p for p in _get_op(alpaca_client)}
                    existing = open_pos_map.get(ticker)
                    if existing:
                        scale_shares = _scale_in(
                            existing_position=existing,
                            current_price=entry_price,
                            confidence=confidence,
                            atr=atr,
                            portfolio_value=portfolio_value,
                        )
                        if scale_shares > 0:
                            from bot.trader import get_latest_quote as _gq, compute_limit_price as _clp
                            quote       = _gq(data_client, ticker)
                            limit_price = _clp("buy", quote, entry_price)
                            order_id    = submit_order(
                                client=alpaca_client,
                                ticker=ticker,
                                side="buy",
                                qty=scale_shares,
                                limit_price=limit_price,
                                dry_run=DRY_RUN,
                            )
                            console.print(
                                f"[blue]↑ SCALE-IN {scale_shares}x {ticker} @ "
                                f"${entry_price:.2f} (conf={confidence:.2f})[/blue]"
                            )
                            executed += 1
                        else:
                            logger.info(f"[execute] Scale-in for {ticker} returned 0 shares — skipped")
                except Exception as _si_err:
                    logger.warning(f"[execute] Scale-in check failed for {ticker}: {_si_err}")
            else:
                logger.info(f"[SKIP] Already have open position in {ticker} — skipping new entry")
                log_rejection(
                    session=session,
                    ticker=ticker,
                    net_score=sig.get("net_score", 0),
                    confidence=confidence,
                    action=action,
                    rejection_reason="duplicate",
                    bull_score=sig.get("bull_score", 0),
                    bear_score=sig.get("bear_score", 0),
                    strategy=strategy,
                )
            continue

        # ── Sector cap guard ───────────────────────────────────────────────
        if action not in ("sell",):
            sector_reason = _check_sector_cap(ticker, alpaca_client)
            if sector_reason:
                logger.info(f"[SECTOR CAP] {ticker} — {sector_reason}")
                log_rejection(
                    session=session,
                    ticker=ticker,
                    net_score=sig.get("net_score", 0),
                    confidence=confidence,
                    action=action,
                    rejection_reason=sector_reason,
                    bull_score=sig.get("bull_score", 0),
                    bear_score=sig.get("bear_score", 0),
                    strategy=strategy,
                )
                continue

        # ── Correlation guard ──────────────────────────────────────────────
        group_info = _correlation_group(ticker)
        if group_info:
            group_name, members = group_info
            from bot.portfolio import get_open_positions as _get_open_pos
            open_tickers = {p["ticker"] for p in _get_open_pos(alpaca_client)}
            overlap = members & open_tickers
            if len(overlap) >= 2:
                logger.info(
                    f"[CORRELATION] Already have {len(overlap)} {group_name} positions "
                    f"({overlap}) — skipping {ticker}"
                )
                continue

        # ── Portfolio exposure cap ─────────────────────────────────────────
        if action == "buy":
            try:
                from bot.logger import get_open_trades as _got
                current_exposure = sum(
                    float(t.get("quantity") or 0) * float(t.get("entry_price") or 0)
                    for t in _got()
                    if t.get("status") in ("open", "dry_run")
                )
                if portfolio_value > 0 and current_exposure / portfolio_value >= MAX_TOTAL_EXPOSURE_PCT:
                    log_rejection(
                        session=session,
                        ticker=ticker,
                        net_score=sig.get("net_score", 0),
                        confidence=confidence,
                        action=action,
                        rejection_reason="total_exposure_cap",
                        bull_score=sig.get("bull_score", 0),
                        bear_score=sig.get("bear_score", 0),
                        strategy=strategy,
                    )
                    continue
            except Exception as _exp_err:
                logger.warning(f"[execute] exposure cap check failed: {_exp_err}")

        pos = calculate_position(
            portfolio_value=portfolio_value,
            confidence=confidence,
            atr=atr,
            price=entry_price,
            vix_multiplier=macro_context.get("vix_multiplier", 1.0),
            high_vol_flag=high_vol,
        )

        shares = pos["shares"]

        # Cap position dollar value at MAX_PORTFOLIO_EXPOSURE_PCT of portfolio
        if action == "buy" and portfolio_value > 0 and entry_price > 0:
            max_shares = int((portfolio_value * MAX_PORTFOLIO_EXPOSURE_PCT) / entry_price)
            if max_shares < shares:
                shares = max_shares
            if shares < 1:
                log_rejection(
                    session=session,
                    ticker=ticker,
                    net_score=sig.get("net_score", 0),
                    confidence=confidence,
                    action=action,
                    rejection_reason="position_too_small",
                    bull_score=sig.get("bull_score", 0),
                    bear_score=sig.get("bear_score", 0),
                    strategy=strategy,
                )
                continue

        if shares <= 0:
            logger.info(f"[execute] {ticker}: 0 shares - skipping ({pos['reason']})")
            continue

        # Quote for limit price calculation; entry_price stays as the real-time price
        quote       = get_latest_quote(data_client, ticker)
        alpaca_side = "buy" if action == "buy" else "sell"
        limit_price = compute_limit_price(alpaca_side, quote, entry_price)

        # ── Stale signal latency guard ─────────────────────────────────────
        signal_age = _time.time() - sig.get("scored_at", _time.time())
        if signal_age > 90:
            logger.warning(f"[main] {ticker} signal is {signal_age:.0f}s stale — skipping")
            continue

        # ── News recheck before order ──────────────────────────────────────
        if action == "buy" and not _quick_news_recheck(ticker):
            log_rejection(
                session=session,
                ticker=ticker,
                net_score=sig.get("net_score", 0),
                confidence=confidence,
                action=action,
                rejection_reason="breaking_negative_news",
                bull_score=sig.get("bull_score", 0),
                bear_score=sig.get("bear_score", 0),
                strategy=strategy,
            )
            continue

        try:
            order_id = submit_order(
                client=alpaca_client,
                ticker=ticker,
                side=alpaca_side,
                qty=shares,
                limit_price=limit_price,
                dry_run=DRY_RUN,
            )

            fill_status = "dry_run" if DRY_RUN else "open"
            if not DRY_RUN:
                fill = check_order_filled(alpaca_client, order_id, timeout=60)
                fill_status = fill.get("status", "open")
                # Update entry_price to actual fill if available; keep original otherwise
                if fill.get("filled_avg_price"):
                    entry_price = fill["filled_avg_price"]

            log_trade(
                session=session,
                ticker=ticker,
                action=action,
                strategy=strategy,
                time_horizon=sig.get("time_horizon", "swing"),
                quantity=shares,
                entry_price=entry_price,     # real-time price (or actual fill)
                limit_price=limit_price,     # what was submitted to the exchange
                stop_loss=sig.get("stop_loss"),
                take_profit=sig.get("take_profit"),
                confidence=confidence,
                net_score=sig.get("net_score", 0),
                bull_score=sig.get("bull_score", 0),
                bear_score=sig.get("bear_score", 0),
                signals_triggered=sig.get("signals_triggered", []),
                signals_against=sig.get("signals_against", []),
                reasoning=sig.get("reasoning", ""),
                risk_reward=sig.get("risk_reward", 2.5),
                macro_bias=macro_context.get("spy_regime", "unknown"),
                vix_level=macro_context.get("vix", 0),
                alpaca_order_id=order_id,
                status=fill_status,
                ai_confirmed=sig.get("ai_confirmed"),
                ai_reasoning=sig.get("ai_reasoning"),
            )
            executed += 1
            horizon    = sig.get("time_horizon", "swing")
            trade_type = {"position": "POSITION TRADE", "swing": "SWING TRADE", "scalp": "SCALP"}.get(horizon, "SWING TRADE")
            console.print(
                f"[green]✓ {action.upper()} {shares}x {ticker} @ ${entry_price:.2f} "
                f"(limit ${limit_price:.2f}) | {trade_type} strat={strategy} conf={confidence:.2f}[/green]"
            )
        except Exception as e:
            logger.error(f"[execute] Order failed for {ticker}: {e}")

    return executed


# ══════════════════════════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════════════════════════
# SESSION HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def session_discovery() -> None:
    """
    8:30 AM EDT - screen large-cap universe for active movers.
    Promotes up to 10 tickers into discovered_tickers.json.
    These are automatically picked up by all subsequent sessions today.
    """
    from bot.discovery import run_discovery, get_discovered_meta
    from rich.table import Table

    console.rule("[bold magenta]DISCOVERY SESSION[/bold magenta]")
    promoted = run_discovery(STATIC_TICKERS)

    if not promoted:
        console.print("[yellow]No new movers found today — trading static watchlist only.[/yellow]")
        return

    meta = get_discovered_meta()
    table = Table(title=f"Discovered Movers ({len(promoted)})", show_header=True)
    table.add_column("Ticker", style="bold magenta")
    table.add_column("Price",      justify="right")
    table.add_column("Chg %",      justify="right")
    table.add_column("Vol Ratio",  justify="right")
    table.add_column("Mkt Cap $B", justify="right")
    table.add_column("Near 52wk?")

    for t in promoted:
        m = meta.get(t, {})
        chg = m.get("pct_change", 0)
        color = "green" if chg >= 0 else "red"
        table.add_row(
            t,
            f"${m.get('price', 0):.2f}",
            f"[{color}]{chg:+.1f}%[/{color}]",
            f"{m.get('vol_ratio', 0):.1f}x",
            f"${m.get('mkt_cap_b', 0):.0f}B" if m.get("mkt_cap_b") else "—",
            "yes" if m.get("near_52wk") else "no",
        )
    console.print(table)


def session_premarket() -> None:
    """
    9:00 AM EDT — full scored dry-run scan + gap-and-go detection.

    Actions:
    1. Run complete score + Claude pass on all tickers (no orders placed).
       Decisions captured from run_full_scan — no second indicator/news fetch.
    2. Write all decisions to live_feed.json for dashboard visibility.
    3. Detect pre-market gaps using Alpaca snapshots (prev_close vs latest_trade).
       Gap-up stocks (>2%) with positive news promoted to discovered_tickers.json
       so the 9:30 continuous session treats them as priority targets.
    """
    console.rule("[bold yellow]PRE-MARKET SESSION[/bold yellow]")
    macro = get_macro_context()

    console.print(
        f"[bold]VIX={macro['vix']:.1f}  Regime={macro['spy_regime']}  "
        f"BearishMarket={macro['bearish_market']}[/bold]"
    )

    # Full scored scan — captures indicators + news internally, no orders fired
    all_decisions = run_full_scan("premarket", macro, alpaca_client=None, data_client=None)

    # Gap detection: use Alpaca snapshots (latest_trade vs prev_close).
    # gap_pct from indicators requires the open price which doesn't exist pre-market,
    # so we compute it directly from the snapshot here.
    from bot.data     import fetch_snapshots_batch
    from bot.news     import get_news_batch
    from bot.discovery import _load_discovered, _save_discovered

    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    all_tickers  = get_all_trade_tickers()

    snapshots = fetch_snapshots_batch(all_tickers)
    news_map  = get_news_batch(all_tickers, COMPANY_NAMES, api_key=NEWS_API_KEY, max_workers=3)

    gap_ups_with_news = []   # (ticker, gap_pct, polarity)
    gap_ups_plain     = []   # (ticker, gap_pct)

    for ticker in all_tickers:
        snap = snapshots.get(ticker, {})
        news = news_map.get(ticker, {})

        price      = snap.get("price")
        prev_close = snap.get("prev_close")
        if not price or not prev_close or prev_close == 0:
            continue

        gap  = (price - prev_close) / prev_close * 100
        pol  = news.get("avg_polarity") or 0.0
        hcnt = news.get("headline_count") or 0

        if gap > 2.0:
            if pol > 0.1 and hcnt >= 2:
                gap_ups_with_news.append((ticker, gap, pol))
                logger.info(f"[premarket] GAP+NEWS: {ticker} +{gap:.1f}% polarity={pol:.2f}")
            else:
                gap_ups_plain.append((ticker, gap))
                logger.info(f"[premarket] GAP UP: {ticker} +{gap:.1f}%")

    # Promote gap+news stocks into discovered_tickers so 9:30 picks them up
    if gap_ups_with_news:
        discovered_data = _load_discovered()
        existing = set(discovered_data.get("tickers", []))
        meta     = discovered_data.get("meta", {})

        for ticker, gap, pol in sorted(gap_ups_with_news, key=lambda x: x[1], reverse=True)[:5]:
            if ticker not in existing:
                existing.add(ticker)
                meta[ticker] = {
                    "ticker":        ticker,
                    "gap_pct":       round(gap, 2),
                    "news_polarity": round(pol, 2),
                    "gap_catalyst":  True,
                    "source":        "premarket_gap_news",
                }
                logger.info(f"[premarket] Promoted gap-catalyst: {ticker} +{gap:.1f}%")

        _save_discovered({"tickers": list(existing), "meta": meta})

    console.print(f"[cyan]Gap Up + News ({len(gap_ups_with_news)}): {[(t, round(g,1)) for t,g,_ in gap_ups_with_news]}[/cyan]")
    console.print(f"[dim]Gap Up plain  ({len(gap_ups_plain)}): {[(t, round(g,1)) for t,g in gap_ups_plain]}[/dim]")
    console.print(f"[bold green]Pre-open scored decisions written to live feed.[/bold green]")

    from bot.logger import log_scan
    log_scan("premarket", len(all_tickers), len(all_decisions),
             0, len(gap_ups_with_news) + len(gap_ups_plain), 0)




def session_continuous(alpaca_client, data_client) -> None:
    """
    Runs all day from market open to close, scanning every SCAN_INTERVAL minutes.
    Replaces the separate market_open + midday + market_close jobs when running
    as a long-lived GitHub Actions job (timeout-minutes: 390).

    Loop behaviour:
      - Every cycle: check stops/targets/time-exits on open positions
      - Every cycle: scan all tickers for new signals and execute if qualified
      - First cycle (9:30-9:45 AM): treated as market_open — wider signal net
      - 3:45 PM onward: close all scalps, check signal flips, then exit loop
    """
    import time as _time
    from datetime import datetime, timezone, timedelta
    from zoneinfo import ZoneInfo
    from bot.portfolio import (check_stops, check_targets, check_time_exits,
                               get_open_positions, close_position_and_log,
                               calculate_partial_exit, update_breakout_stops)
    from bot.discovery import scan_rising_movers

    SCAN_INTERVAL = 10          # minutes between scans
    MOVER_SCAN_EVERY = 3        # run rising-movers screen every N cycles (~15 min)
    MARKET_OPEN_ET  = (9,  30)  # 9:30 AM ET
    SCALP_CLOSE_ET  = (15, 45)  # 3:45 PM ET — close scalps before market close
    LOOP_END_ET     = (16, 0)   # 4:00 PM ET — stop looping

    ET = ZoneInfo("America/New_York")  # handles EST/EDT automatically

    def _et_now():
        return datetime.now(ET)

    def _et_hm():
        n = _et_now()
        return (n.hour, n.minute)

    console.rule("[bold cyan]CONTINUOUS TRADING SESSION[/bold cyan]")
    console.print(f"[dim]Scanning every {SCAN_INTERVAL} min from 9:30 AM to 4:00 PM ET[/dim]")

    # Load gap-catalyst tickers flagged by premarket session
    from bot.discovery import _load_discovered
    _pm_disc = _load_discovered()
    gap_catalyst_tickers = [
        t for t, m in _pm_disc.get("meta", {}).items()
        if m.get("gap_catalyst")
    ]
    if gap_catalyst_tickers:
        console.print(f"[bold yellow]Gap-catalyst tickers from pre-market: {gap_catalyst_tickers}[/bold yellow]")

    # ── One-time session setup ─────────────────────────────────────────────
    global _session_start_equity, _daily_loss_halt
    _daily_loss_halt = False
    _session_start_equity = None  # will be set on first execute_signals call

    # Initial earnings proximity check for any already-open positions
    try:
        from bot.logger import get_open_trades as _got_init
        _check_earnings_proximity(_got_init())
    except Exception as _e:
        logger.warning(f"[main] initial earnings check failed: {_e}")

    cycle = 0
    extra_tickers: list[str] = []   # rising movers appended each mover-scan cycle
    while True:
        now_hm = _et_hm()

        # Wait for market open
        if now_hm < MARKET_OPEN_ET:
            wait_sec = ((MARKET_OPEN_ET[0] - now_hm[0]) * 60 +
                        (MARKET_OPEN_ET[1] - now_hm[1])) * 60
            console.print(f"[dim]Pre-open — waiting {wait_sec//60}m for 9:30 AM ET...[/dim]")
            _time.sleep(min(wait_sec, 60))
            continue

        # Past 4 PM — done
        if now_hm >= LOOP_END_ET:
            console.print("[bold]4:00 PM ET reached — continuous session complete.[/bold]")
            break

        cycle += 1
        ts = _et_now().strftime("%H:%M")
        console.rule(f"[dim]Cycle {cycle} — {ts} ET[/dim]")

        macro = get_macro_context()

        # ── First cycle: include gap-catalyst tickers immediately ─────────
        if cycle == 1 and gap_catalyst_tickers:
            extra_tickers = gap_catalyst_tickers
            console.print(f"[bold yellow]Cycle 1: scanning gap-catalyst tickers first: {extra_tickers}[/bold yellow]")

        # ── Rising movers screen (every MOVER_SCAN_EVERY cycles) ──────────
        if cycle % MOVER_SCAN_EVERY == 1:
            fresh_movers = scan_rising_movers(STATIC_TICKERS)
            # Merge with gap-catalysts on first cycle, replace on subsequent
            if cycle == 1:
                for t in fresh_movers:
                    if t not in extra_tickers:
                        extra_tickers.append(t)
            else:
                extra_tickers = fresh_movers
            if extra_tickers:
                console.print(f"[bold cyan]Rising movers: {extra_tickers}[/bold cyan]")

        # ── Earnings proximity check ───────────────────────────────────────
        try:
            from bot.logger import get_open_trades as _got_earnings
            _check_earnings_proximity(_got_earnings())
        except Exception as _earn_err:
            logger.warning(f"[main] earnings proximity check error: {_earn_err}")

        # ── Exit checks on all open positions ─────────────────────────────
        stopped  = check_stops(alpaca_client)
        targeted = check_targets(alpaca_client)
        timed    = check_time_exits(alpaca_client)

        for pos in stopped:
            cp = pos.get("current_price") or pos.get("entry_price", 0)
            close_position_and_log(alpaca_client, pos, cp, "continuous", status="stopped")
            console.print(f"[red]STOP: {pos['ticker']} @ {cp}[/red]")

        for pos in targeted:
            cp = pos.get("current_price") or pos.get("entry_price", 0)
            close_position_and_log(alpaca_client, pos, cp, "continuous", status="target_hit")
            console.print(f"[green]TARGET: {pos['ticker']} @ {cp}[/green]")

        for pos in timed:
            cp = pos.get("current_price") or pos.get("entry_price", 0)
            pnl = pos.get("pnl_pct", 0)
            close_position_and_log(alpaca_client, pos, cp, "continuous", status="time_exit")
            console.print(f"[yellow]TIME EXIT: {pos['ticker']} age={pos.get('age_days')}d pnl={pnl:+.1f}%[/yellow]")

        # ── Trailing stop checks ───────────────────────────────────────────
        from bot.risk import update_trailing_stop
        from bot.logger import update_trade_trailing
        for pos in get_open_positions(alpaca_client):
            current_price = pos.get("current_price") or pos.get("entry_price")
            if not current_price:
                continue
            updated = update_trailing_stop(pos, current_price)
            if updated.get("trailing_stop_updated") and pos.get("id"):
                update_trade_trailing(
                    pos["id"],
                    updated["highest_price_seen"],
                    updated.get("trailing_stop_price") or 0,
                )
            if updated.get("trailing_stop_triggered"):
                close_position_and_log(alpaca_client, pos, current_price, "continuous",
                                       status="trailing_stop")
                console.print(
                    f"[red]TRAILING STOP: {pos['ticker']} @ {current_price} "
                    f"(trail={updated.get('trailing_stop_price', 0):.2f})[/red]"
                )

        # ── Breakout chandelier stop updates ──────────────────────────────
        try:
            update_breakout_stops(alpaca_client, data_client)
        except Exception as _bse:
            logger.warning(f"[main] update_breakout_stops error: {_bse}")

        # ── Partial-exit checks ────────────────────────────────────────────
        from bot.trader import submit_order as _submit, get_latest_quote as _quote
        from bot.logger import update_trade_stop as _upd_stop
        for pos in get_open_positions(alpaca_client):
            if pos.get("action", "buy") != "buy":
                continue
            cp = pos.get("current_price") or pos.get("entry_price")
            if not cp:
                continue
            partial = calculate_partial_exit(pos, float(cp))
            if partial["close_pct"] <= 0:
                continue
            shares_to_close = partial["shares_to_close"]
            reason          = partial["reason"]
            try:
                q  = _quote(data_client, pos["ticker"])
                lp = float(q.get("ask_price") or q.get("bid_price") or cp) * 0.999
                _submit(
                    client=alpaca_client,
                    ticker=pos["ticker"],
                    side="sell",
                    qty=shares_to_close,
                    limit_price=round(lp, 2),
                    dry_run=DRY_RUN,
                )
                console.print(
                    f"[cyan]PARTIAL EXIT ({reason}): {shares_to_close}x {pos['ticker']} "
                    f"@ ${float(cp):.2f} ({partial['close_pct']*100:.0f}%)[/cyan]"
                )
                if partial["new_stop"] and pos.get("id"):
                    try:
                        _upd_stop(pos["id"], partial["new_stop"])
                    except Exception:
                        pass
            except Exception as _pe:
                logger.warning(f"[continuous] partial exit failed for {pos['ticker']}: {_pe}")

        # ── Scalp close at 3:45 PM ─────────────────────────────────────────
        if now_hm >= SCALP_CLOSE_ET:
            open_pos = get_open_positions(alpaca_client)
            for pos in open_pos:
                if pos.get("time_horizon") == "scalp":
                    cp = pos.get("current_price") or pos.get("entry_price", 0)
                    close_position_and_log(alpaca_client, pos, cp, "continuous", status="closed")
                    console.print(f"[yellow]EOD scalp close: {pos['ticker']} @ {cp}[/yellow]")

        # ── Re-score open positions for thesis check ──────────────────────
        _rescore_open_positions(
            tickers=get_all_trade_tickers(),
            alpaca_client=alpaca_client,
            data_client=data_client,
            dry_run=DRY_RUN,
        )

        # ── Scan for new signals ───────────────────────────────────────────
        session_label = "market_open" if cycle == 1 else "continuous"
        signals = run_full_scan(session_label, macro, alpaca_client, data_client,
                                extra_tickers=extra_tickers)
        if signals:
            execute_signals(signals, alpaca_client, data_client, macro,
                            session_label, max_trades=MAX_TRADES_PER_SESSION)

        # ── Signal flip closes ─────────────────────────────────────────────
        open_pos   = get_open_positions(alpaca_client)
        scored_map = {s["ticker"]: s for s in signals}
        for pos in open_pos:
            ticker = pos["ticker"]
            if ticker in scored_map:
                s = scored_map[ticker]
                if pos.get("action") == "buy" and s.get("action") in ("short", "sell"):
                    cp = pos.get("current_price") or pos.get("entry_price", 0)
                    close_position_and_log(alpaca_client, pos, cp, "continuous", status="closed")
                    console.print(f"[red]Signal flip: {ticker}[/red]")

        # Sleep until next scan (loop exits at LOOP_END_ET = 4:00 PM)


        # Sleep until next scan
        _time.sleep(SCAN_INTERVAL * 60)


def session_eod_summary(alpaca_client) -> None:
    """4:15 PM EDT - compute daily P&L, write summary, print Rich report."""
    console.rule("[bold white]END OF DAY SUMMARY[/bold white]")

    from bot.logger import get_trades_today, get_daily_summaries, log_daily_summary
    from bot.risk   import is_kill_switch_active
    from bot.trader import get_account

    account         = get_account(alpaca_client)
    portfolio_value = account.get("portfolio_value", 0)
    cash            = account.get("cash", 0)

    trades_today = get_trades_today()
    closed   = [t for t in trades_today if t.get("pnl_dollar") is not None]
    winners  = [t for t in closed if (t.get("pnl_dollar") or 0) > 0]
    losers   = [t for t in closed if (t.get("pnl_dollar") or 0) <= 0]
    gross_pnl = sum(float(t.get("pnl_dollar") or 0) for t in closed)
    win_rate  = len(winners) / len(closed) if closed else 0

    best  = max(closed, key=lambda t: t.get("pnl_dollar") or 0, default=None)
    worst = min(closed, key=lambda t: t.get("pnl_dollar") or 0, default=None)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    macro = get_macro_context()

    log_daily_summary(
        date=today,
        starting_value=portfolio_value - gross_pnl,
        ending_value=portfolio_value,
        cash=cash,
        total_trades=len(closed),
        winning_trades=len(winners),
        losing_trades=len(losers),
        gross_pnl=gross_pnl,
        win_rate=win_rate,
        best_trade=f"{best['ticker']} ${best['pnl_dollar']:.2f}"  if best  else "N/A",
        worst_trade=f"{worst['ticker']} ${worst['pnl_dollar']:.2f}" if worst else "N/A",
        macro_bias=macro.get("spy_regime", "unknown"),
        vix_level=macro.get("vix", 0),
        kill_switch_triggered=is_kill_switch_active(),
    )

    from rich.table import Table
    from rich.panel import Panel

    pnl_color = "green" if gross_pnl >= 0 else "red"
    best_str  = f"{best['ticker']} ${float(best.get('pnl_dollar') or 0):.2f}"  if best  else "N/A $0.00"
    worst_str = f"{worst['ticker']} ${float(worst.get('pnl_dollar') or 0):.2f}" if worst else "N/A $0.00"
    console.print(Panel(
        f"[bold]Date:[/bold] {today}\n"
        f"[bold]Portfolio Value:[/bold] ${portfolio_value:,.2f}\n"
        f"[bold]Cash:[/bold] ${cash:,.2f}\n"
        f"[bold]Daily P&L:[/bold] [{pnl_color}]${gross_pnl:,.2f}[/{pnl_color}]\n"
        f"[bold]Win Rate:[/bold] {win_rate*100:.1f}% ({len(winners)}W / {len(losers)}L)\n"
        f"[bold]Total Closed Trades:[/bold] {len(closed)}\n"
        f"[bold]Best:[/bold] {best_str}\n"
        f"[bold]Worst:[/bold] {worst_str}\n"
        f"[bold]VIX:[/bold] {macro['vix']:.1f}  [bold]Regime:[/bold] {macro['spy_regime']}",
        title="[bold]Daily Summary[/bold]",
        border_style=pnl_color,
    ))

    summaries = get_daily_summaries(7)
    if summaries:
        table = Table(title="Last 7 Days P&L")
        table.add_column("Date")
        table.add_column("P&L", justify="right")
        table.add_column("Win%", justify="right")
        table.add_column("Trades", justify="right")
        for s in reversed(summaries):
            p = float(s.get("gross_pnl") or 0)
            c = "green" if p >= 0 else "red"
            table.add_row(
                s["date"],
                f"[{c}]${p:,.2f}[/{c}]",
                f"{(s.get('win_rate') or 0)*100:.1f}%",
                str(s.get("total_trades") or 0),
            )
        console.print(table)

    # Patch live feed with EOD P&L so dashboard shows daily totals
    try:
        from bot.live_feed import write_eod_summary
        write_eod_summary(today, gross_pnl, len(closed), win_rate, portfolio_value)
    except Exception as e:
        logger.warning(f"[eod] live feed EOD patch failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST SESSION
# ══════════════════════════════════════════════════════════════════════════════

def session_backtest(days: int = 30, relaxed: bool = False) -> None:
    """
    Walk-forward backtest over the last N trading days.

    Re-scores every SCAN_STEP trading days throughout the full window.
    At each step: compute SPY regime, check existing positions for exits,
    then score free tickers and enter new positions.
    This avoids the single-point-in-time problem where one bad entry date
    suppresses all signals for the entire window.

    relaxed=True lowers thresholds to net_score>=40 / confidence>=0.60.
    Live trading is unaffected.
    """
    from bot.data import fetch_daily_bars, fetch_vix
    import numpy as np
    from bot.indicators import compute_indicators_from_df
    from bot.scorer     import score_ticker
    from bot.strategies import classify_strategy
    from rich.table     import Table
    from rich.panel     import Panel

    SCAN_STEP = 5   # re-score every 5 trading days (weekly)

    mode_label = "RELAXED" if relaxed else "STRICT"
    console.rule(f"[bold]WALK-FORWARD BACKTEST - Last {days} Trading Days ({mode_label})[/bold]")
    console.print("[dim]Re-scoring every 5 trading days throughout the window...[/dim]")
    if relaxed:
        console.print("[yellow]Relaxed mode: net_score>=40, confidence>=0.60 — for signal evaluation only[/yellow]")

    sim_news = {"avg_polarity": 0.0, "headline_count": 0, "top_headlines": [],
                "sec_8k_flag": False, "earnings_risk": False}

    # ── Fetch full history upfront ────────────────────────────────────────
    fetch_period = f"{days + 300}d"
    tickers_to_fetch = get_all_trade_tickers()

    console.print(f"[dim]Fetching {len(tickers_to_fetch)} tickers + SPY/VIX history...[/dim]")
    all_hist: dict = {}
    for ticker in tickers_to_fetch:
        try:
            h = fetch_daily_bars(ticker, days=730)
            if h is not None and len(h) >= 60:
                all_hist[ticker] = h
        except Exception as e:
            logger.warning(f"[backtest] fetch {ticker}: {e}")

    try:
        spy_full = fetch_daily_bars("SPY", days=730)
    except Exception:
        spy_full = None

    try:
        vix_full = fetch_vix(days=730)
    except Exception:
        vix_full = None

    # SPY buy-and-hold over full window
    spy_return = 0.0
    if spy_full is not None and len(spy_full) >= days:
        spy_bh_entry = float(spy_full["Close"].iloc[-days])
        spy_bh_exit  = float(spy_full["Close"].iloc[-1])
        spy_return   = (spy_bh_exit - spy_bh_entry) / spy_bh_entry * 100

    # ── Walk-forward engine ───────────────────────────────────────────────
    sim_trades: list[dict] = []
    # open_positions: ticker -> {entry_price, stop, target, action, horizon,
    #                            max_hold, entry_step_idx, score_meta}
    open_positions: dict = {}

    # Total trading bars available (use SPY as timeline anchor)
    if spy_full is None or len(spy_full) < days + 10:
        console.print("[red]Insufficient SPY history for walk-forward backtest.[/red]")
        return

    total_bars = len(spy_full)
    # window_start_idx: index into spy_full where the backtest window begins
    window_start_idx = total_bars - days

    def _spy_regime_at(bar_idx: int) -> tuple[str, bool, float]:
        """Compute SPY regime and VIX at a given bar index."""
        if bar_idx < 50:
            return "bull", False, 18.0
        spy_slice = spy_full["Close"].iloc[:bar_idx]
        ema50  = float(spy_slice.ewm(span=50,  adjust=False).mean().iloc[-1])
        ema200 = float(spy_slice.ewm(span=200, adjust=False).mean().iloc[-1])
        price  = float(spy_slice.iloc[-1])
        if price > ema50 and price > ema200:
            regime, bearish = "bull", False
        elif price < ema50:
            regime, bearish = "caution", True
        else:
            regime, bearish = "bear", True
        vix_val = 18.0
        if vix_full is not None and bar_idx < len(vix_full):
            try:
                vix_val = float(vix_full["Close"].iloc[bar_idx - 1])
            except Exception:
                pass
        return regime, bearish, vix_val

    def _try_promote_relaxed(action, score, ind):
        """In relaxed mode, promote a 'hold' to buy/short if nearly qualified."""
        if not relaxed or action != "hold":
            return action
        net  = score.get("net_score", 0)
        conf = score.get("confidence", 0.0)
        if net >= 40 and conf >= 0.60:
            return "buy"
        if net <= -40 and conf >= 0.60:
            e50   = ind.get("ema50") or 0
            cp    = ind.get("current_price") or 0
            rsi   = ind.get("rsi") or 50
            adx   = ind.get("adx") or 0
            dip   = ind.get("adx_di_plus") or 0
            dim   = ind.get("adx_di_minus") or 0
            bb_pb = ind.get("bb_pctb")
            short_extreme = (
                (e50 > 0 and cp < e50) or (rsi > 75) or
                (bb_pb is not None and bb_pb > 0.95)
            )
            if short_extreme and not (adx > 30 and dip > 0 and dip > dim):
                return "short"
        return action

    # Scan steps: every SCAN_STEP bars from window_start_idx to end
    scan_steps = list(range(window_start_idx, total_bars, SCAN_STEP))
    if not scan_steps or scan_steps[-1] < total_bars - 1:
        scan_steps.append(total_bars - 1)

    console.print(f"[dim]{len(scan_steps)} scan steps over {days} trading days[/dim]")
    console.print("[dim]Press [bold]Enter[/bold] at any time to stop and show results so far.[/dim]")

    # Background thread: set stop_early when user presses Enter
    import threading
    stop_early = threading.Event()
    def _watch_enter():
        try:
            input()
        except Exception:
            pass
        stop_early.set()
    threading.Thread(target=_watch_enter, daemon=True).start()

    # Silence all INFO logs to terminal during the scan — they'd drown the progress.
    # Logs still go to the log file. Warnings/errors remain visible.
    _bt_handler = None
    for h in logging.getLogger().handlers:
        if isinstance(h, RichHandler):
            _bt_handler = h
            break
    if _bt_handler:
        _bt_handler.setLevel(logging.WARNING)

    for step_num, bar_idx in enumerate(scan_steps):
        if stop_early.is_set():
            console.print(f"[yellow]Stopped early at step {step_num}/{len(scan_steps)} — showing results so far.[/yellow]")
            break
        regime, bearish, vix_val = _spy_regime_at(bar_idx)
        sim_macro = {
            "vix":            vix_val,
            "spy_regime":     regime,
            "bearish_market": bearish,
            "vix_multiplier": 1.0,
        }

        if step_num % 10 == 0:
            pct = step_num / len(scan_steps) * 100
            console.print(
                f"[dim]  Step {step_num+1}/{len(scan_steps)} ({pct:.0f}%)  "
                f"open={len(open_positions)}  closed={len(sim_trades)}  "
                f"regime={regime}[/dim]",
                end="\r",
            )

        # ── Check exits for open positions ────────────────────────────────
        closed_tickers = []
        for tk, pos in open_positions.items():
            hist = all_hist.get(tk)
            if hist is None:
                closed_tickers.append(tk)
                continue

            # Map the walk-forward bar_idx to this ticker's history
            # (SPY and individual ticker bars may differ slightly; align by count)
            # We stored entry_bar_abs (absolute index into this ticker's history)
            entry_bar = pos["entry_bar_abs"]
            age_days  = bar_idx - pos["entry_bar_spy"]

            action    = pos["action"]
            stop      = pos["stop"]
            target    = pos["target"]
            max_hold  = pos["max_hold"]

            # Check each day since last scan
            last_checked = pos.get("last_checked_bar", entry_bar)
            check_end    = min(entry_bar + age_days + 1, len(hist))
            exit_price   = None
            exit_status  = None

            for j in range(last_checked, check_end):
                if j >= len(hist):
                    break
                day_low  = float(hist["Low"].iloc[j])
                day_high = float(hist["High"].iloc[j])
                if action == "buy":
                    if stop   and day_low  <= stop:
                        exit_price = stop;   exit_status = "stopped";    break
                    if target and day_high >= target:
                        exit_price = target; exit_status = "target_hit"; break
                else:
                    if stop   and day_high >= stop:
                        exit_price = stop;   exit_status = "stopped";    break
                    if target and day_low  <= target:
                        exit_price = target; exit_status = "target_hit"; break
            else:
                pos["last_checked_bar"] = check_end

            if exit_price is None and age_days >= max_hold:
                close_bar  = min(entry_bar + max_hold, len(hist) - 1)
                exit_price  = float(hist["Close"].iloc[close_bar])
                exit_status = "time_exit"

            if exit_price is not None:
                entry_p = pos["entry_price"]
                if action == "buy":
                    pnl_pct = (exit_price - entry_p) / entry_p * 100
                else:
                    pnl_pct = (entry_p - exit_price) / entry_p * 100

                nat_bar   = min(entry_bar + max_hold, len(hist) - 1)
                nat_price = float(hist["Close"].iloc[nat_bar])
                if action == "buy":
                    natural_pnl = (nat_price - entry_p) / entry_p * 100
                else:
                    natural_pnl = (entry_p - nat_price) / entry_p * 100

                exit_days = age_days
                meta = pos["score_meta"]
                sim_trades.append({
                    "ticker":       tk,
                    "action":       action,
                    "strategy":     meta.get("strategy", "?"),
                    "horizon":      pos["horizon"],
                    "confidence":   meta.get("confidence", 0),
                    "net_score":    meta.get("net_score", 0),
                    "entry":        entry_p,
                    "exit":         exit_price,
                    "exit_day":     exit_days,
                    "status":       exit_status,
                    "pnl_pct":      pnl_pct,
                    "natural_pnl":  natural_pnl,
                    "natural_exit": nat_price,
                    "entry_step":   step_num,
                })
                logger.debug(
                    f"[backtest] CLOSE {tk} {action} entry=${entry_p:.2f} "
                    f"exit=${exit_price:.2f} pnl={pnl_pct:+.2f}% ({exit_status}) day={exit_days}"
                )
                closed_tickers.append(tk)

        for tk in closed_tickers:
            open_positions.pop(tk, None)

        # ── Score free tickers ────────────────────────────────────────────
        for ticker in tickers_to_fetch:
            if ticker in open_positions:
                continue
            hist = all_hist.get(ticker)
            if hist is None or bar_idx >= len(hist):
                continue
            hist_slice = hist.iloc[:bar_idx].copy()
            if len(hist_slice) < 50:
                continue

            try:
                ind = compute_indicators_from_df(ticker, hist_slice,
                                                 intraday=None, realtime_price=False)
                if ind.get("error"):
                    continue

                score  = score_ticker(ticker, ind, sim_news, sim_macro)
                score  = classify_strategy(score, ind)
                action = score["action"]
                action = _try_promote_relaxed(action, score, ind)

                if relaxed:
                    score["action"] = action

                if action not in ("buy", "short"):
                    continue
                if bearish and action == "buy":
                    continue

                # Entry price: next bar's open
                entry_bar = bar_idx
                if entry_bar >= len(hist):
                    continue
                entry_price = float(hist["Open"].iloc[entry_bar])
                if entry_price <= 0:
                    continue

                atr    = ind.get("atr") or (entry_price * 0.02)
                stop   = score.get("stop_loss")
                target = score.get("take_profit")

                # High-volatility tickers need 4× ATR stops to survive normal noise
                stop_mult = 4.0 if score.get("high_vol_flag") else 3.0
                if action == "buy":
                    hard_stop = entry_price - atr * stop_mult
                    stop = max(stop, hard_stop) if stop else hard_stop
                else:
                    hard_stop = entry_price + atr * stop_mult
                    stop = min(stop, hard_stop) if stop else hard_stop

                horizon  = score.get("time_horizon", "swing")
                max_hold = {"scalp": 5, "swing": 20, "position": 45}.get(horizon, 20)

                open_positions[ticker] = {
                    "entry_price":       entry_price,
                    "stop":              stop,
                    "target":            target,
                    "action":            action,
                    "horizon":           horizon,
                    "max_hold":          max_hold,
                    "entry_bar_abs":     entry_bar,
                    "entry_bar_spy":     bar_idx,
                    "last_checked_bar":  entry_bar,
                    "score_meta":        score,
                }
                logger.debug(
                    f"[backtest] ENTER {ticker} {action} @ ${entry_price:.2f} "
                    f"stop={stop:.2f} target={target:.2f if target else 0:.2f} "
                    f"horizon={horizon} step={step_num}"
                )
            except Exception as e:
                logger.debug(f"[backtest] score {ticker} step {step_num}: {e}")

    # ── Close any still-open positions at end of window ──────────────────
    for tk, pos in open_positions.items():
        hist = all_hist.get(tk)
        if hist is None:
            continue
        entry_p     = pos["entry_price"]
        action      = pos["action"]
        close_bar   = min(pos["entry_bar_abs"] + pos["max_hold"], len(hist) - 1)
        exit_price  = float(hist["Close"].iloc[close_bar])
        age_days    = total_bars - pos["entry_bar_spy"]

        if action == "buy":
            pnl_pct = (exit_price - entry_p) / entry_p * 100
        else:
            pnl_pct = (entry_p - exit_price) / entry_p * 100

        meta = pos["score_meta"]
        sim_trades.append({
            "ticker":       tk,
            "action":       action,
            "strategy":     meta.get("strategy", "?"),
            "horizon":      pos["horizon"],
            "confidence":   meta.get("confidence", 0),
            "net_score":    meta.get("net_score", 0),
            "entry":        entry_p,
            "exit":         exit_price,
            "exit_day":     age_days,
            "status":       "open_at_end",
            "pnl_pct":      pnl_pct,
            "natural_pnl":  pnl_pct,
            "natural_exit": exit_price,
            "entry_step":   -1,
        })

    # Restore terminal log level
    if _bt_handler:
        _bt_handler.setLevel(logging.INFO)
    console.print()  # newline after progress line

    # ── Report ────────────────────────────────────────────────────────────
    winners = [t for t in sim_trades if t["pnl_pct"] > 0]
    losers  = [t for t in sim_trades if t["pnl_pct"] <= 0]
    total   = len(sim_trades)
    avg_win  = float(np.mean([t["pnl_pct"] for t in winners])) if winners else 0
    avg_loss = float(np.mean([t["pnl_pct"] for t in losers]))  if losers  else 0
    avg_pnl  = float(np.mean([t["pnl_pct"] for t in sim_trades])) if sim_trades else 0
    win_rate = len(winners) / total if total else 0

    best_trade  = max(sim_trades, key=lambda t: t["pnl_pct"], default=None)
    worst_trade = min(sim_trades, key=lambda t: t["pnl_pct"], default=None)

    bot_color = "green" if avg_pnl >= spy_return else "red"

    best_line  = (f"[bold]Best:[/bold]  {best_trade['ticker']} {best_trade['pnl_pct']:+.2f}% ({best_trade['status']})"
                  if best_trade else "[bold]Best:[/bold]  N/A")
    worst_line = (f"[bold]Worst:[/bold] {worst_trade['ticker']} {worst_trade['pnl_pct']:+.2f}% ({worst_trade['status']})"
                  if worst_trade else "[bold]Worst:[/bold] N/A")

    console.print(Panel(
        f"[bold]Period:[/bold] Last {days} trading days  |  Scan step: every {SCAN_STEP} days\n"
        f"[bold]Total Trades:[/bold] {total}\n"
        f"[bold]Win Rate:[/bold] {win_rate*100:.1f}% ({len(winners)}W / {len(losers)}L)\n"
        f"[bold]Avg Trade P&L:[/bold] [{bot_color}]{avg_pnl:+.2f}%[/{bot_color}]\n"
        f"[bold]Avg Winner:[/bold] +{avg_win:.2f}%   [bold]Avg Loser:[/bold] {avg_loss:.2f}%\n"
        f"{best_line}\n{worst_line}\n"
        f"[bold]SPY Buy-and-Hold:[/bold] {spy_return:+.2f}%",
        title=f"[bold]Walk-Forward Backtest Results ({days}d)[/bold]",
        border_style="cyan",
    ))

    if sim_trades:
        NOTIONAL = 10_000  # assumed dollars per trade for dollar P&L display
        total_dollar_pnl = sum(t["pnl_pct"] / 100 * NOTIONAL for t in sim_trades)
        table = Table(title=f"Simulated Trades  (assumes ${NOTIONAL:,} per trade)")
        table.add_column("Ticker",    style="bold")
        table.add_column("Action")
        table.add_column("Strategy")
        table.add_column("Conf",         justify="right")
        table.add_column("Entry",        justify="right")
        table.add_column("Exit",         justify="right")
        table.add_column("P&L %",        justify="right")
        table.add_column("P&L $",        justify="right")
        table.add_column("Days",         justify="right")
        table.add_column("Status")
        table.add_column("Natural P&L",  justify="right")

        for t in sorted(sim_trades, key=lambda x: x["pnl_pct"], reverse=True):
            p      = t["pnl_pct"]
            dollar = p / 100 * NOTIONAL
            np_    = t.get("natural_pnl", 0)
            c      = "green" if p   > 0 else "red"
            nc     = "green" if np_ > 0 else "red"
            table.add_row(
                t["ticker"],
                t["action"],
                t["strategy"],
                f"{t['confidence']:.2f}",
                f"${t['entry']:.2f}",
                f"${t['exit']:.2f}",
                f"[{c}]{p:+.2f}%[/{c}]",
                f"[{c}]{dollar:+.0f}[/{c}]",
                str(t["exit_day"]),
                t["status"],
                f"[{nc}]{np_:+.2f}%[/{nc}]",
            )
        tc = "green" if total_dollar_pnl >= 0 else "red"
        console.print(table)
        console.print(f"  Total P&L on ${NOTIONAL:,}/trade: [{tc}]{total_dollar_pnl:+,.0f}[/{tc}]")


def session_holdings(alpaca_client=None) -> None:
    """Print open positions from DB, enriched with live Alpaca prices if available."""
    from bot.logger import get_open_trades
    from rich.table import Table

    positions = get_open_trades()
    if not positions:
        console.print("[bold yellow]No open positions.[/bold yellow]")
        return

    # Try to enrich with live prices
    live_prices: dict = {}
    if alpaca_client:
        try:
            from bot.trader import get_positions
            for p in get_positions(alpaca_client):
                live_prices[p["symbol"]] = p
        except Exception:
            pass

    table = Table(title="Open Positions")
    table.add_column("Ticker",   style="bold")
    table.add_column("Action")
    table.add_column("Qty",      justify="right")
    table.add_column("Entry",    justify="right")
    table.add_column("Current",  justify="right")
    table.add_column("P&L %",    justify="right")
    table.add_column("Stop",     justify="right")
    table.add_column("Target",   justify="right")
    table.add_column("Strategy")
    table.add_column("Entered")

    for pos in positions:
        ticker  = pos["ticker"]
        entry   = float(pos.get("entry_price") or 0)
        lp      = live_prices.get(ticker, {})
        current = lp.get("current_price") or entry
        if entry > 0 and current:
            pnl_pct = (current - entry) / entry * 100
            if pos.get("action") in ("short", "sell"):
                pnl_pct = -pnl_pct
        else:
            pnl_pct = 0.0
        color   = "green" if pnl_pct >= 0 else "red"
        cur_str = f"${float(current):.2f}" if current else "—"
        ts      = (pos.get("timestamp") or "")[:16]
        table.add_row(
            ticker,
            pos.get("action", ""),
            str(pos.get("quantity", "")),
            f"${entry:.2f}",
            cur_str,
            f"[{color}]{pnl_pct:+.2f}%[/{color}]",
            f"${float(pos.get('stop_loss') or 0):.2f}",
            f"${float(pos.get('take_profit') or 0):.2f}",
            pos.get("strategy", ""),
            ts,
        )

    console.print(table)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Trading Bot")
    parser.add_argument(
        "--session",
        required=True,
        choices=["discovery", "premarket", "eod_summary", "backtest", "continuous", "holdings"],
        help="Which session to run",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Number of days for backtest window (default: 30)",
    )
    parser.add_argument(
        "--relaxed",
        action="store_true",
        default=False,
        help="Backtest only: lower thresholds to net>=40/conf>=0.60 to surface more signals",
    )
    args    = parser.parse_args()
    session = args.session

    # Discovery and backtest skip market-open check and Alpaca setup
    if session == "discovery":
        from bot.logger import init_db
        init_db()
        session_discovery()
        return

    if session == "holdings":
        from bot.logger import init_db
        init_db()
        try:
            from bot.trader import build_client
            alpaca_client = build_client()
        except Exception:
            alpaca_client = None
        session_holdings(alpaca_client)
        return

    if session == "backtest":
        from bot.logger import init_db
        init_db()
        session_backtest(days=args.days, relaxed=args.relaxed)
        return

    # Market holiday check for all live sessions
    if not is_market_open_today():
        console.print(f"[bold yellow]Market closed today - skipping {session}.[/bold yellow]")
        sys.exit(0)

    if DRY_RUN:
        console.print("[bold yellow]DRY_RUN=true - orders will be simulated[/bold yellow]")

    logger.info(f"Starting session: {session}")

    from bot.logger import init_db
    init_db()

    # Build Alpaca clients
    try:
        from bot.trader import build_client, build_data_client
        alpaca_client = build_client()
        data_client   = build_data_client()
    except Exception as e:
        logger.error(f"Alpaca client init failed: {e}")
        if session not in ("premarket",):
            sys.exit(1)
        alpaca_client = None
        data_client   = None

    try:
        if session == "premarket":
            session_premarket()
        elif session == "eod_summary":
            session_eod_summary(alpaca_client)
        elif session == "continuous":
            session_continuous(alpaca_client, data_client)
    except Exception as e:
        logger.critical(f"Session {session} crashed: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Session {session} complete.")


if __name__ == "__main__":
    main()
