"""
Main entry point for the autonomous trading bot.
Routes to session-specific logic based on --session argument.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

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


def get_all_trade_tickers() -> list[str]:
    """Return static watchlist merged with any tickers promoted by discovery."""
    from bot.discovery import get_discovered_tickers
    discovered = get_discovered_tickers()
    combined = list(STATIC_TICKERS)
    for t in discovered:
        if t not in combined:
            combined.append(t)
    return combined

# Max trades per session to limit overexposure
MAX_TRADES_PER_SESSION = 3


def is_market_open_today() -> bool:
    nyse = mcal.get_calendar("NYSE")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty


def _has_open_position(ticker: str, alpaca_client=None) -> bool:
    """Return True if ticker has an open position (DB or live Alpaca) or was traded today."""
    from bot.logger import get_open_trades, get_trades_today
    for t in get_open_trades():
        if t.get("ticker") == ticker and t.get("status") in ("open", "dry_run"):
            return True
    # Also block if we already bought/shorted this ticker today
    for t in get_trades_today():
        if t.get("ticker") == ticker and t.get("action") in ("buy", "short"):
            return True
    # Cross-check Alpaca live positions (catches positions from previous runs not in DB)
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
    import yfinance as yf
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
        vix_data = yf.Ticker("^VIX").history(period="2d", interval="1d")
        if not vix_data.empty:
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

    all_tickers = get_all_trade_tickers()
    if extra_tickers:
        for t in extra_tickers:
            if t not in all_tickers:
                all_tickers.append(t)

    console.print(f"[bold cyan]Scanning {len(all_tickers)} tickers...[/bold cyan]")
    indicators_map = get_indicators_batch(all_tickers, max_workers=2)
    news_map       = get_news_batch(all_tickers, COMPANY_NAMES, api_key=NEWS_API_KEY, max_workers=3)

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

        try:
            score = score_ticker(ticker, ind, news, macro_context)
            score = classify_strategy(score, ind)
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

        # Store indicators for AI batch (run on all tickers)
        score["_indicators"] = ind
        score["_news"]       = news
        signals_all.append(score)

    # ── Claude second-opinion pass on every ticker ─────────────────────────
    from bot.ai_filter import run_ai_filter_batch
    pairs = [(s, s.get("_indicators", {})) for s in signals_all]
    signals_all = run_ai_filter_batch(pairs)

    # Re-tally after AI may have upgraded/downgraded actions
    for score in signals_all:
        action     = score.get("action", "hold")
        confidence = score.get("confidence", 0.0)

        # Re-apply strategy/confidence gates after AI changes
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

    from bot.logger import log_scan
    log_scan(
        session=session,
        tickers_scanned=len(get_all_trade_tickers()),
        signals_generated=len(signals),
        trades_executed=0,
        total_bull=bull_count,
        total_bear=bear_count,
    )
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
    from bot.logger import log_trade
    from bot.risk   import calculate_position, is_kill_switch_active, init_daily_state
    from bot.trader import (
        get_account, submit_order, compute_limit_price,
        get_latest_quote, check_order_filled,
    )

    if is_kill_switch_active():
        logger.warning("[execute] Kill switch active - no orders will be placed")
        return 0

    account         = get_account(alpaca_client)
    portfolio_value = account.get("portfolio_value", 100_000)
    init_daily_state(portfolio_value)

    # Sort by confidence descending, cap at max_trades
    ranked  = sorted(signals, key=lambda s: s.get("confidence", 0), reverse=True)
    ranked  = ranked[:max_trades]

    executed = 0
    for sig in ranked:
        if is_kill_switch_active():
            break

        ticker      = sig["ticker"]
        action      = sig["action"]
        confidence  = sig["confidence"]
        atr         = sig.get("atr") or (sig.get("entry_price", 100) * 0.02)
        entry_price = sig.get("entry_price") or 0
        strategy    = sig.get("strategy", "mixed")
        high_vol    = sig.get("high_vol_flag", False)

        if entry_price == 0:
            continue

        # ── Duplicate position guard ───────────────────────────────────────
        if _has_open_position(ticker, alpaca_client):
            logger.info(f"[SKIP] Already have open position in {ticker}")
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

        pos = calculate_position(
            portfolio_value=portfolio_value,
            confidence=confidence,
            atr=atr,
            price=entry_price,
            vix_multiplier=macro_context.get("vix_multiplier", 1.0),
            high_vol_flag=high_vol,
        )

        shares = pos["shares"]
        if shares <= 0:
            logger.info(f"[execute] {ticker}: 0 shares - skipping ({pos['reason']})")
            continue

        # Quote for limit price calculation; entry_price stays as the real-time price
        quote       = get_latest_quote(data_client, ticker)
        alpaca_side = "buy" if action == "buy" else "sell"
        limit_price = compute_limit_price(alpaca_side, quote, entry_price)

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
    """9:00 AM EDT - fetch overnight news, flag gap moves. No trades."""
    console.rule("[bold yellow]PRE-MARKET SESSION[/bold yellow]")
    macro = get_macro_context()

    from bot.indicators import get_indicators_batch
    from bot.news       import get_news_batch

    NEWS_API_KEY   = os.getenv("NEWS_API_KEY", "")
    indicators_map = get_indicators_batch(get_all_trade_tickers(), max_workers=2)
    news_map       = get_news_batch(get_all_trade_tickers(), COMPANY_NAMES, api_key=NEWS_API_KEY, max_workers=3)

    gap_ups   = []
    gap_downs = []

    for ticker in get_all_trade_tickers():
        ind  = indicators_map.get(ticker, {})
        news = news_map.get(ticker, {})
        gap  = ind.get("gap_pct")

        if gap is not None:
            if gap > 2.0:
                gap_ups.append((ticker, gap))
                logger.info(f"[premarket] GAP UP: {ticker} +{gap:.2f}%")
            elif gap < -2.0:
                gap_downs.append((ticker, gap))
                logger.info(f"[premarket] GAP DOWN: {ticker} {gap:.2f}%")

        pol = news.get("avg_polarity", 0)
        if abs(pol) > 0.3:
            logger.info(
                f"[premarket] News signal: {ticker} polarity={pol:.2f} "
                f"headlines={news.get('headline_count', 0)}"
            )

    console.print(f"[cyan]Gap Ups (>2%):    {gap_ups}[/cyan]")
    console.print(f"[red]Gap Downs (<-2%): {gap_downs}[/red]")
    console.print(f"[bold]VIX={macro['vix']:.1f}  Regime={macro['spy_regime']}  "
                  f"BearishMarket={macro['bearish_market']}[/bold]")

    from bot.logger import log_scan
    log_scan("premarket", len(get_all_trade_tickers()), 0, 0, len(gap_ups), len(gap_downs))


def session_market_open(alpaca_client, data_client) -> None:
    """9:35 AM EDT - full score run, execute top-3 signals above threshold."""
    console.rule("[bold green]MARKET OPEN SESSION[/bold green]")
    macro = get_macro_context()

    if macro["vix"] > 35:
        console.print("[bold red]VIX > 35 - EXTREME FEAR. No new positions.[/bold red]")
        return

    signals  = run_full_scan("market_open", macro, alpaca_client, data_client)
    executed = execute_signals(signals, alpaca_client, data_client, macro, "market_open",
                               max_trades=MAX_TRADES_PER_SESSION)
    console.print(f"[bold green]Market open complete: {executed} trades executed[/bold green]")


def session_midday(alpaca_client, data_client) -> None:
    """12:00 PM EDT - check stops/targets. New entries only on extreme conviction."""
    console.rule("[bold blue]MIDDAY SESSION[/bold blue]")

    from bot.portfolio import check_stops, check_targets, check_time_exits, close_position_and_log
    macro = get_macro_context()

    # Check stops
    breached = check_stops(alpaca_client)
    for pos in breached:
        cp = pos.get("current_price") or pos.get("entry_price", 0)
        close_position_and_log(alpaca_client, pos, cp, "midday", status="stopped")
        console.print(f"[red]STOP LOSS: {pos['ticker']} closed @ {cp}[/red]")

    # Check targets
    targets_hit = check_targets(alpaca_client)
    for pos in targets_hit:
        cp = pos.get("current_price") or pos.get("entry_price") or 0
        close_position_and_log(alpaca_client, pos, cp, "midday", status="closed")
        console.print(f"[green]TAKE PROFIT: {pos['ticker']} closed @ {cp}[/green]")

    # Time-based exits — close stale positions regardless of P&L
    expired = check_time_exits(alpaca_client)
    for pos in expired:
        cp     = pos.get("current_price") or pos.get("entry_price") or 0
        pnl    = pos.get("pnl_pct", 0)
        age    = pos.get("age_days", 0)
        color  = "green" if pnl >= 0 else "yellow"
        close_position_and_log(alpaca_client, pos, cp, "midday", status="time_exit")
        console.print(f"[{color}]TIME EXIT: {pos['ticker']} age={age}d pnl={pnl:+.1f}% closed @ {cp}[/{color}]")

    # New entries only at extreme confidence — net_score > 80 AND confidence >= 0.85
    signals   = run_full_scan("midday", macro, alpaca_client, data_client)
    high_conf = [
        s for s in signals
        if s.get("net_score", 0) > 80 and s.get("confidence", 0) >= 0.85
    ]
    if high_conf:
        executed = execute_signals(high_conf, alpaca_client, data_client, macro, "midday",
                                   max_trades=1)   # only 1 new trade midday max
        console.print(f"[bold]High-conviction midday entry: {executed}[/bold]")
    else:
        console.print("[dim]No extreme-conviction entries (net>80, conf>=0.85) at midday[/dim]")


def session_market_close(alpaca_client, data_client) -> None:
    """3:30 PM EDT - close scalps, re-score, decide what to hold overnight."""
    console.rule("[bold magenta]MARKET CLOSE SESSION[/bold magenta]")

    from bot.portfolio import get_open_positions, close_position_and_log, check_time_exits, check_stops, check_targets
    macro = get_macro_context()

    # Check stops and targets before anything else
    for pos in check_stops(alpaca_client):
        cp = pos.get("current_price") or pos.get("entry_price", 0)
        close_position_and_log(alpaca_client, pos, cp, "market_close", status="stopped")
        console.print(f"[red]STOP LOSS: {pos['ticker']} closed @ {cp}[/red]")

    for pos in check_targets(alpaca_client):
        cp = pos.get("current_price") or pos.get("entry_price") or 0
        close_position_and_log(alpaca_client, pos, cp, "market_close", status="closed")
        console.print(f"[green]TAKE PROFIT: {pos['ticker']} closed @ {cp}[/green]")

    # Time-based exits
    for pos in check_time_exits(alpaca_client):
        cp    = pos.get("current_price") or pos.get("entry_price") or 0
        pnl   = pos.get("pnl_pct", 0)
        age   = pos.get("age_days", 0)
        color = "green" if pnl >= 0 else "yellow"
        close_position_and_log(alpaca_client, pos, cp, "market_close", status="time_exit")
        console.print(f"[{color}]TIME EXIT: {pos['ticker']} age={age}d pnl={pnl:+.1f}% closed @ {cp}[/{color}]")

    # Close all scalp positions at EOD regardless
    open_positions = get_open_positions(alpaca_client)
    for pos in open_positions:
        if pos.get("time_horizon") == "scalp":
            cp = pos.get("current_price") or pos.get("entry_price", 0)
            close_position_and_log(alpaca_client, pos, cp, "market_close", status="closed")
            console.print(f"[yellow]EOD scalp close: {pos['ticker']} @ {cp}[/yellow]")

    # Re-score
    signals = run_full_scan("market_close", macro, alpaca_client, data_client)

    # Overnight entries: raised thresholds vs market_open
    overnight = [
        s for s in signals
        if abs(s.get("net_score", 0)) >= 65 and s.get("confidence", 0) >= 0.65
           and s.get("strategy") != "mixed"
    ]
    if overnight:
        executed = execute_signals(overnight, alpaca_client, data_client, macro,
                                   "market_close", max_trades=MAX_TRADES_PER_SESSION)
        console.print(f"[bold]Overnight holds initiated: {executed}[/bold]")

    # Close positions where signal has flipped
    open_positions = get_open_positions(alpaca_client)
    scored_map = {s["ticker"]: s for s in signals}
    for pos in open_positions:
        ticker = pos["ticker"]
        if ticker in scored_map:
            s = scored_map[ticker]
            if pos.get("action") == "buy" and s.get("action") in ("short", "sell"):
                cp = pos.get("current_price") or pos.get("entry_price", 0)
                close_position_and_log(alpaca_client, pos, cp, "market_close", status="closed")
                console.print(f"[red]Signal flip close: {ticker}[/red]")


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
    from bot.portfolio import (check_stops, check_targets, check_time_exits,
                               get_open_positions, close_position_and_log)
    from bot.discovery import scan_rising_movers

    SCAN_INTERVAL = 5           # minutes between scans
    MOVER_SCAN_EVERY = 3        # run rising-movers screen every N cycles (~15 min)
    MARKET_OPEN_ET  = (9,  30)  # 9:30 AM ET
    SCALP_CLOSE_ET  = (15, 45)  # 3:45 PM ET — close scalps before market close
    LOOP_END_ET     = (16, 0)   # 4:00 PM ET — stop looping

    ET = timezone(timedelta(hours=-4))  # EDT; Nov-Mar use -5

    def _et_now():
        return datetime.now(ET)

    def _et_hm():
        n = _et_now()
        return (n.hour, n.minute)

    console.rule("[bold cyan]CONTINUOUS TRADING SESSION[/bold cyan]")
    console.print(f"[dim]Scanning every {SCAN_INTERVAL} min from 9:30 AM to 4:00 PM ET[/dim]")

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

        # ── Rising movers screen (every MOVER_SCAN_EVERY cycles) ──────────
        if cycle % MOVER_SCAN_EVERY == 1:
            extra_tickers = scan_rising_movers(STATIC_TICKERS)
            if extra_tickers:
                console.print(f"[bold cyan]Rising movers: {extra_tickers}[/bold cyan]")

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

        # ── Scalp close at 3:45 PM ─────────────────────────────────────────
        if now_hm >= SCALP_CLOSE_ET:
            open_pos = get_open_positions(alpaca_client)
            for pos in open_pos:
                if pos.get("time_horizon") == "scalp":
                    cp = pos.get("current_price") or pos.get("entry_price", 0)
                    close_position_and_log(alpaca_client, pos, cp, "continuous", status="closed")
                    console.print(f"[yellow]EOD scalp close: {pos['ticker']} @ {cp}[/yellow]")

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

    # Generate HTML report
    try:
        from reports.daily_report import generate_report
        report_path = generate_report()
        console.print(f"[bold cyan]HTML report generated: {report_path}[/bold cyan]")
    except Exception as e:
        logger.warning(f"[eod] HTML report generation failed: {e}")


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
    import yfinance as yf
    import numpy as np
    from bot.indicators import compute_indicators_from_df
    from bot.scorer     import score_ticker
    from bot.strategies import classify_strategy
    from rich.table     import Table
    from rich.panel     import Panel
    from rich.progress  import Progress, SpinnerColumn, TextColumn

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
            h = yf.Ticker(ticker).history(period=fetch_period, interval="1d", auto_adjust=True)
            if h is not None and len(h) >= 60:
                all_hist[ticker] = h
        except Exception as e:
            logger.warning(f"[backtest] fetch {ticker}: {e}")

    try:
        spy_full = yf.Ticker("SPY").history(period=fetch_period, interval="1d", auto_adjust=True)
    except Exception:
        spy_full = None

    try:
        vix_full = yf.Ticker("^VIX").history(period=fetch_period, interval="1d", auto_adjust=True)
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
        choices=["discovery", "premarket", "market_open", "midday", "market_close", "eod_summary", "backtest", "continuous", "holdings"],
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
        elif session == "market_open":
            session_market_open(alpaca_client, data_client)
        elif session == "midday":
            session_midday(alpaca_client, data_client)
        elif session == "market_close":
            session_market_close(alpaca_client, data_client)
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
