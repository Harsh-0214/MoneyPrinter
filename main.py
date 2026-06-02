"""
Main entry point for the autonomous trading bot.
Routes to session-specific logic based on --session argument.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
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
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
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

ALL_TRADE_TICKERS = (
    WATCHLIST["trade"]["tech"]
    + WATCHLIST["trade"]["momentum"]
    + WATCHLIST["trade"]["financials"]
    + WATCHLIST["trade"]["energy"]
)
MACRO_TICKERS   = WATCHLIST["macro_context_only"]
COMPANY_NAMES   = WATCHLIST["company_names"]
DRY_RUN         = os.getenv("DRY_RUN", "true").lower() == "true"


def is_market_open_today() -> bool:
    """Check NYSE calendar to determine if today is a trading day."""
    nyse = mcal.get_calendar("NYSE")
    today = datetime.utcnow().strftime("%Y-%m-%d")
    schedule = nyse.schedule(start_date=today, end_date=today)
    return not schedule.empty


def get_macro_context() -> dict:
    """
    Compute macro context: VIX level, SPY regime, position size multiplier.
    """
    import yfinance as yf
    from bot.indicators import get_indicators
    from bot.risk import get_vix_multiplier

    macro = {
        "vix": 20.0,
        "spy_regime": "bull",
        "vix_multiplier": 1.0,
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
        if price and ema50 and ema200:
            if price > ema50 and price > ema200:
                macro["spy_regime"] = "bull"
            elif price < ema50 and price > ema200:
                macro["spy_regime"] = "caution"
            else:
                macro["spy_regime"] = "bear"
    except Exception as e:
        logger.warning(f"SPY regime check failed: {e}")

    macro["vix_multiplier"] = get_vix_multiplier(macro["vix"])
    logger.info(
        f"[macro] VIX={macro['vix']:.1f} regime={macro['spy_regime']} "
        f"size_mult={macro['vix_multiplier']:.2f}"
    )
    return macro


def run_full_scan(session: str, macro_context: dict, alpaca_client=None, data_client=None) -> list[dict]:
    """
    Score all tickers. Returns list of actionable signal dicts.
    """
    from bot.indicators  import get_indicators_batch
    from bot.news        import get_news_batch
    from bot.scorer      import score_ticker
    from bot.strategies  import classify_strategy

    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

    console.print(f"[bold cyan]Scanning {len(ALL_TRADE_TICKERS)} tickers...[/bold cyan]")
    indicators_map = get_indicators_batch(ALL_TRADE_TICKERS, max_workers=5)
    news_map = get_news_batch(ALL_TRADE_TICKERS, COMPANY_NAMES, api_key=NEWS_API_KEY, max_workers=3)

    signals = []
    bull_count = 0
    bear_count = 0

    for ticker in ALL_TRADE_TICKERS:
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

        action = score.get("action", "hold")
        net    = score.get("net_score", 0)

        logger.info(
            f"[{ticker}] action={action} net={net} bull={score.get('bull_score')} "
            f"bear={score.get('bear_score')} conf={score.get('confidence'):.2f} "
            f"strategy={score.get('strategy')}"
        )

        if action != "hold":
            signals.append(score)
            if action == "buy":
                bull_count += 1
            else:
                bear_count += 1

    from bot.logger import log_scan
    log_scan(
        session=session,
        tickers_scanned=len(ALL_TRADE_TICKERS),
        signals_generated=len(signals),
        trades_executed=0,
        total_bull=bull_count,
        total_bear=bear_count,
    )
    return signals


def execute_signals(signals: list, alpaca_client, data_client, macro_context: dict, session: str) -> int:
    """Submit orders for actionable signals. Returns number of trades executed."""
    from bot.logger import log_scan, log_trade
    from bot.risk   import calculate_position, is_kill_switch_active, init_daily_state
    from bot.trader import (
        get_account, submit_order, compute_limit_price,
        get_latest_quote, check_order_filled,
    )

    if is_kill_switch_active():
        logger.warning("[execute] Kill switch active — no orders will be placed")
        return 0

    account = get_account(alpaca_client)
    portfolio_value = account.get("portfolio_value", 100_000)
    init_daily_state(portfolio_value)

    executed = 0
    for sig in signals:
        if is_kill_switch_active():
            break

        ticker     = sig["ticker"]
        action     = sig["action"]
        confidence = sig["confidence"]
        atr        = sig.get("atr") or (sig.get("entry_price", 100) * 0.02)
        entry_price = sig.get("entry_price") or 0
        strategy   = sig.get("strategy", "mixed")
        high_vol   = sig.get("high_vol_flag", False)

        if entry_price == 0:
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
            logger.info(f"[execute] {ticker}: 0 shares — skipping ({pos['reason']})")
            continue

        # Get quote for limit price
        quote = get_latest_quote(data_client, ticker)
        alpaca_side   = "buy" if action == "buy" else "sell"
        limit_price   = compute_limit_price(alpaca_side, quote, entry_price)

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
                if fill.get("filled_avg_price"):
                    entry_price = fill["filled_avg_price"]

            log_trade(
                session=session,
                ticker=ticker,
                action=action,
                strategy=strategy,
                time_horizon=sig.get("time_horizon", "swing"),
                quantity=shares,
                entry_price=entry_price,
                limit_price=limit_price,
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
            )
            executed += 1
            console.print(
                f"[green]✓ {action.upper()} {shares}x {ticker} @ ${limit_price:.2f} "
                f"| strat={strategy} conf={confidence:.2f}[/green]"
            )
        except Exception as e:
            logger.error(f"[execute] Order failed for {ticker}: {e}")

    return executed


# ══════════════════════════════════════════════════════════════════════════════
# SESSION HANDLERS
# ══════════════════════════════════════════════════════════════════════════════

def session_premarket() -> None:
    """9:00 AM EDT — fetch overnight news, flag gap moves. No trades."""
    console.rule("[bold yellow]PRE-MARKET SESSION[/bold yellow]")

    macro = get_macro_context()

    from bot.indicators import get_indicators_batch
    from bot.news       import get_news_batch

    NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")
    indicators_map = get_indicators_batch(ALL_TRADE_TICKERS, max_workers=5)
    news_map = get_news_batch(ALL_TRADE_TICKERS, COMPANY_NAMES, api_key=NEWS_API_KEY, max_workers=3)

    gap_ups   = []
    gap_downs = []

    for ticker in ALL_TRADE_TICKERS:
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
            logger.info(f"[premarket] News signal: {ticker} polarity={pol:.2f} headlines={news.get('headline_count',0)}")

    console.print(f"[cyan]Gap Ups (>2%):   {gap_ups}[/cyan]")
    console.print(f"[red]Gap Downs (<-2%): {gap_downs}[/red]")
    console.print(f"[bold]VIX={macro['vix']:.1f}  Regime={macro['spy_regime']}[/bold]")

    from bot.logger import log_scan
    log_scan("premarket", len(ALL_TRADE_TICKERS), 0, 0, len(gap_ups), len(gap_downs))


def session_market_open(alpaca_client, data_client) -> None:
    """9:35 AM EDT — full score run, execute all signals above threshold."""
    console.rule("[bold green]MARKET OPEN SESSION[/bold green]")
    macro = get_macro_context()

    # VIX extreme: no longs
    if macro["vix"] > 35:
        console.print("[bold red]VIX > 35 — EXTREME FEAR. No new long positions.[/bold red]")
        return

    signals = run_full_scan("market_open", macro, alpaca_client, data_client)
    executed = execute_signals(signals, alpaca_client, data_client, macro, "market_open")
    console.print(f"[bold green]Market open complete: {executed} trades executed[/bold green]")


def session_midday(alpaca_client, data_client) -> None:
    """12:00 PM EDT — check stops/targets, no new entries unless score > 80."""
    console.rule("[bold blue]MIDDAY SESSION[/bold blue]")

    from bot.portfolio import check_stops, check_targets, close_position_and_log, get_open_positions
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
        cp = pos.get("current_price") or pos.get("take_profit", 0)
        close_position_and_log(alpaca_client, pos, cp, "midday", status="closed")
        console.print(f"[green]TAKE PROFIT: {pos['ticker']} closed @ {cp}[/green]")

    # New entries only for very high confidence
    signals = run_full_scan("midday", macro, alpaca_client, data_client)
    high_conf = [s for s in signals if s.get("net_score", 0) > 80]
    if high_conf:
        executed = execute_signals(high_conf, alpaca_client, data_client, macro, "midday")
        console.print(f"[bold]High-confidence midday entries: {executed}[/bold]")
    else:
        console.print("[dim]No high-confidence entries (net > 80) found at midday[/dim]")


def session_market_close(alpaca_client, data_client) -> None:
    """3:30 PM EDT — close scalps, re-score, decide what to hold overnight."""
    console.rule("[bold magenta]MARKET CLOSE SESSION[/bold magenta]")

    from bot.portfolio import get_open_positions, close_position_and_log
    macro = get_macro_context()

    # Close all scalp positions
    open_positions = get_open_positions(alpaca_client)
    for pos in open_positions:
        if pos.get("time_horizon") == "scalp":
            cp = pos.get("current_price") or pos.get("entry_price", 0)
            close_position_and_log(alpaca_client, pos, cp, "market_close", status="closed")
            console.print(f"[yellow]EOD scalp close: {pos['ticker']} @ {cp}[/yellow]")

    # Re-score remaining tickers
    signals = run_full_scan("market_close", macro, alpaca_client, data_client)

    # Only execute if score is strong enough to hold overnight
    overnight_candidates = [s for s in signals if abs(s.get("net_score", 0)) >= 40 and s.get("confidence", 0) >= 0.60]
    if overnight_candidates:
        executed = execute_signals(overnight_candidates, alpaca_client, data_client, macro, "market_close")
        console.print(f"[bold]Overnight holds initiated: {executed}[/bold]")

    # Close open positions where score has turned against them
    open_positions = get_open_positions(alpaca_client)
    scored_map = {s["ticker"]: s for s in signals}
    for pos in open_positions:
        ticker = pos["ticker"]
        if ticker in scored_map:
            s = scored_map[ticker]
            pos_action = pos.get("action", "buy")
            sig_action = s.get("action", "hold")
            # Flip signal: was long, now bearish
            if pos_action == "buy" and sig_action in ("short", "sell"):
                cp = pos.get("current_price") or pos.get("entry_price", 0)
                close_position_and_log(alpaca_client, pos, cp, "market_close", status="closed")
                console.print(f"[red]Signal flip close: {ticker}[/red]")


def session_eod_summary(alpaca_client) -> None:
    """4:15 PM EDT — compute daily P&L, write summary, print Rich report."""
    console.rule("[bold white]END OF DAY SUMMARY[/bold white]")

    from bot.logger import (
        get_trades_today, get_daily_summaries, log_daily_summary,
    )
    from bot.portfolio import get_daily_pnl
    from bot.risk import is_kill_switch_active
    from bot.trader import get_account

    account = get_account(alpaca_client)
    portfolio_value = account.get("portfolio_value", 0)
    cash = account.get("cash", 0)

    trades_today = get_trades_today()
    closed = [t for t in trades_today if t.get("pnl_dollar") is not None]
    winners = [t for t in closed if (t.get("pnl_dollar") or 0) > 0]
    losers  = [t for t in closed if (t.get("pnl_dollar") or 0) <= 0]
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
        best_trade=f"{best['ticker']} ${best['pnl_dollar']:.2f}" if best else "N/A",
        worst_trade=f"{worst['ticker']} ${worst['pnl_dollar']:.2f}" if worst else "N/A",
        macro_bias=macro.get("spy_regime", "unknown"),
        vix_level=macro.get("vix", 0),
        kill_switch_triggered=is_kill_switch_active(),
    )

    # Rich report
    from rich.table import Table
    from rich.panel import Panel

    pnl_color = "green" if gross_pnl >= 0 else "red"
    console.print(Panel(
        f"[bold]Date:[/bold] {today}\n"
        f"[bold]Portfolio Value:[/bold] ${portfolio_value:,.2f}\n"
        f"[bold]Cash:[/bold] ${cash:,.2f}\n"
        f"[bold]Daily P&L:[/bold] [{pnl_color}]${gross_pnl:,.2f}[/{pnl_color}]\n"
        f"[bold]Win Rate:[/bold] {win_rate*100:.1f}% ({len(winners)}W / {len(losers)}L)\n"
        f"[bold]Total Closed Trades:[/bold] {len(closed)}\n"
        f"[bold]Best:[/bold] {best['ticker'] if best else 'N/A'} ${(best.get('pnl_dollar') or 0):.2f}\n"
        f"[bold]Worst:[/bold] {worst['ticker'] if worst else 'N/A'} ${(worst.get('pnl_dollar') or 0):.2f}\n"
        f"[bold]VIX:[/bold] {macro['vix']:.1f}  [bold]Regime:[/bold] {macro['spy_regime']}",
        title="[bold]Daily Summary[/bold]",
        border_style=pnl_color,
    ))

    # 7-day P&L bar
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


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(description="Autonomous Trading Bot")
    parser.add_argument(
        "--session",
        required=True,
        choices=["premarket", "market_open", "midday", "market_close", "eod_summary"],
        help="Which session to run",
    )
    args = parser.parse_args()
    session = args.session

    # Market holiday check
    if not is_market_open_today():
        console.print(f"[bold yellow]Market closed today — skipping {session}.[/bold yellow]")
        sys.exit(0)

    if DRY_RUN:
        console.print("[bold yellow]DRY_RUN=true — orders will be simulated[/bold yellow]")

    logger.info(f"Starting session: {session}")

    # Initialize DB
    from bot.logger import init_db
    init_db()

    # Build Alpaca clients
    try:
        from bot.trader import build_client, build_data_client
        alpaca_client = build_client()
        data_client   = build_data_client()
    except Exception as e:
        logger.error(f"Alpaca client init failed: {e}")
        if session not in ("premarket", "eod_summary"):
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
    except Exception as e:
        logger.critical(f"Session {session} crashed: {e}", exc_info=True)
        sys.exit(1)

    logger.info(f"Session {session} complete.")


if __name__ == "__main__":
    main()
