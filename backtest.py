"""
Backtester — replays the full indicator + scorer pipeline on historical OHLCV data.

Simulates what the bot would have traded day-by-day, without Claude calls
(cost-prohibitive for multi-month replay). Uses the scorer net score threshold
as the sole entry gate (same threshold Claude would receive).

Execution model:
  - Signals generated using EOD close of day N
  - Entries executed at OPEN of day N+1 (realistic: limit orders set overnight)
  - Stops / targets / time exits checked against intraday High/Low of each day
  - All position sizing mirrors bot/risk.py calculate_position()

Usage:
    python backtest.py
    python backtest.py --start 2024-06-01 --end 2025-06-01
    python backtest.py --start 2024-01-01 --capital 50000 --min-net 65
    python backtest.py --tickers NVDA AMD TSLA AAPL   # specific tickers only
"""

import argparse
import csv
import json
import logging
import os
import sys
from datetime import date, timedelta, datetime
from math import floor
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.text import Text

load_dotenv()

# Silence noisy sub-loggers during the replay
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("backtest")

console = Console()

sys.path.insert(0, str(Path(__file__).parent))

from bot.data import fetch_daily_bars_batch
from bot.indicators import compute_indicators_from_df, compute_entry_triggers
from bot.scorer import score_ticker
from bot.strategies import classify_strategy

# ── Universe ─────────────────────────────────────────────────────────────────
from bot.discovery import UNIVERSE
from main import STATIC_TICKERS, SECTOR_GROUPS

ALL_TICKERS = list(dict.fromkeys(STATIC_TICKERS + UNIVERSE))  # deduped, order preserved

# ── Backtest defaults ─────────────────────────────────────────────────────────
DEFAULT_START      = "2024-06-01"
DEFAULT_END        = "2025-06-01"
STARTING_CAPITAL   = 100_000.0
MIN_NET_SCORE      = 60          # no Claude: use scorer threshold directly
MAX_OPEN_POSITIONS = 5
MAX_POSITION_PCT   = 0.10        # 10% of portfolio per trade
RISK_PCT           = 0.02        # 2% portfolio risk per trade
ATR_STOP_MULT      = 1.5         # stop = entry - atr * mult
MAX_PER_SECTOR     = 2
MAX_HOLD_DAYS      = {"scalp": 2, "swing": 7, "mixed": 5}
MIN_CONFIDENCE     = 0.65        # mirrors live bot gate

# ── Sector mapping ────────────────────────────────────────────────────────────
_SECTOR_OF: dict[str, str] = {}
for _sec, _tks in SECTOR_GROUPS.items():
    for _tk in _tks:
        _SECTOR_OF[_tk.upper()] = _sec


# ─────────────────────────────────────────────────────────────────────────────
#  Data layer
# ─────────────────────────────────────────────────────────────────────────────

def load_all_bars(tickers: list[str], start: date, end: date) -> dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for all tickers in one batch call.
    We fetch extra history before `start` so indicators have enough warm-up bars.
    Returns {ticker: df} where df covers full history up to `end`.
    """
    fetch_start = start - timedelta(days=400)   # 400-day warm-up for EMA200 + ATR
    console.print(f"[cyan]Fetching bars for {len(tickers)} tickers "
                  f"({fetch_start} → {end})…[/cyan]")
    days_needed = (end - fetch_start).days + 10
    bars = fetch_daily_bars_batch(tickers, days=days_needed)
    console.print(f"[green]Loaded {len(bars)}/{len(tickers)} tickers[/green]")
    return bars


def get_trading_days(bars_map: dict[str, pd.DataFrame], start: date, end: date) -> list[date]:
    """Extract actual NYSE trading days from the loaded data."""
    all_dates: set[date] = set()
    for df in bars_map.values():
        for ts in df.index:
            d = ts.date() if hasattr(ts, "date") else ts
            if start <= d <= end:
                all_dates.add(d)
    return sorted(all_dates)


# ─────────────────────────────────────────────────────────────────────────────
#  Indicator computation (historical, no API calls)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ind_for_day(ticker: str, df: pd.DataFrame, as_of: date) -> dict:
    """
    Slice df to [start, as_of] and compute indicators — no lookahead.
    realtime_price=False uses the last close as current price.
    """
    mask = pd.Series(df.index).apply(lambda x: (x.date() if hasattr(x, "date") else x) <= as_of)
    sliced = df.iloc[mask.values]
    if len(sliced) < 30:
        return {"ticker": ticker, "error": "insufficient_history"}
    return compute_indicators_from_df(ticker, sliced, intraday=None, realtime_price=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Position sizing (mirrors bot/risk.py)
# ─────────────────────────────────────────────────────────────────────────────

def size_position(portfolio_value: float, confidence: float,
                  atr: float, price: float) -> int:
    if price <= 0 or atr <= 0:
        return 0
    dollar_risk = portfolio_value * RISK_PCT * confidence
    shares      = floor(dollar_risk / (atr * ATR_STOP_MULT))
    max_shares  = floor(portfolio_value * MAX_POSITION_PCT / price)
    return max(0, min(shares, max_shares))


# ─────────────────────────────────────────────────────────────────────────────
#  Core simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_backtest(
    tickers: list[str],
    start:   date,
    end:     date,
    starting_capital: float = STARTING_CAPITAL,
    min_net: int = MIN_NET_SCORE,
) -> dict:
    """
    Full day-by-day simulation. Returns results dict with trades + equity curve.
    """
    bars_map     = load_all_bars(tickers, start, end)
    trading_days = get_trading_days(bars_map, start, end)

    if not trading_days:
        console.print("[red]No trading days found — check date range and API credentials[/red]")
        return {}

    console.print(f"[cyan]Simulating {len(trading_days)} trading days "
                  f"({trading_days[0]} → {trading_days[-1]})[/cyan]\n")

    # ── State ─────────────────────────────────────────────────────────────────
    cash      = starting_capital
    positions: dict[str, dict] = {}  # ticker -> position record
    all_trades: list[dict]     = []
    equity_curve: list[dict]   = []
    prev_ind_map: dict[str, dict] = {}  # for entry trigger comparison

    # Macro approximation: SPY above EMA50 = bull, else caution
    spy_bars = bars_map.get("SPY")

    def _spy_regime(as_of: date) -> str:
        if spy_bars is None:
            return "bull"
        try:
            mask = pd.Series(spy_bars.index).apply(
                lambda x: (x.date() if hasattr(x, "date") else x) <= as_of
            )
            sl = spy_bars.iloc[mask.values]
            if len(sl) < 50:
                return "bull"
            price = float(sl["Close"].iloc[-1])
            ema50 = float(sl["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
            return "bull" if price >= ema50 else "caution"
        except Exception:
            return "bull"

    # ── Day loop ──────────────────────────────────────────────────────────────
    for day_idx, today in enumerate(trading_days):

        # Portfolio value = cash + mark-to-market open positions
        mkt_value = cash
        for ticker, pos in positions.items():
            df = bars_map.get(ticker)
            if df is None:
                continue
            mask = pd.Series(df.index).apply(
                lambda x: (x.date() if hasattr(x, "date") else x) <= today
            )
            sl = df.iloc[mask.values]
            if not sl.empty:
                cp = float(sl["Close"].iloc[-1])
                mkt_value += pos["shares"] * cp

        equity_curve.append({"date": today, "equity": round(mkt_value, 2)})

        regime = _spy_regime(today)
        macro  = {
            "vix": 18.0,           # static approximation — VIX not available in backtest
            "spy_regime": regime,
            "bearish_market": regime != "bull",
            "vix_multiplier": 1.0,
        }

        # ── 1. Check exits on all open positions ──────────────────────────────
        closed_tickers = []
        for ticker, pos in list(positions.items()):
            df = bars_map.get(ticker)
            if df is None:
                continue
            mask = pd.Series(df.index).apply(
                lambda x: (x.date() if hasattr(x, "date") else x) == today
            )
            today_bars = df.iloc[mask.values]
            if today_bars.empty:
                continue

            day_high  = float(today_bars["High"].iloc[-1])
            day_low   = float(today_bars["Low"].iloc[-1])
            day_close = float(today_bars["Close"].iloc[-1])
            day_open  = float(today_bars["Open"].iloc[-1])

            stop   = pos["stop_loss"]
            target = pos["take_profit"]
            entry  = pos["entry_price"]
            shares = pos["shares"]
            horizon = pos.get("strategy", "swing")
            max_hold = MAX_HOLD_DAYS.get(horizon, 5)
            age_days = (today - pos["entry_date"]).days

            exit_price  = None
            exit_reason = None

            # Stop hit — worst case: gap down through stop uses day open
            if day_low <= stop:
                exit_price  = max(min(stop, day_open), day_low)  # realistic fill
                exit_reason = "stop"
            # Target hit
            elif target and day_high >= target:
                exit_price  = min(target, day_high)
                exit_reason = "target"
            # Time exit
            elif age_days >= max_hold:
                exit_price  = day_close
                exit_reason = "time_exit"

            if exit_price is not None:
                pnl_dollar = (exit_price - entry) * shares
                pnl_pct    = (exit_price - entry) / entry * 100
                cash      += exit_price * shares
                trade_rec  = {
                    "ticker":       ticker,
                    "entry_date":   pos["entry_date"].isoformat(),
                    "exit_date":    today.isoformat(),
                    "entry_price":  round(entry, 2),
                    "exit_price":   round(exit_price, 2),
                    "shares":       shares,
                    "stop_loss":    round(stop, 2),
                    "take_profit":  round(target, 2) if target else None,
                    "pnl_dollar":   round(pnl_dollar, 2),
                    "pnl_pct":      round(pnl_pct, 2),
                    "hold_days":    age_days,
                    "exit_reason":  exit_reason,
                    "strategy":     pos.get("strategy", "unknown"),
                    "net_score":    pos.get("net_score", 0),
                    "confidence":   pos.get("confidence", 0),
                    "signals":      pos.get("signals", []),
                }
                all_trades.append(trade_rec)
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del positions[t]

        # ── 2. Generate signals using today's close ───────────────────────────
        if regime == "caution" and day_idx % 5 != 0:
            # In caution regime, still scan but suppress buys (done in scorer filter below)
            pass

        if len(positions) >= MAX_OPEN_POSITIONS:
            continue   # already full

        sector_counts: dict[str, int] = {}
        for held in positions:
            sec = _SECTOR_OF.get(held.upper())
            if sec:
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

        new_signals: list[dict] = []

        for ticker in tickers:
            if ticker in positions:
                continue
            if ticker == "SPY":
                continue

            df = bars_map.get(ticker)
            if df is None:
                continue

            # Compute current indicators (no lookahead)
            ind = compute_ind_for_day(ticker, df, today)
            if ind.get("error"):
                continue

            # Compute entry triggers vs previous cycle
            prev_ind = prev_ind_map.get(ticker, {})
            triggers = compute_entry_triggers(ind, prev_ind)
            ind["entry_triggers"] = triggers
            prev_ind_map[ticker] = {k: v for k, v in ind.items()
                                    if k not in ("entry_triggers",)}

            try:
                score = score_ticker(ticker, ind, {}, macro)
                score = classify_strategy(score, ind)
            except Exception as e:
                logger.debug(f"[backtest] scorer failed {ticker} on {today}: {e}")
                continue

            action     = score.get("action", "hold")
            net        = score.get("net_score", 0)
            confidence = score.get("confidence", 0.0)

            # Apply same gates as live bot (no Claude)
            if action != "buy":
                continue
            if net < min_net:
                continue
            if confidence < MIN_CONFIDENCE:
                continue
            if macro["bearish_market"]:
                continue   # suppress buys in downtrend

            # Sector cap
            sec = _SECTOR_OF.get(ticker.upper())
            if sec and sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
                continue

            new_signals.append(score)

        # Rank by confidence, take top N to fill remaining slots
        new_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        slots = MAX_OPEN_POSITIONS - len(positions)

        for score in new_signals[:slots]:
            ticker     = score["ticker"]
            ind        = prev_ind_map.get(ticker, {})  # just saved current
            atr        = score.get("atr") or ind.get("atr") or 0
            entry_ref  = score.get("entry_price") or ind.get("current_price") or 0
            stop_loss  = score.get("stop_loss") or 0
            take_profit = score.get("take_profit") or 0
            confidence = score.get("confidence", 0.65)
            net        = score.get("net_score", 0)
            strategy   = score.get("strategy", "swing")

            if entry_ref <= 0 or atr <= 0:
                continue

            # Entry at NEXT DAY's open (realistic execution)
            if day_idx + 1 >= len(trading_days):
                continue
            next_day = trading_days[day_idx + 1]
            df = bars_map.get(ticker)
            if df is None:
                continue
            mask_next = pd.Series(df.index).apply(
                lambda x: (x.date() if hasattr(x, "date") else x) == next_day
            )
            next_bars = df.iloc[mask_next.values]
            if next_bars.empty:
                continue

            entry_price = float(next_bars["Open"].iloc[0])
            if entry_price <= 0:
                continue

            # Position sizing
            shares = size_position(
                portfolio_value=cash + sum(
                    p["shares"] * entry_price for p in positions.values()
                ),
                confidence=confidence,
                atr=atr,
                price=entry_price,
            )
            if shares < 1:
                continue

            cost = shares * entry_price
            if cost > cash:
                shares = floor(cash * MAX_POSITION_PCT / entry_price)
                cost   = shares * entry_price
            if shares < 1:
                continue

            # Recompute stop/target relative to actual entry price
            if stop_loss <= 0 or stop_loss >= entry_price:
                stop_loss = round(entry_price - atr * ATR_STOP_MULT, 2)
            if take_profit <= 0 or take_profit <= entry_price:
                rr = 2.5
                risk = entry_price - stop_loss
                take_profit = round(entry_price + risk * rr, 2)

            cash -= cost

            sec = _SECTOR_OF.get(ticker.upper())
            if sec:
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

            positions[ticker] = {
                "ticker":      ticker,
                "entry_date":  next_day,
                "entry_price": entry_price,
                "shares":      shares,
                "stop_loss":   stop_loss,
                "take_profit": take_profit,
                "strategy":    strategy,
                "net_score":   net,
                "confidence":  confidence,
                "signals":     score.get("signals_triggered", []),
            }

    # ── Close any remaining open positions at end date ────────────────────────
    final_day = trading_days[-1] if trading_days else end
    for ticker, pos in list(positions.items()):
        df = bars_map.get(ticker)
        exit_price = pos["entry_price"]  # fallback
        if df is not None:
            mask = pd.Series(df.index).apply(
                lambda x: (x.date() if hasattr(x, "date") else x) <= final_day
            )
            sl = df.iloc[mask.values]
            if not sl.empty:
                exit_price = float(sl["Close"].iloc[-1])
        pnl_dollar = (exit_price - pos["entry_price"]) * pos["shares"]
        pnl_pct    = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        cash += exit_price * pos["shares"]
        all_trades.append({
            "ticker":       ticker,
            "entry_date":   pos["entry_date"].isoformat(),
            "exit_date":    final_day.isoformat(),
            "entry_price":  round(pos["entry_price"], 2),
            "exit_price":   round(exit_price, 2),
            "shares":       pos["shares"],
            "stop_loss":    round(pos["stop_loss"], 2),
            "take_profit":  round(pos["take_profit"], 2),
            "pnl_dollar":   round(pnl_dollar, 2),
            "pnl_pct":      round(pnl_pct, 2),
            "hold_days":    (final_day - pos["entry_date"]).days,
            "exit_reason":  "end_of_backtest",
            "strategy":     pos.get("strategy", "unknown"),
            "net_score":    pos.get("net_score", 0),
            "confidence":   pos.get("confidence", 0),
            "signals":      pos.get("signals", []),
        })

    final_equity = cash
    return {
        "trades":       all_trades,
        "equity_curve": equity_curve,
        "final_equity": final_equity,
        "start_equity": starting_capital,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────────────────

def compute_stats(results: dict) -> dict:
    trades = results.get("trades", [])
    eq     = results.get("equity_curve", [])
    start  = results["start_equity"]
    end_eq = results["final_equity"]

    if not trades:
        return {"total_trades": 0}

    pnls    = [t["pnl_dollar"] for t in trades]
    wins    = [p for p in pnls if p > 0]
    losses  = [p for p in pnls if p <= 0]
    win_pcts = [t["pnl_pct"] for t in trades if t["pnl_dollar"] > 0]
    loss_pcts = [t["pnl_pct"] for t in trades if t["pnl_dollar"] <= 0]

    total_return_pct = (end_eq - start) / start * 100

    # Max drawdown from equity curve
    max_dd = 0.0
    if eq:
        peak = eq[0]["equity"]
        for e in eq:
            if e["equity"] > peak:
                peak = e["equity"]
            dd = (peak - e["equity"]) / peak * 100
            if dd > max_dd:
                max_dd = dd

    # Sharpe (annualised, daily returns, rf=0)
    sharpe = 0.0
    if len(eq) > 1:
        daily_returns = []
        for i in range(1, len(eq)):
            prev_eq = eq[i-1]["equity"]
            if prev_eq > 0:
                daily_returns.append((eq[i]["equity"] - prev_eq) / prev_eq)
        if daily_returns:
            dr = np.array(daily_returns)
            std = dr.std()
            sharpe = (dr.mean() / std * np.sqrt(252)) if std > 0 else 0.0

    avg_hold = np.mean([t["hold_days"] for t in trades]) if trades else 0

    # Strategy breakdown
    by_strategy: dict[str, list] = {}
    for t in trades:
        s = t.get("strategy", "unknown")
        by_strategy.setdefault(s, []).append(t["pnl_dollar"])

    # Monthly P&L
    monthly: dict[str, float] = {}
    for t in trades:
        ym = t["exit_date"][:7]
        monthly[ym] = monthly.get(ym, 0) + t["pnl_dollar"]

    return {
        "total_trades":   len(trades),
        "winners":        len(wins),
        "losers":         len(losses),
        "win_rate":       len(wins) / len(trades) * 100,
        "total_pnl":      sum(pnls),
        "total_return_pct": total_return_pct,
        "avg_win_pct":    np.mean(win_pcts) if win_pcts else 0,
        "avg_loss_pct":   np.mean(loss_pcts) if loss_pcts else 0,
        "best_trade_pct": max(t["pnl_pct"] for t in trades),
        "worst_trade_pct": min(t["pnl_pct"] for t in trades),
        "profit_factor":  abs(sum(wins) / sum(losses)) if losses else float("inf"),
        "max_drawdown_pct": max_dd,
        "sharpe":         round(sharpe, 2),
        "avg_hold_days":  round(avg_hold, 1),
        "by_strategy":    {s: {"trades": len(v), "total_pnl": round(sum(v), 2),
                               "win_rate": round(len([x for x in v if x > 0]) / len(v) * 100, 1)}
                           for s, v in by_strategy.items()},
        "monthly_pnl":    {k: round(v, 2) for k, v in sorted(monthly.items())},
    }


def print_report(results: dict, stats: dict, args) -> None:
    trades = results.get("trades", [])
    start  = results["start_equity"]
    end_eq = results["final_equity"]
    ret    = stats.get("total_return_pct", 0)

    ret_color = "green" if ret >= 0 else "red"

    console.print()
    console.print(Panel(
        f"[bold]MoneyPrinter Backtest Report[/bold]\n"
        f"{args.start}  →  {args.end}\n"
        f"Universe: {len(args.tickers or ALL_TICKERS)} tickers  |  "
        f"Min net score: {args.min_net}",
        style="bold cyan",
        expand=False,
    ))

    # ── Summary ───────────────────────────────────────────────────────────────
    t = Table(title="Summary", box=box.SIMPLE_HEAVY, show_header=False)
    t.add_column("Metric", style="bold")
    t.add_column("Value")

    def _pct(v): return f"{v:+.2f}%"
    def _usd(v): return f"${v:,.2f}"

    t.add_row("Starting capital",  _usd(start))
    t.add_row("Ending equity",     _usd(end_eq))
    t.add_row("Total return",      f"[{ret_color}]{_pct(ret)}[/{ret_color}]")
    t.add_row("Total trades",      str(stats.get("total_trades", 0)))
    t.add_row("Win rate",          f"{stats.get('win_rate', 0):.1f}%  "
                                   f"({stats.get('winners',0)}W / {stats.get('losers',0)}L)")
    t.add_row("Avg win",           f"[green]{_pct(stats.get('avg_win_pct', 0))}[/green]")
    t.add_row("Avg loss",          f"[red]{_pct(stats.get('avg_loss_pct', 0))}[/red]")
    t.add_row("Profit factor",     f"{stats.get('profit_factor', 0):.2f}")
    t.add_row("Max drawdown",      f"[red]{stats.get('max_drawdown_pct', 0):.2f}%[/red]")
    t.add_row("Sharpe ratio",      str(stats.get("sharpe", 0)))
    t.add_row("Avg hold (days)",   str(stats.get("avg_hold_days", 0)))
    t.add_row("Best trade",        f"[green]{_pct(stats.get('best_trade_pct', 0))}[/green]")
    t.add_row("Worst trade",       f"[red]{_pct(stats.get('worst_trade_pct', 0))}[/red]")
    console.print(t)

    # ── Strategy breakdown ────────────────────────────────────────────────────
    by_s = stats.get("by_strategy", {})
    if by_s:
        st = Table(title="By Strategy", box=box.SIMPLE)
        st.add_column("Strategy")
        st.add_column("Trades", justify="right")
        st.add_column("Win %",  justify="right")
        st.add_column("Total P&L", justify="right")
        for strategy, data in sorted(by_s.items(), key=lambda x: -x[1]["total_pnl"]):
            color = "green" if data["total_pnl"] >= 0 else "red"
            st.add_row(
                strategy,
                str(data["trades"]),
                f"{data['win_rate']}%",
                f"[{color}]{_usd(data['total_pnl'])}[/{color}]",
            )
        console.print(st)

    # ── Monthly P&L ───────────────────────────────────────────────────────────
    monthly = stats.get("monthly_pnl", {})
    if monthly:
        mt = Table(title="Monthly P&L", box=box.SIMPLE)
        mt.add_column("Month")
        mt.add_column("P&L", justify="right")
        for ym, pnl in monthly.items():
            color = "green" if pnl >= 0 else "red"
            mt.add_row(ym, f"[{color}]{_usd(pnl)}[/{color}]")
        console.print(mt)

    # ── Best 10 trades ────────────────────────────────────────────────────────
    if trades:
        best = sorted(trades, key=lambda x: x["pnl_pct"], reverse=True)[:10]
        bt = Table(title="Best 10 Trades", box=box.SIMPLE)
        for col in ["Ticker", "Entry Date", "Exit Date", "Entry $", "Exit $",
                    "Shares", "P&L $", "P&L %", "Hold", "Reason", "Strategy"]:
            bt.add_column(col)
        for tr in best:
            bt.add_row(
                tr["ticker"],
                tr["entry_date"],
                tr["exit_date"],
                f"${tr['entry_price']:.2f}",
                f"${tr['exit_price']:.2f}",
                str(tr["shares"]),
                f"[green]${tr['pnl_dollar']:,.2f}[/green]",
                f"[green]{tr['pnl_pct']:+.1f}%[/green]",
                f"{tr['hold_days']}d",
                tr["exit_reason"],
                tr.get("strategy", ""),
            )
        console.print(bt)

    # ── Worst 10 trades ───────────────────────────────────────────────────────
    if trades:
        worst = sorted(trades, key=lambda x: x["pnl_pct"])[:10]
        wt = Table(title="Worst 10 Trades", box=box.SIMPLE)
        for col in ["Ticker", "Entry Date", "Exit Date", "Entry $", "Exit $",
                    "Shares", "P&L $", "P&L %", "Hold", "Reason", "Strategy"]:
            wt.add_column(col)
        for tr in worst:
            wt.add_row(
                tr["ticker"],
                tr["entry_date"],
                tr["exit_date"],
                f"${tr['entry_price']:.2f}",
                f"${tr['exit_price']:.2f}",
                str(tr["shares"]),
                f"[red]${tr['pnl_dollar']:,.2f}[/red]",
                f"[red]{tr['pnl_pct']:+.1f}%[/red]",
                f"{tr['hold_days']}d",
                tr["exit_reason"],
                tr.get("strategy", ""),
            )
        console.print(wt)

    # ── All trades table ──────────────────────────────────────────────────────
    if trades:
        at = Table(title=f"All {len(trades)} Trades", box=box.MINIMAL)
        for col in ["Ticker", "Entry", "Exit", "Entry $", "Exit $",
                    "Shs", "P&L $", "P&L %", "Hold", "Why Out", "Strategy", "Net", "Conf"]:
            at.add_column(col)
        for tr in sorted(trades, key=lambda x: x["entry_date"]):
            color = "green" if tr["pnl_dollar"] >= 0 else "red"
            at.add_row(
                tr["ticker"],
                tr["entry_date"],
                tr["exit_date"],
                f"${tr['entry_price']:.2f}",
                f"${tr['exit_price']:.2f}",
                str(tr["shares"]),
                f"[{color}]${tr['pnl_dollar']:,.2f}[/{color}]",
                f"[{color}]{tr['pnl_pct']:+.1f}%[/{color}]",
                f"{tr['hold_days']}d",
                tr["exit_reason"],
                tr.get("strategy", ""),
                str(tr.get("net_score", "")),
                f"{tr.get('confidence', 0):.2f}",
            )
        console.print(at)

    console.print()
    console.print("[bold yellow]Note:[/bold yellow] Backtest does not include Claude AI calls. "
                  "Live bot's Claude filter will further reduce trade count and may improve quality.")
    console.print()


def save_csv(trades: list[dict], path: Path) -> None:
    if not trades:
        return
    fieldnames = [
        "ticker", "entry_date", "exit_date", "entry_price", "exit_price",
        "shares", "stop_loss", "take_profit", "pnl_dollar", "pnl_pct",
        "hold_days", "exit_reason", "strategy", "net_score", "confidence",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trades)
    console.print(f"[green]Trades exported → {path}[/green]")


def save_equity_csv(equity_curve: list[dict], path: Path) -> None:
    if not equity_curve:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["date", "equity"])
        writer.writeheader()
        writer.writerows(equity_curve)
    console.print(f"[green]Equity curve exported → {path}[/green]")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MoneyPrinter Backtester")
    parser.add_argument("--start",   default=DEFAULT_START, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",     default=DEFAULT_END,   help="End date YYYY-MM-DD")
    parser.add_argument("--capital", type=float, default=STARTING_CAPITAL,
                        help="Starting capital (default 100000)")
    parser.add_argument("--min-net", type=int, default=MIN_NET_SCORE,
                        help="Minimum scorer net score for entry (default 60)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Specific tickers to test (default: full universe)")
    parser.add_argument("--out",     default="backtest_results",
                        help="Output file prefix for CSV exports")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    if end <= start:
        console.print("[red]--end must be after --start[/red]")
        sys.exit(1)

    tickers = args.tickers or ALL_TICKERS
    # Always include SPY for regime detection
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers

    console.print(f"\n[bold cyan]MoneyPrinter Backtester[/bold cyan]")
    console.print(f"Period:  {start} → {end}  ({(end-start).days} calendar days)")
    console.print(f"Capital: ${args.capital:,.0f}")
    console.print(f"Tickers: {len(tickers)} (including SPY for regime)")
    console.print(f"Min net: {args.min_net}\n")

    results = run_backtest(
        tickers=tickers,
        start=start,
        end=end,
        starting_capital=args.capital,
        min_net=args.min_net,
    )

    if not results:
        sys.exit(1)

    stats = compute_stats(results)
    print_report(results, stats, args)

    # ── Export ────────────────────────────────────────────────────────────────
    out_dir = Path("backtest_output")
    out_dir.mkdir(exist_ok=True)
    prefix  = f"{out_dir}/{args.out}_{args.start}_{args.end}"
    save_csv(results["trades"], Path(f"{prefix}_trades.csv"))
    save_equity_csv(results["equity_curve"], Path(f"{prefix}_equity.csv"))

    # JSON summary
    summary_path = Path(f"{prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "args": vars(args),
            "stats": {k: v for k, v in stats.items()
                      if not isinstance(v, (dict,))},
            "by_strategy": stats.get("by_strategy", {}),
            "monthly_pnl": stats.get("monthly_pnl", {}),
        }, f, indent=2)
    console.print(f"[green]Summary exported → {summary_path}[/green]")


if __name__ == "__main__":
    main()
