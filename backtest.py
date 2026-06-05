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
from concurrent.futures import ThreadPoolExecutor, as_completed
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
MAX_HOLD_DAYS      = {
    "scalp":            2,
    "swing":            7,
    "mixed":            5,
    "breakout":         4,   # exit fast if breakout doesn't follow through
    "squeeze_breakout": 4,   # exit fast if expansion stalls
}
MIN_CONFIDENCE     = 0.65        # mirrors live bot gate

# ── Improvement flags (all on by default) ─────────────────────────────────────
# Change 1: mixed has no coherent edge; squeeze_breakout/breakout re-enabled
# with tightened classifier conditions in bot/strategies.py
BAD_STRATEGIES     = {"mixed"}
# Change 2: high-beta names that keep hitting -10% stops get half the risk budget
HIGH_VOL_TICKERS   = {"SOFI", "TSLA", "MSTR", "ARM"}
HIGH_VOL_RISK_PCT  = 0.01        # 1% instead of 2%
# Change 3: after this many days with no new entry, drop threshold to catch recovery
REENTRY_SILENCE_DAYS   = 7
REENTRY_NET_REDUCTION  = 5      # 60 → 55
# Change 4: require confirmed trend before entering trend_follow (ADX > threshold)
ADX_TREND_MIN      = 20         # below this = sideways chop, not a real trend

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


def _precompute_ticker(args: tuple) -> tuple[str, dict]:
    """
    Worker: compute indicators + entry triggers for one ticker across all trading days.
    Runs each day sequentially (triggers need prev-day), but all tickers are independent
    so the outer pool parallelises across tickers.
    Returns (ticker, {date: ind_with_triggers}).
    """
    ticker, df, trading_days = args
    if df is None:
        return ticker, {}

    # Build a sorted date→positional-index map once for fast slicing
    idx_dates = [
        (ts.date() if hasattr(ts, "date") else ts)
        for ts in df.index
    ]
    # Pre-sort so we can binary-search for the <= as_of cut point
    import bisect
    cache: dict[date, dict] = {}
    prev_ind: dict = {}

    for d in trading_days:
        # Find the slice boundary with bisect (O(log n)) instead of applying a lambda
        cut = bisect.bisect_right(idx_dates, d)
        if cut < 30:
            cache[d] = {"ticker": ticker, "error": "insufficient_history"}
            continue
        sliced = df.iloc[:cut]
        try:
            ind = compute_indicators_from_df(ticker, sliced, intraday=None, realtime_price=False)
        except Exception as e:
            ind = {"ticker": ticker, "error": str(e)}

        if not ind.get("error"):
            triggers = compute_entry_triggers(ind, prev_ind)
            ind["entry_triggers"] = triggers
            prev_ind = {k: v for k, v in ind.items() if k != "entry_triggers"}

        cache[d] = ind

    return ticker, cache


def precompute_all_indicators(
    tickers: list[str],
    bars_map: dict[str, pd.DataFrame],
    trading_days: list[date],
    workers: int = 12,
) -> dict[str, dict[date, dict]]:
    """
    Parallel pre-computation of indicators for all tickers across all trading days.
    Returns {ticker: {date: ind_dict}} — same format as compute_ind_for_day output.
    With 12 workers this is ~10-12x faster than sequential per-day computation.
    """
    console.print(
        f"[cyan]Pre-computing indicators for {len(tickers)} tickers × "
        f"{len(trading_days)} days ({len(tickers)*len(trading_days):,} total calls)…[/cyan]"
    )
    args_list = [
        (t, bars_map.get(t), trading_days)
        for t in tickers
        if t != "SPY"
    ]
    result: dict[str, dict[date, dict]] = {}
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_precompute_ticker, a): a[0] for a in args_list}
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                t, cache = future.result()
                result[t] = cache
            except Exception as e:
                logger.warning(f"[backtest] precompute failed for {ticker}: {e}")
                result[ticker] = {}
            done += 1
            if done % 20 == 0 or done == len(args_list):
                console.print(
                    f"[dim]  {done}/{len(args_list)} tickers done[/dim]"
                )
    console.print(f"[green]Pre-computation complete.[/green]\n")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  Position sizing (mirrors bot/risk.py)
# ─────────────────────────────────────────────────────────────────────────────

def size_position(portfolio_value: float, confidence: float,
                  atr: float, price: float, risk_pct: float = RISK_PCT) -> int:
    if price <= 0 or atr <= 0:
        return 0
    dollar_risk = portfolio_value * risk_pct * confidence
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
    filter_bad_strategies: bool = True,
    apply_vol_cap: bool = True,
    reentry_relax: bool = True,
    adx_filter: bool = True,
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
                  f"({trading_days[0]} → {trading_days[-1]})[/cyan]")

    # ── Pre-compute all indicators (parallel, replaces per-day per-ticker calls) ─
    ind_cache = precompute_all_indicators(tickers, bars_map, trading_days)

    # ── State ─────────────────────────────────────────────────────────────────
    cash      = starting_capital
    positions: dict[str, dict] = {}  # ticker -> position record
    all_trades: list[dict]     = []
    equity_curve: list[dict]   = []
    last_entry_day: date | None  = None  # tracks last day an entry was made

    # Macro approximation: SPY above EMA50 = bull, else caution
    spy_bars = bars_map.get("SPY")

    # Pre-compute SPY regime for every trading day (fast: done once, not per-day)
    import bisect as _bisect
    _spy_dates = (
        [(ts.date() if hasattr(ts, "date") else ts) for ts in spy_bars.index]
        if spy_bars is not None else []
    )
    _spy_ema50 = (
        spy_bars["Close"].ewm(span=50, adjust=False).mean()
        if spy_bars is not None else None
    )
    _spy_close = spy_bars["Close"] if spy_bars is not None else None

    def _spy_regime(as_of: date) -> str:
        if not _spy_dates:
            return "bull"
        try:
            cut = _bisect.bisect_right(_spy_dates, as_of)
            if cut < 50:
                return "bull"
            price = float(_spy_close.iloc[cut - 1])
            ema50 = float(_spy_ema50.iloc[cut - 1])
            return "bull" if price >= ema50 else "caution"
        except Exception:
            return "bull"

    # Pre-build date-keyed OHLCV lookup for all tickers — replaces all per-day
    # lambda-mask operations (O(n) each) with O(1) dict lookups.
    console.print("[cyan]Building OHLCV date index…[/cyan]")
    ohlcv_by_date: dict[str, dict[date, dict]] = {}
    for t, df in bars_map.items():
        day_map: dict[date, dict] = {}
        for ts, row in df.iterrows():
            d = ts.date() if hasattr(ts, "date") else ts
            day_map[d] = {
                "O": float(row["Open"]),
                "H": float(row["High"]),
                "L": float(row["Low"]),
                "C": float(row["Close"]),
            }
        ohlcv_by_date[t] = day_map

    # For mark-to-market, also build cumulative close (last close on or before date)
    # by storing sorted dates per ticker for bisect lookups
    sorted_dates_by_ticker: dict[str, list[date]] = {
        t: sorted(dm.keys()) for t, dm in ohlcv_by_date.items()
    }

    def _last_close(ticker: str, as_of: date) -> float | None:
        dm = ohlcv_by_date.get(ticker)
        sd = sorted_dates_by_ticker.get(ticker)
        if not dm or not sd:
            return None
        cut = _bisect.bisect_right(sd, as_of) - 1
        if cut < 0:
            return None
        return dm[sd[cut]]["C"]

    # ── Day loop ──────────────────────────────────────────────────────────────
    for day_idx, today in enumerate(trading_days):

        # Portfolio value = cash + mark-to-market open positions
        mkt_value = cash
        for ticker, pos in positions.items():
            cp = _last_close(ticker, today)
            if cp is not None:
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
            today_ohlcv = ohlcv_by_date.get(ticker, {}).get(today)
            if today_ohlcv is None:
                continue

            day_high  = today_ohlcv["H"]
            day_low   = today_ohlcv["L"]
            day_close = today_ohlcv["C"]
            day_open  = today_ohlcv["O"]

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

        # Re-entry relaxation: after REENTRY_SILENCE_DAYS with no new entry,
        # lower the threshold by REENTRY_NET_REDUCTION to avoid missing recoveries
        silence_days = (today - last_entry_day).days if last_entry_day else 0
        effective_min_net = (
            min_net - REENTRY_NET_REDUCTION
            if reentry_relax and silence_days >= REENTRY_SILENCE_DAYS
            else min_net
        )

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

            # Cache lookup — indicators + triggers already pre-computed
            ind = ind_cache.get(ticker, {}).get(today)
            if ind is None or ind.get("error"):
                continue

            try:
                score = score_ticker(ticker, ind, {}, macro)
                score = classify_strategy(score, ind)
            except Exception as e:
                logger.debug(f"[backtest] scorer failed {ticker} on {today}: {e}")
                continue

            action     = score.get("action", "hold")
            net        = score.get("net_score", 0)
            confidence = score.get("confidence", 0.0)
            strategy   = score.get("strategy", "")

            # Apply same gates as live bot (no Claude)
            if action != "buy":
                continue
            if net < effective_min_net:
                continue
            if confidence < MIN_CONFIDENCE:
                continue
            if macro["bearish_market"]:
                continue   # suppress buys in downtrend

            # Change 1: skip strategies with demonstrated negative edge
            if filter_bad_strategies and strategy in BAD_STRATEGIES:
                continue

            # Change 4: require ADX confirmation before entering trend_follow
            # — prevents entering sideways stocks that score well on pattern alone
            if adx_filter and strategy == "trend_follow":
                adx_val = ind.get("adx")
                if adx_val is not None and adx_val < ADX_TREND_MIN:
                    continue

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
            ind        = ind_cache.get(ticker, {}).get(today, {})
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
            next_ohlcv = ohlcv_by_date.get(ticker, {}).get(next_day)
            if next_ohlcv is None:
                continue

            entry_price = next_ohlcv["O"]
            if entry_price <= 0:
                continue

            # Change 2: halve risk budget for high-volatility names
            ticker_risk_pct = (
                HIGH_VOL_RISK_PCT
                if apply_vol_cap and ticker.upper() in HIGH_VOL_TICKERS
                else RISK_PCT
            )

            # Position sizing
            shares = size_position(
                portfolio_value=cash + sum(
                    p["shares"] * entry_price for p in positions.values()
                ),
                confidence=confidence,
                atr=atr,
                price=entry_price,
                risk_pct=ticker_risk_pct,
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
            last_entry_day = next_day  # reset silence counter

    # ── Close any remaining open positions at end date ────────────────────────
    final_day = trading_days[-1] if trading_days else end
    for ticker, pos in list(positions.items()):
        exit_price = _last_close(ticker, final_day) or pos["entry_price"]
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
    filter_bad = not getattr(args, "no_strategy_filter", False)
    vol_cap    = not getattr(args, "no_vol_cap", False)
    relax      = not getattr(args, "no_reentry_relax", False)
    adx_filt   = not getattr(args, "no_adx_filter", False)
    flags_str  = (
        f"strat_filter={'on' if filter_bad else 'off'}  "
        f"vol_cap={'on' if vol_cap else 'off'}  "
        f"reentry_relax={'on' if relax else 'off'}  "
        f"adx_filter={'on' if adx_filt else 'off'}"
    )
    console.print(Panel(
        f"[bold]MoneyPrinter Backtest Report[/bold]\n"
        f"{args.start}  →  {args.end}\n"
        f"Universe: {len(args.tickers or ALL_TICKERS)} tickers  |  "
        f"Min net score: {args.min_net}\n"
        f"{flags_str}",
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
    # Improvement toggles (all on by default)
    parser.add_argument("--no-strategy-filter", action="store_true",
                        help="Disable squeeze_breakout/breakout/mixed strategy filter")
    parser.add_argument("--no-vol-cap", action="store_true",
                        help="Disable reduced position sizing for high-vol tickers")
    parser.add_argument("--no-reentry-relax", action="store_true",
                        help="Disable net-score relaxation after silence period")
    parser.add_argument("--no-adx-filter", action="store_true",
                        help="Disable ADX > 20 confirmation gate for trend_follow entries")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    if end <= start:
        console.print("[red]--end must be after --start[/red]")
        sys.exit(1)

    tickers = args.tickers or ALL_TICKERS
    # Flatten comma-separated entries (e.g. --tickers "AAPL,MSFT,NVDA" or copy-paste)
    tickers = [t.strip() for raw in tickers for t in raw.split(",") if t.strip()]
    # Always include SPY for regime detection
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers

    filter_bad = not args.no_strategy_filter
    vol_cap    = not args.no_vol_cap
    relax      = not args.no_reentry_relax
    adx_filt   = not args.no_adx_filter

    console.print(f"\n[bold cyan]MoneyPrinter Backtester[/bold cyan]")
    console.print(f"Period:  {start} → {end}  ({(end-start).days} calendar days)")
    console.print(f"Capital: ${args.capital:,.0f}")
    console.print(f"Tickers: {len(tickers)} (including SPY for regime)")
    console.print(f"Min net: {args.min_net}")
    console.print(f"Improvements: "
                  f"strategy_filter={'[green]ON[/green]' if filter_bad else '[red]OFF[/red]'}  "
                  f"vol_cap={'[green]ON[/green]' if vol_cap else '[red]OFF[/red]'}  "
                  f"reentry_relax={'[green]ON[/green]' if relax else '[red]OFF[/red]'}  "
                  f"adx_filter={'[green]ON[/green]' if adx_filt else '[red]OFF[/red]'}\n")

    results = run_backtest(
        tickers=tickers,
        start=start,
        end=end,
        starting_capital=args.capital,
        min_net=args.min_net,
        filter_bad_strategies=filter_bad,
        apply_vol_cap=vol_cap,
        reentry_relax=relax,
        adx_filter=adx_filt,
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
