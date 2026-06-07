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
DEFAULT_START      = "2024-01-01"
DEFAULT_END        = "2025-12-31"
STARTING_CAPITAL   = 100_000.0
MIN_NET_SCORE      = 60          # lowered from 70 — breakout at 70 produced only 11 trades/5mo
MAX_OPEN_POSITIONS        = 5
MAX_POSITION_PCT          = 0.18   # 18% cap — raised from 12% so risk-based sizing can reach ~1.5% realized risk
MAX_POSITION_PCT_HIGH_VOL = 0.10   # 10% cap for gap-prone / high-vol tickers (was 8%)
RISK_PCT                  = 0.02   # 2% portfolio risk per trade
ATR_STOP_MULT_MAP  = {"scalp": 4.0, "swing": 5.0, "mixed": 5.0, "intraday": 2.4}  # per-horizon stop distance (legacy — used only when fix_sizing=False)
MAX_PER_SECTOR     = 2
MAX_PER_SECTOR_ENERGY = 1       # energy stocks are oil-correlated — cap tighter
# Time exits: profitable trades get extended hold — cut losers, let winners run
TIME_EXIT_PROFIT_THRESHOLD = 0.03  # if unrealised >= 3%, extend hold
TIME_EXIT_EXTEND_DAYS      = 14    # extra calendar days (technical check gets another 14 on top)
MAX_HOLD_DAYS      = {
    # time_horizon keys (set by classify_strategy in time_horizon field)
    "scalp":            3,
    "swing":           21,
    "mixed":           10,
    # strategy name keys (stored in position["strategy"])
    "trend_follow":    21,   # let trends run; trail stop exits winners
    "mean_reversion":   5,   # reversal; cut if thesis doesn't play out
    "news_momentum":    3,   # event-driven
    "breakout":        14,   # breakouts need room to develop
    "breakdown":       14,
    "squeeze_breakout":10,
}
MIN_CONFIDENCE               = 0.65        # baseline gate (all strategies)
TREND_FOLLOW_MIN_CONFIDENCE  = 0.80        # re-enabled: strict but not zero-trade
TICKER_STOP_COOLDOWN         = 7           # days before re-entering same ticker after stop

# ── Improvement flags (all on by default) ─────────────────────────────────────
# squeeze_breakout disabled: 0% win rate even after strict momentum filtering.
# In a declining market, KC breakouts are bull traps. These setups must not
# bleed into trend_follow via reclassification — disable at strategy level.
BAD_STRATEGIES         = {"mixed", "squeeze_breakout", "trend_follow"}
# High-volatility detection — data-driven, no hardcoded ticker list (Change 9)
HIGH_VOL_ATR_PCT       = 0.05   # atr/price >= 5%: whippy daily range
HIGH_VOL_PRICE_MAX     = 5.00   # price < $5: gap/slip risk
HIGH_VOL_UNIV_PCTILE   = 80    # top-20% by universe atr_pct that day
HIGH_VOL_RISK_PCT      = 0.01   # 1% risk budget for flagged names (vs 2% normal)
# Market-regime tiered risk multiplier — scales NEW long entries only
REGIME_RISK_MULT = {
    "confirmed_uptrend": 1.0,   # SPY >= EMA50 AND EMA50 >= EMA200 → full risk
    "caution":           0.5,   # one condition met (XOR) → half risk, 1 entry/day
    "downtrend":         0.3,   # Change 4: half-size entries allowed (was 0.0); SPY shock still hard-zeros
}
SPY_SHOCK_THRESHOLD  = -0.06   # 5-day SPY return <= -6% overrides regime → 0.0 (was -4%; -4% froze bot for weeks after the Apr-2025 tariff crash during valid recovery days)
CAUTION_MAX_ENTRIES  =  2      # max new entries per day in "caution" (was 1 — too restrictive in recovery)
# Re-entry relaxation
REENTRY_SILENCE_DAYS   = 7
REENTRY_NET_REDUCTION  = 5
ADX_TREND_MIN          = 25   # strict but allows trending-market setups through
# Trailing stop — structure-based, activates earlier, giveback tightened for Change 3
TRAIL_ACTIVATE_PCT     = 0.05   # start trailing at +5%
TRAIL_GIVEBACK_PCT     = 0.03   # Change 3: trail 3% below highest (was 5%) — captures more of the move
TRAIL_TIGHT_PCT        = 0.025  # tighten to 2.5% once PARTIAL_TIGHT_PCT hit
BREAKEVEN_TRIGGER_PCT  = 0.04   # Change 2: arm breakeven at +4% close (was 2.5%) — needs real cushion first
BREAKEVEN_LOCK_PCT     = 0.005  # Change 2: lock in +0.5% (was -0.2% via BREAKEVEN_BUFFER — never lose on armed trade)
PARTIAL_TIGHT_PCT      = 0.08   # arm tight trail when CLOSE reaches +8%
BREAKEVEN_BUFFER       = 0.002  # kept for --no-breakeven-fix fallback only
# Midpoint ratchet — Change 3: fires earlier and locks more gain
RATCHET_TRIGGER_PCT    = 0.30   # lock floor once highest crosses 30% of range (was 50%)
RATCHET_LOCK_PCT       = 0.40   # lock this fraction of (target−entry) as stop floor (was 25%)
# Partial profit-taking — Change 3
PARTIAL_PROFIT_PCT      = 0.06  # take partial exit at +6% close gain
PARTIAL_PROFIT_FRACTION = 1/3   # sell this fraction of shares at partial exit
# Stale-trade breakeven
STALE_EXIT_DAYS        = 3
STALE_LOSS_THRESHOLD   = -0.01
# Mean reversion regime gate
MEAN_REV_MAX_ADX       = 20
# Breakout entry quality minimums
BREAKOUT_MIN_VOL       = 2.0   # 2x volume — breakout is already well-filtered by level + ADX
BREAKOUT_MIN_ADX       = 25
# Concentration cap for momentum strategies
MAX_TREND_FOLLOW_POSITIONS  = 2  # max concurrent open trend_follow trades
MAX_SQUEEZE_POSITIONS       = 2  # max concurrent open squeeze_breakout trades
MAX_MEAN_REV_POSITIONS      = 2  # max concurrent open mean_reversion trades

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
            # Compute velocity returns from the same historical slice — no live network call
            from bot.scorer import get_velocity_returns as _gvr
            ind.update(_gvr(ticker, sliced))
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
                  atr: float, price: float, risk_pct: float = RISK_PCT,
                  time_horizon: str = "swing",
                  is_high_vol: bool = False) -> int:
    if price <= 0 or atr <= 0:
        return 0
    stop_mult   = ATR_STOP_MULT_MAP.get(time_horizon, 2.0)
    dollar_risk = portfolio_value * risk_pct * confidence
    shares      = floor(dollar_risk / (atr * stop_mult))
    pct_cap     = MAX_POSITION_PCT_HIGH_VOL if is_high_vol else MAX_POSITION_PCT
    max_shares  = floor(portfolio_value * pct_cap / price)
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
    fix_sizing: bool = True,          # Change 1: size off actual stop distance
    fix_breakeven: bool = True,       # Change 2: raise trigger to 4%, lock in +0.5% not -0.2%
    fix_trail: bool = True,           # Change 3: tighten giveback 5%→3%, ratchet 50%→30%
    enable_partial: bool = True,      # Change 3: sell 1/3 at +6%
    fix_regime_deploy: bool = True,   # Change 4: downtrend→0.3 instead of 0.0
    disable_trend_follow: bool = False, # Change 5: exclude trend_follow strategy
    verbose: bool = False,
) -> dict:
    """
    Full day-by-day simulation. Returns results dict with trades + equity curve.
    """
    # ── Per-run effective constants (flags override module-level defaults) ────
    from bot.strategies import STRATEGY_CONFIGS as _SCFG, ATR_STOP_FLOOR as _SL_FLOOR, ATR_STOP_CAP as _SL_CAP
    _bad_strategies = set(BAD_STRATEGIES)
    if disable_trend_follow:
        _bad_strategies.add("trend_follow")
    _breakeven_trigger = BREAKEVEN_TRIGGER_PCT if fix_breakeven else 0.025
    _breakeven_lock    = BREAKEVEN_LOCK_PCT    if fix_breakeven else None  # None → old -0.2% buffer
    _trail_giveback    = TRAIL_GIVEBACK_PCT    if fix_trail     else 0.05
    _ratchet_trigger   = RATCHET_TRIGGER_PCT   if fix_trail     else 0.50
    _ratchet_lock      = RATCHET_LOCK_PCT      if fix_trail     else 0.25

    if verbose:
        logger.setLevel(logging.INFO)

    def vprint(*args, **kwargs):
        if verbose:
            console.print(*args, **kwargs)

    bars_map     = load_all_bars(tickers, start, end)
    trading_days = get_trading_days(bars_map, start, end)

    if not trading_days:
        console.print("[red]No trading days found — check date range and API credentials[/red]")
        return {}

    console.print(f"[cyan]Simulating {len(trading_days)} trading days "
                  f"({trading_days[0]} → {trading_days[-1]})[/cyan]")

    # ── Pre-compute all indicators (parallel, replaces per-day per-ticker calls) ─
    ind_cache = precompute_all_indicators(tickers, bars_map, trading_days)

    # ── Pre-fetch historical earnings dates for earnings-exit guard ───────────
    # For each ticker, build a set of dates on which earnings were released so
    # we can exit held positions the day BEFORE earnings rather than holding
    # through a binary gap event.
    console.print("[cyan]Pre-fetching earnings dates…[/cyan]")
    import yfinance as _yf
    import logging as _logging
    _logging.getLogger("yfinance").setLevel(_logging.CRITICAL)  # suppress ETF "no earnings" noise
    _earnings_map: dict[str, set] = {}
    for _t in tickers:
        if _t == "SPY":
            continue
        try:
            _ed = _yf.Ticker(_t).earnings_dates
            if _ed is not None and not _ed.empty:
                _dates: set[date] = set()
                for _idx in _ed.index:
                    _d = _idx.date() if hasattr(_idx, "date") else _idx
                    # Keep a wider window (±3 days of period) to catch release timing variance
                    if (start - timedelta(days=3)) <= _d <= (end + timedelta(days=3)):
                        _dates.add(_d)
                _earnings_map[_t] = _dates
        except Exception:
            _earnings_map[_t] = set()
    console.print(f"[green]Earnings dates loaded for {len(_earnings_map)} tickers[/green]")

    # ── State ─────────────────────────────────────────────────────────────────
    cash      = starting_capital
    positions: dict[str, dict] = {}  # ticker -> position record
    all_trades: list[dict]     = []
    equity_curve: list[dict]   = []
    last_entry_day: date | None  = None  # tracks last day an entry was made
    ticker_last_stop: dict[str, date] = {}  # ticker -> date of last stop/stale exit

    # Macro approximation: SPY above EMA50 = bull, else caution
    spy_bars = bars_map.get("SPY")

    # Pre-compute SPY regime for every trading day (fast: done once, not per-day)
    import bisect as _bisect
    _spy_dates = (
        [(ts.date() if hasattr(ts, "date") else ts) for ts in spy_bars.index]
        if spy_bars is not None else []
    )
    _spy_close  = spy_bars["Close"] if spy_bars is not None else None
    _spy_ema50  = spy_bars["Close"].ewm(span=50,  adjust=False).mean() if spy_bars is not None else None
    _spy_ema200 = spy_bars["Close"].ewm(span=200, adjust=False).mean() if spy_bars is not None else None
    _spy_ret5d  = spy_bars["Close"].pct_change(5) if spy_bars is not None else None

    def _spy_regime(as_of: date) -> str:
        """
        Three-state SPY regime:
          confirmed_uptrend — SPY >= EMA50 AND EMA50 >= EMA200 → risk_mult 1.0
          caution           — one but not both (XOR)           → risk_mult 0.5
          downtrend         — SPY < EMA50 AND EMA50 < EMA200   → risk_mult 0.0
        """
        if not _spy_dates:
            return "confirmed_uptrend"
        try:
            cut = _bisect.bisect_right(_spy_dates, as_of)
            if cut < 200:
                return "confirmed_uptrend"
            price  = float(_spy_close.iloc[cut - 1])
            ema50  = float(_spy_ema50.iloc[cut - 1])
            ema200 = float(_spy_ema200.iloc[cut - 1])
            above_50        = price >= ema50
            ema50_above_200 = ema50  >= ema200
            if above_50 and ema50_above_200:
                return "confirmed_uptrend"
            elif above_50 or ema50_above_200:   # XOR
                return "caution"
            else:
                return "downtrend"
        except Exception:
            return "confirmed_uptrend"

    def _spy_stressed(as_of: date) -> bool:
        """True when SPY has dropped >4% over 5 days (shock override).
        Overrides regime_mult to 0.0 regardless of EMA state."""
        if _spy_ret5d is None or not _spy_dates:
            return False
        try:
            cut = _bisect.bisect_right(_spy_dates, as_of)
            if cut < 10:
                return False
            return float(_spy_ret5d.iloc[cut - 1]) <= SPY_SHOCK_THRESHOLD
        except Exception:
            return False

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
    _prev_regime: str = ""
    _regime_days: dict[str, int] = {"confirmed_uptrend": 0, "caution": 0, "downtrend": 0, "shocked": 0}
    for day_idx, today in enumerate(trading_days):

        # Portfolio value = cash + mark-to-market open positions
        mkt_value = cash
        for ticker, pos in positions.items():
            cp = _last_close(ticker, today)
            if cp is not None:
                mkt_value += pos["shares"] * cp

        equity_curve.append({"date": today, "equity": round(mkt_value, 2), "cash": round(cash, 2)})

        regime      = _spy_regime(today)
        regime_mult = REGIME_RISK_MULT.get(regime, 1.0)
        if not fix_regime_deploy and regime == "downtrend":
            regime_mult = 0.0  # revert to hard block when flag is off
        if _spy_stressed(today):
            regime_mult = 0.0  # SPY 5-day shock always hard-zeros regardless of fix_regime_deploy
            _regime_days["shocked"] += 1
        else:
            _regime_days[regime] = _regime_days.get(regime, 0) + 1

        if regime != _prev_regime:
            _rc = {"confirmed_uptrend": "green", "caution": "yellow", "downtrend": "red"}.get(regime, "white")
            vprint(
                f"\n[bold {_rc}]REGIME → {regime.upper()}[/bold {_rc}] "
                f"({today}) risk_mult={regime_mult:.1f} "
                f"equity=${mkt_value:,.0f}"
            )
            _prev_regime = regime

        macro  = {
            "vix": 18.0,
            "spy_regime": regime,
            "bearish_market": regime == "downtrend",
            "vix_multiplier": 1.0,
            "regime_mult": regime_mult,
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
            # Look up by strategy name first (trend_follow → 10), fall back to time_horizon
            _strat   = pos.get("strategy", "")
            _horizon = pos.get("time_horizon", "swing")
            max_hold = MAX_HOLD_DAYS.get(_strat, MAX_HOLD_DAYS.get(_horizon, 7))
            age_days = (today - pos["entry_date"]).days

            exit_price  = None
            exit_reason = None

            # ── Partial profit: execute at today's open if flagged by prior bar ─
            # Flag is set when prior-bar CLOSE hit PARTIAL_PROFIT_PCT.
            # Executing at today's open is look-ahead-free (prior close → next open).
            if enable_partial and pos.get("partial_pending") and not pos.get("partial_taken"):
                partial_shares = max(1, pos["shares"] // 3)
                partial_pnl    = (day_open - entry) * partial_shares
                partial_pnl_pct = (day_open - entry) / entry * 100
                cash  += day_open * partial_shares
                pos["shares"]       -= partial_shares
                pos["partial_taken"]  = True
                pos["partial_pending"]= False
                all_trades.append({
                    "ticker":       ticker,
                    "entry_date":   pos["entry_date"].isoformat(),
                    "exit_date":    today.isoformat(),
                    "entry_price":  round(entry, 2),
                    "exit_price":   round(day_open, 2),
                    "shares":       partial_shares,
                    "stop_loss":    round(stop, 2),
                    "take_profit":  round(target, 2) if target else None,
                    "initial_stop_loss": round(pos.get("initial_stop_loss", stop), 2),
                    "pnl_dollar":   round(partial_pnl, 2),
                    "pnl_pct":      round(partial_pnl_pct, 2),
                    "hold_days":    age_days,
                    "exit_reason":  "partial_profit",
                    "strategy":     pos.get("strategy", "unknown"),
                    "net_score":    pos.get("net_score", 0),
                    "confidence":   pos.get("confidence", 0),
                    "signals":      pos.get("signals", []),
                })
                vprint(f"  [cyan]PARTIAL[/cyan] {ticker} {today} | "
                       f"sold {partial_shares} @ ${day_open:.2f} "
                       f"({partial_pnl_pct:+.2f}%) | {pos['shares']} remain")
                shares = pos["shares"]
                if shares < 1:
                    closed_tickers.append(ticker)
                    continue

            # ── Exit checks: use stops armed by PRIOR bars only ───────────────
            # Breakeven and trailing-stop updates happen at END of this block so
            # today's high cannot raise the stop before today's low is checked.
            # (Intra-bar order of high vs low is unknowable from daily bars.)
            trail_price = pos.get("trailing_stop_price")

            # Earnings exit: if earnings are tomorrow, exit at today's close.
            # Holding a swing position through a binary event is a coin-flip, not a trade.
            _earn_dates = _earnings_map.get(ticker, set())
            _tomorrow   = today + timedelta(days=1)
            if any((ed - today).days in (0, 1) for ed in _earn_dates):
                exit_price  = day_close
                exit_reason = "earnings_exit"
                vprint(
                    f"  [magenta]EARNINGS EXIT[/magenta] {ticker} {today} | "
                    f"earnings imminent | close={day_close:.2f} "
                    f"unrealised={((day_close-entry)/entry)*100:+.1f}%"
                )

            if exit_price is None and day_low <= stop:
                exit_price  = max(min(stop, day_open), day_low)
                exit_reason = "stop"
            if exit_price is None and target and day_high >= target:
                exit_price  = min(target, day_high)
                exit_reason = "target"
            if exit_price is None and trail_price is not None and day_low <= trail_price:
                exit_price  = trail_price
                exit_reason = "trailing_stop"
            if exit_price is None and age_days >= max_hold:
                unrealised_pct = (day_close - entry) / entry if entry > 0 else 0.0

                # Technical continuation check — if trend still intact, keep holding
                _ind_t   = ind_cache.get(ticker, {}).get(today, {})
                _adx_t   = float(_ind_t.get("adx") or 0)
                _mh_t    = float(_ind_t.get("macd_hist") or 0)
                _mh_p_t  = float(_ind_t.get("macd_hist_prev1") or 0)
                _rsi_t   = float(_ind_t.get("rsi") or 50)
                _e50_t   = float(_ind_t.get("ema50") or 0)
                _e200_t  = float(_ind_t.get("ema200") or 0)
                _trend_intact = (
                    _adx_t > 20
                    and _mh_t > 0
                    and _mh_t >= _mh_p_t   # MACD still accelerating
                    and _e50_t > 0 and day_close > _e50_t
                    and _e200_t > 0 and day_close > _e200_t
                    and _rsi_t < 80
                )

                vprint(
                    f"  [yellow]TIME EXIT CHECK[/yellow] {ticker} {today} | "
                    f"age={age_days}d max={max_hold}d ({_strat}) | "
                    f"unrealised={unrealised_pct:+.1%} trend_intact={_trend_intact} "
                    f"adx={_adx_t:.0f} macd={_mh_t:.3f} rsi={_rsi_t:.0f}"
                )

                hard_cap = max_hold + TIME_EXIT_EXTEND_DAYS * 2  # absolute ceiling
                if unrealised_pct >= TIME_EXIT_PROFIT_THRESHOLD and _trend_intact:
                    # Profitable + trend still good — let trailing stop do the work
                    if age_days >= hard_cap:
                        exit_price  = day_close
                        exit_reason = "time_exit_hard_cap"
                        vprint(f"    → [red]HARD CAP EXIT[/red] age={age_days}d >= {hard_cap}d")
                    else:
                        vprint(f"    → [green]HOLDING[/green] trend intact, hard cap at {hard_cap}d")
                elif unrealised_pct >= TIME_EXIT_PROFIT_THRESHOLD:
                    # Profitable but trend fading — limited extension
                    if age_days >= max_hold + TIME_EXIT_EXTEND_DAYS:
                        exit_price  = day_close
                        exit_reason = "time_exit_extended"
                        vprint(f"    → [red]EXTENDED EXIT[/red] trend gone, age={age_days}d")
                    else:
                        vprint(f"    → [yellow]SHORT HOLD[/yellow] profitable, trend fading, {max_hold+TIME_EXIT_EXTEND_DAYS-age_days}d left")
                else:
                    exit_price  = day_close
                    exit_reason = "time_exit"
                    vprint(f"    → [red]TIME EXIT[/red] unrealised={unrealised_pct:+.1%}")

            # ── Stale exit: dead trade not moving ─────────────────────────────
            if exit_price is None:
                unrealised_pct = (day_close - entry) / entry if entry > 0 else 0.0
                if age_days >= STALE_EXIT_DAYS and unrealised_pct < STALE_LOSS_THRESHOLD:
                    # For trend_follow: skip stale exit if trend is still technically intact.
                    # Trends take time to develop — a -1% after 3 days isn't broken, just slow.
                    _skip_stale = False
                    if _strat == "trend_follow":
                        _ind_stale = ind_cache.get(ticker, {}).get(today, {})
                        _adx_s  = float(_ind_stale.get("adx") or 0)
                        _e50_s  = float(_ind_stale.get("ema50") or 0)
                        _mh_s   = float(_ind_stale.get("macd_hist") or 0)
                        _mhp_s  = float(_ind_stale.get("macd_hist_prev1") or 0)
                        if _adx_s > 22 and _e50_s > 0 and day_close > _e50_s and _mh_s >= _mhp_s:
                            _skip_stale = True
                            vprint(f"  [dim]stale suppressed {ticker} | trend intact ADX={_adx_s:.0f}[/dim]")
                    if not _skip_stale:
                        if pos["stop_loss"] < entry * (1 - BREAKEVEN_BUFFER):
                            stale_stop = round(entry * (1 - BREAKEVEN_BUFFER), 2)
                            pos["stop_loss"] = stale_stop
                            stop = stale_stop
                        if day_close < stop:
                            exit_price  = day_close
                            exit_reason = "stale_exit"

            # ── UPDATE stops for NEXT bar — armed off today's CLOSE ───────────
            # Only runs when the trade remains open this bar.
            # Using close (not high) prevents intraday wicks from tightening the
            # stop and triggering a stop on the same candle's low (look-ahead).
            if exit_price is None:
                highest = max(pos.get("highest_price_seen", entry), day_high)
                pos["highest_price_seen"] = highest
                gain_pct   = (highest   - entry) / entry if entry > 0 else 0.0
                close_gain = (day_close - entry) / entry if entry > 0 else 0.0

                # Breakeven off CLOSE — Change 2: lock in +0.5% once close reaches trigger
                if close_gain >= _breakeven_trigger:
                    if _breakeven_lock is not None:
                        # New behaviour: stop never below entry + lock_pct (locked positive)
                        _be_floor = round(entry * (1 + _breakeven_lock), 2)
                        if pos["stop_loss"] < _be_floor:
                            pos["stop_loss"] = _be_floor
                    else:
                        # Old behaviour: stop at entry*(1-buffer) — slightly negative
                        if pos["stop_loss"] < entry * (1 - BREAKEVEN_BUFFER):
                            pos["stop_loss"] = round(entry * (1 - BREAKEVEN_BUFFER), 2)

                # Midpoint ratchet — Change 3: fires at 30% of range, locks 40% (was 50%/25%)
                if target and target > entry > 0:
                    _tp_range = target - entry
                    if highest >= entry + _ratchet_trigger * _tp_range:
                        _ratchet_stop = round(entry + _ratchet_lock * _tp_range, 2)
                        if pos["stop_loss"] < _ratchet_stop:
                            pos["stop_loss"] = _ratchet_stop

                # Tight trail armed off CLOSE >= +8% (not intraday wick)
                if not pos.get("tight_trail_activated") and close_gain >= PARTIAL_TIGHT_PCT:
                    pos["tight_trail_activated"] = True

                # Change 3: use tightened _trail_giveback (3% vs 5%)
                trail_pct = TRAIL_TIGHT_PCT if pos.get("tight_trail_activated") else _trail_giveback

                # Trailing stop: activates once highest-seen >= TRAIL_ACTIVATE_PCT
                if gain_pct >= TRAIL_ACTIVATE_PCT:
                    pct_trail = round(highest * (1.0 - trail_pct), 2)
                    _sd_t = sorted_dates_by_ticker.get(ticker, [])
                    _t_i  = _bisect.bisect_right(_sd_t, today) - 1
                    struct_trail = pct_trail
                    if _t_i >= 2:
                        _recent = _sd_t[max(0, _t_i - 2) : _t_i + 1]
                        _r_lows = [ohlcv_by_date[ticker][d]["L"] for d in _recent
                                   if d in ohlcv_by_date.get(ticker, {})]
                        if len(_r_lows) >= 2:
                            struct_trail = round(min(_r_lows) * 0.995, 2)
                    new_trail = max(pct_trail, struct_trail)
                    cur_trail = pos.get("trailing_stop_price")
                    if cur_trail is None or new_trail > cur_trail:
                        pos["trailing_stop_price"] = new_trail

                # Partial profit flag — Change 3: armed off CLOSE, executes at NEXT open
                if (enable_partial
                        and close_gain >= PARTIAL_PROFIT_PCT
                        and not pos.get("partial_taken")
                        and not pos.get("partial_pending")):
                    pos["partial_pending"] = True

            if exit_price is not None:
                pnl_dollar = (exit_price - entry) * shares
                pnl_pct    = (exit_price - entry) / entry * 100
                cash      += exit_price * shares
                _ec = "green" if pnl_dollar >= 0 else "red"
                vprint(
                    f"  [bold {_ec}]EXIT[/bold {_ec}] {ticker} {today} | "
                    f"{exit_reason} | {age_days}d | "
                    f"${entry:.2f}→${exit_price:.2f} "
                    f"[{_ec}]{pnl_pct:+.2f}% (${pnl_dollar:+,.0f})[/{_ec}] | "
                    f"stop=${stop:.2f} "
                    f"trail={'${:.2f}'.format(trail_price) if trail_price else 'none'} "
                    f"high=${pos.get('highest_price_seen', entry):.2f}"
                )
                trade_rec  = {
                    "ticker":           ticker,
                    "entry_date":       pos["entry_date"].isoformat(),
                    "exit_date":        today.isoformat(),
                    "entry_price":      round(entry, 2),
                    "exit_price":       round(exit_price, 2),
                    "shares":           shares,
                    "stop_loss":        round(stop, 2),
                    "initial_stop_loss":round(pos.get("initial_stop_loss", stop), 2),
                    "take_profit":      round(target, 2) if target else None,
                    "pnl_dollar":       round(pnl_dollar, 2),
                    "pnl_pct":          round(pnl_pct, 2),
                    "hold_days":        age_days,
                    "exit_reason":      exit_reason,
                    "strategy":         pos.get("strategy", "unknown"),
                    "net_score":        pos.get("net_score", 0),
                    "confidence":       pos.get("confidence", 0),
                    "signals":          pos.get("signals", []),
                }
                all_trades.append(trade_rec)
                closed_tickers.append(ticker)
                if exit_reason in ("stop", "stale_exit"):
                    ticker_last_stop[ticker] = today

        for t in closed_tickers:
            del positions[t]

        # ── 2. Generate signals using today's close ───────────────────────────
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

        # ── Universe ATR/price 80th-percentile for dynamic high-vol detection ────
        # Computed once per day; used in sizing to flag whippy names.
        _atr_pcts: list[float] = []
        for _t in tickers:
            if _t == "SPY":
                continue
            _i = ind_cache.get(_t, {}).get(today)
            if _i and not _i.get("error"):
                _a = float(_i.get("atr") or 0)
                _c = float(_i.get("current_price") or 0)
                if _a > 0 and _c > 0:
                    _atr_pcts.append(_a / _c)
        _atr_pcts.sort()
        _high_vol_univ_threshold = (
            _atr_pcts[int(len(_atr_pcts) * HIGH_VOL_UNIV_PCTILE / 100)]
            if len(_atr_pcts) >= 10 else float("inf")
        )

        new_entries_today = 0   # per-day counter for regime-aware entry cap
        new_signals: list[dict] = []

        for ticker in tickers:
            if ticker in positions:
                continue
            if ticker == "SPY":
                continue
            # Per-ticker cooldown: don't re-enter a ticker for TICKER_STOP_COOLDOWN days
            # after it was stopped out — prevents double-dipping into the same failing trade
            if ticker in ticker_last_stop:
                if (today - ticker_last_stop[ticker]).days < TICKER_STOP_COOLDOWN:
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
                vprint(f"  [dim]skip {ticker} {strategy} net={net} | conf={confidence:.2f} < {MIN_CONFIDENCE:.2f}[/dim]")
                continue
            # Regime gate: 0.0 = no new longs (downtrend or shock)
            if regime_mult == 0.0:
                vprint(f"  [dim]skip {ticker} {strategy} net={net} | regime={regime} blocks all longs[/dim]")
                continue

            # Skip strategies with no coherent edge
            if filter_bad_strategies and strategy in _bad_strategies:
                vprint(f"  [dim]skip {ticker} | bad_strategy={strategy} net={net} conf={confidence:.2f}[/dim]")
                continue

            # ── Strategy-regime alignment ─────────────────────────────────────
            # trend_follow, squeeze_breakout, and mean_reversion only work in a
            # clearly trending broad market. In caution or downtrend they produce
            # a high rate of false signals and fast stops.
            if strategy in ("trend_follow", "squeeze_breakout", "mean_reversion") and regime != "confirmed_uptrend":
                vprint(f"  [dim]skip {ticker} | {strategy} blocked in {regime}[/dim]")
                continue

            # trend_follow guards
            if adx_filter and strategy == "trend_follow":
                adx_val = ind.get("adx")
                if adx_val is not None and adx_val < ADX_TREND_MIN:
                    vprint(f"  [dim]skip {ticker} | trend_follow: ADX={adx_val:.1f} < {ADX_TREND_MIN}[/dim]")
                    continue
                # SPY must be rising — index sinking means individual trends fail
                _spy_5d = float(_spy_ret5d.iloc[_bisect.bisect_right(_spy_dates, today) - 1]) if _spy_ret5d is not None and _spy_dates else None
                if _spy_5d is not None and _spy_5d <= 0.005:
                    vprint(f"  [dim]skip {ticker} | trend_follow: SPY 5d return={_spy_5d:.2%} <= 0.5%[/dim]")
                    continue
                # Stock must have real recent momentum (+1.5% 5d) and confirmed medium-term trend (+2% 1m)
                r5d = ind.get("return_5d")
                if r5d is not None and r5d <= 0.015:
                    vprint(f"  [dim]skip {ticker} | trend_follow: return_5d={r5d:.2%} <= 1.5%[/dim]")
                    continue
                # Don't enter stocks already up >12% in 5 days — overextended, late entry
                if r5d is not None and r5d > 0.12:
                    vprint(f"  [dim]skip {ticker} | trend_follow: return_5d={r5d:.2%} > 12% overextended[/dim]")
                    continue
                r1m = ind.get("return_1m")
                if r1m is not None and r1m <= 0.02:
                    vprint(f"  [dim]skip {ticker} | trend_follow: return_1m={r1m:.2%} <= 2%[/dim]")
                    continue
                # RSI cap: tightened to 68 — don't enter stocks already extended
                _rsi_tf = float(ind.get("rsi") or 0)
                if _rsi_tf > 68:
                    vprint(f"  [dim]skip {ticker} | trend_follow: RSI={_rsi_tf:.1f} > 68 overbought[/dim]")
                    continue
                # Volume confirmation: require institutional participation
                _vol_tf = float(ind.get("volume_ratio") or 0)
                if _vol_tf < 1.5:
                    vprint(f"  [dim]skip {ticker} | trend_follow: vol_ratio={_vol_tf:.2f} < 1.5[/dim]")
                    continue
                # MACD must be accelerating (momentum building, not fading).
                # Use explicit None check — `or 0` treats 0.0 as missing and would
                # always skip when MACD hist is exactly zero (crossed signal line).
                # If either value is absent, skip the check rather than defaulting 0<=0.
                _mh   = ind.get("macd_hist")
                _mh_p = ind.get("macd_hist_prev1")
                if _mh is not None and _mh_p is not None and float(_mh) <= float(_mh_p):
                    logger.debug(f"[backtest] {ticker} trend_follow skip {today}: MACD not accelerating ({_mh:.3f} <= {_mh_p:.3f})")
                    vprint(f"  [dim]skip {ticker} | trend_follow: MACD_hist {_mh:.3f} not > prev {_mh_p:.3f}[/dim]")
                    continue
                # Reclassification guard: squeeze+KC signals → not a clean trend trade
                _sigs = set(score.get("signals_triggered", []))
                if "bb_squeeze_detected" in _sigs and "kc_breakout_bull" in _sigs:
                    logger.debug(f"[backtest] {ticker} trend_follow skip {today}: squeeze reclassification guard")
                    vprint(f"  [dim]skip {ticker} | trend_follow: squeeze reclassification guard[/dim]")
                    continue
                # Pre-earnings guard: don't open within 5 calendar days of earnings binary event
                _tf_earn = _earnings_map.get(ticker, set())
                if any(1 <= (ed - today).days <= 6 for ed in _tf_earn):
                    vprint(f"  [dim]skip {ticker} | trend_follow: earnings within 5d[/dim]")
                    continue
                # Signal-day momentum: stock must be going UP on signal day.
                # Entering a stock that closed down today means we're fading immediate momentum.
                _r1d_tf = ind.get("return_1d")
                if _r1d_tf is not None and _r1d_tf <= 0:
                    vprint(f"  [dim]skip {ticker} | trend_follow: return_1d={_r1d_tf:.2%} <= 0 (fading)[/dim]")
                    continue
                # High-conviction gate: trend_follow needs stronger confidence than baseline
                if confidence < TREND_FOLLOW_MIN_CONFIDENCE:
                    vprint(
                        f"  [dim]skip {ticker} | trend_follow: conf={confidence:.2f} < "
                        f"{TREND_FOLLOW_MIN_CONFIDENCE:.2f} (need higher conviction)[/dim]"
                    )
                    continue
                # Sector-regime guards: macro-sensitive sectors only trend reliably
                # in confirmed uptrends; block them in caution/downtrend regimes.
                _tf_sec = _SECTOR_OF.get(ticker.upper())
                if _tf_sec in ("energy", "financials", "defense") and regime != "confirmed_uptrend":
                    vprint(f"  [dim]skip {ticker} | trend_follow: {_tf_sec} blocked in {regime}[/dim]")
                    continue

            # ── Mean reversion guard ──────────────────────────────────────────
            # RSI<30 + BB extreme + MACD improving = reversal forming (not free-fall).
            # MACD improving filters buying into ongoing collapses — single oversold
            # readings are insufficient in trending-down markets.
            if strategy == "mean_reversion":
                _cp    = ind.get("current_price") or 0
                _e50   = ind.get("ema50")   or 0
                _e200  = ind.get("ema200")  or 0
                _adx   = ind.get("adx")     or 0
                _rsi   = ind.get("rsi")     or 50
                _bb_pb = ind.get("bb_pctb")
                _mh    = ind.get("macd_hist") or 0
                _mh_p  = ind.get("macd_hist_prev1") or 0
                if _cp > 0 and _e50 > 0 and _e200 > 0 and _cp < _e50 and _cp < _e200:
                    logger.debug(f"[backtest] {ticker} mean_rev skip {today}: downtrend")
                    continue
                if _adx > MEAN_REV_MAX_ADX:
                    logger.debug(f"[backtest] {ticker} mean_rev skip {today}: trending adx={_adx:.1f}")
                    continue
                _rsi_extreme = _rsi < 30 or _rsi > 70
                _bb_extreme  = _bb_pb is not None and (_bb_pb < 0.15 or _bb_pb > 0.85)
                if not (_rsi_extreme and _bb_extreme):
                    logger.debug(f"[backtest] {ticker} mean_rev skip {today}: not extreme rsi={_rsi:.1f} bb={_bb_pb}")
                    continue
                if _mh <= _mh_p:
                    logger.debug(f"[backtest] {ticker} mean_rev skip {today}: MACD not improving ({_mh:.3f} <= {_mh_p:.3f})")
                    continue
                # Long mean_reversion: RSI must be genuinely oversold, not ambiguous mid-range
                # RSI 45-70 is "normal"; only RSI<45 represents real oversold territory
                if action == "buy" and _rsi >= 45:
                    vprint(f"  [dim]skip {ticker} | mean_rev long: RSI={_rsi:.1f} >= 45 not oversold[/dim]")
                    continue
                # Knife-catch guard: if down >25% in 3 months it's fundamental, not technical
                _r3m = ind.get("return_3m")
                if _r3m is not None and _r3m < -0.25:
                    vprint(f"  [dim]skip {ticker} | mean_rev: return_3m={_r3m:.1%} < -25% knife_catch[/dim]")
                    continue
                _mr_open = sum(1 for p in positions.values() if p.get("strategy") == "mean_reversion")
                if _mr_open >= MAX_MEAN_REV_POSITIONS:
                    continue

            # squeeze_breakout is in BAD_STRATEGIES — no guard needed here.

            # ── News momentum quality guard ───────────────────────────────────
            # news_momentum has the lowest R:R (2.0x) and is news-feed dependent.
            # Require conf >= 0.80 and volume >= 2x to filter low-conviction entries.
            if strategy == "news_momentum":
                _nm_vol = float(ind.get("volume_ratio") or 0)
                if confidence < 0.80:
                    vprint(f"  [dim]skip {ticker} | news_momentum: conf={confidence:.2f} < 0.80[/dim]")
                    continue
                if _nm_vol < 2.0:
                    vprint(f"  [dim]skip {ticker} | news_momentum: vol_ratio={_nm_vol:.2f} < 2.0x[/dim]")
                    continue

            # ── Breakout guard ────────────────────────────────────────────────
            # On top of the classifier's conditions, require strong volume,
            # strong trend momentum, uptrend context, and prior consolidation.
            if strategy == "breakout":
                _vol   = ind.get("volume_ratio") or 0
                _adx   = ind.get("adx")          or 0
                _cp    = ind.get("current_price") or 0
                _e50   = ind.get("ema50")         or 0
                _e200  = ind.get("ema200")        or 0

                _macd_brk = ind.get("macd_hist") or 0
                _skip_reason = None
                if _vol < BREAKOUT_MIN_VOL:
                    _skip_reason = f"vol_ratio={_vol:.2f} < {BREAKOUT_MIN_VOL}"
                elif _macd_brk <= 0:
                    _skip_reason = f"macd_hist={_macd_brk:.3f} not positive — breakout lacks momentum"
                elif _adx < BREAKOUT_MIN_ADX:
                    _skip_reason = f"adx={_adx:.1f} < {BREAKOUT_MIN_ADX}"
                elif _e50 > 0 and _cp < _e50:
                    _skip_reason = f"price={_cp:.2f} < ema50={_e50:.2f}"
                elif _e200 > 0 and _cp < _e200:
                    _skip_reason = f"price={_cp:.2f} < ema200={_e200:.2f}"
                else:
                    # 15-day consolidation check: range must be < 15% to confirm
                    # the stock was coiling before the break, not already extended.
                    # Uses preloaded OHLCV — fails open if data unavailable.
                    try:
                        _sd = sorted_dates_by_ticker.get(ticker, [])
                        _dm = ohlcv_by_date.get(ticker, {})
                        _today_i = _bisect.bisect_right(_sd, today) - 1
                        if _today_i >= 15:
                            _prior = _sd[_today_i - 15 : _today_i]
                            _highs = [_dm[d]["H"] for d in _prior if d in _dm]
                            _lows  = [_dm[d]["L"] for d in _prior if d in _dm]
                            if len(_highs) >= 10 and min(_lows) > 0:
                                _range = (max(_highs) - min(_lows)) / min(_lows)
                                if _range >= 0.15:
                                    _skip_reason = f"no_consolidation range={_range:.1%} >= 15%"
                    except Exception:
                        pass  # fail open — don't block on data errors

                if _skip_reason:
                    logger.debug(
                        f"[backtest] {ticker} breakout skip {today}: {_skip_reason}"
                    )
                    continue

            # Sector cap — energy gets a tighter cap (oil-correlated, cluster losses)
            sec = _SECTOR_OF.get(ticker.upper())
            _sec_cap = MAX_PER_SECTOR_ENERGY if sec == "energy" else MAX_PER_SECTOR
            if sec and sector_counts.get(sec, 0) >= _sec_cap:
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
            strategy     = score.get("strategy", "swing")
            time_horizon = score.get("time_horizon", "swing")

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

            signal_day_close = ohlcv_by_date.get(ticker, {}).get(today, {}).get("C") or 0

            # ── Gap-chase guard: skip if stock already ran > 3% past signal close ──
            if signal_day_close > 0 and entry_price > signal_day_close * 1.03:
                logger.debug(
                    f"[backtest] {ticker} {next_day} gap-chase skip: "
                    f"open={entry_price:.2f} signal_close={signal_day_close:.2f}"
                )
                continue

            # ── Breakout failed-break rejection ───────────────────────────────
            # If next-day open is already below the breakout level the break failed
            # overnight — don't chase it (fill is no better than a broken setup).
            if strategy == "breakout":
                _R1_brk   = float(ind.get("R1") or 0)
                _w52h_brk = float(ind.get("wk52_high") or 0)
                _brk_lvl  = (
                    max(_R1_brk, _w52h_brk * 0.99)
                    if _w52h_brk > 0 and _R1_brk > 0
                    else (_R1_brk or _w52h_brk * 0.99)
                )
                if _brk_lvl > 0 and entry_price < _brk_lvl * 0.99:
                    vprint(
                        f"  [dim]skip {ticker} | breakout: entry ${entry_price:.2f} < "
                        f"level ${_brk_lvl:.2f} (failed break on open)[/dim]"
                    )
                    continue

            # ── Gap-down guard for momentum strategies ────────────────────────
            if strategy in ("trend_follow", "breakout", "squeeze_breakout"):
                if signal_day_close > 0 and (signal_day_close - entry_price) > atr:
                    logger.debug(
                        f"[backtest] {ticker} {next_day} gap-down skip: "
                        f"open={entry_price:.2f} prev_close={signal_day_close:.2f} atr={atr:.2f}"
                    )
                    continue

            # ── Max concurrent trend_follow cap ───────────────────────────────
            if strategy == "trend_follow":
                tf_open = sum(1 for p in positions.values() if p.get("strategy") == "trend_follow")
                if tf_open >= MAX_TREND_FOLLOW_POSITIONS:
                    continue

            # ── Dynamic high-vol classification ───────────────────────────────
            # Three sources: ATR%, price floor, universe percentile, and scorer
            # flag (which covers known gap-prone tickers like AVGO via HIGH_VOLATILITY_TICKERS).
            atr_pct = atr / entry_price if entry_price > 0 else 0.0
            _hv_reasons: list[str] = []
            if apply_vol_cap:
                if atr_pct >= HIGH_VOL_ATR_PCT:
                    _hv_reasons.append(f"atr_pct={atr_pct:.1%}>={HIGH_VOL_ATR_PCT:.0%}")
                if entry_price < HIGH_VOL_PRICE_MAX:
                    _hv_reasons.append(f"price={entry_price:.2f}<{HIGH_VOL_PRICE_MAX:.2f}")
                if atr_pct >= _high_vol_univ_threshold:
                    _hv_reasons.append(f"top20%_vol(thresh={_high_vol_univ_threshold:.1%})")
                if score.get("high_vol_flag"):
                    _hv_reasons.append("scorer:high_vol_ticker")
            is_high_vol = bool(_hv_reasons)
            if is_high_vol:
                logger.info(
                    f"[backtest] {ticker} {today} high-vol: {'; '.join(_hv_reasons)}"
                )

            ticker_risk_pct = (HIGH_VOL_RISK_PCT if is_high_vol else RISK_PCT) * regime_mult

            # Per-day entry cap in caution (only 1 new long allowed per day)
            if regime == "caution" and new_entries_today >= CAUTION_MAX_ENTRIES:
                continue

            # ── Stop/target first — sizing must know the actual stop distance ──
            # Change 1: compute stop before sizing so both use the same distance.
            _rr      = score.get("risk_reward") or 2.5
            _sl_mult = _SCFG.get(strategy, _SCFG["mixed"])["sl_atr_mult"]
            _atr_pct = (atr / entry_price) if entry_price > 0 else 0.02
            _sl_pct  = max(_SL_FLOOR, min(_atr_pct * _sl_mult, _SL_CAP))
            stop_loss   = round(entry_price * (1 - _sl_pct), 2)
            take_profit = round(entry_price * (1 + _sl_pct * _rr), 2)

            # Position sizing — value each open position at its own last close (mark-to-market)
            mark_to_market = cash + sum(
                p["shares"] * (_last_close(t, today) or p["entry_price"])
                for t, p in positions.items()
            )
            pos_cap = MAX_POSITION_PCT_HIGH_VOL if is_high_vol else MAX_POSITION_PCT
            if fix_sizing:
                # Change 1: size off the SAME stop distance that is placed — RISK_PCT is now real
                stop_dist = max(entry_price * 0.005, entry_price - stop_loss)
                dollar_risk = mark_to_market * ticker_risk_pct * confidence
                shares = floor(dollar_risk / stop_dist) if stop_dist > 0 else 0
                shares = min(shares, floor(mark_to_market * pos_cap / entry_price))
            else:
                shares = size_position(
                    portfolio_value=mark_to_market,
                    confidence=confidence,
                    atr=atr,
                    price=entry_price,
                    risk_pct=ticker_risk_pct,
                    time_horizon=time_horizon,
                    is_high_vol=is_high_vol,
                )
            if shares < 1:
                continue

            cost = shares * entry_price
            if cost > cash:
                shares = floor(cash * pos_cap / entry_price)
                cost   = shares * entry_price
            if shares < 1:
                continue

            cash -= cost

            sec = _SECTOR_OF.get(ticker.upper())
            if sec:
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

            positions[ticker] = {
                "ticker":            ticker,
                "entry_date":        next_day,
                "entry_price":       entry_price,
                "shares":            shares,
                "stop_loss":         stop_loss,
                "initial_stop_loss": stop_loss,   # preserved for risk metric
                "take_profit":       take_profit,
                "strategy":          strategy,
                "time_horizon":      time_horizon,
                "net_score":         net,
                "confidence":        confidence,
                "signals":           score.get("signals_triggered", []),
            }
            last_entry_day = next_day  # reset silence counter
            new_entries_today += 1
            _risk_pct_entry = (entry_price - stop_loss) / entry_price * 100 if entry_price > 0 else 0
            _top_sigs = score.get("signals_triggered", [])[:6]
            vprint(
                f"\n  [bold green]ENTRY[/bold green] {ticker} signal={today}→exec={next_day} | "
                f"[cyan]{strategy}[/cyan] conf={confidence:.2f} net={net} "
                f"regime={regime}({regime_mult:.1f}x) hv={is_high_vol}"
            )
            vprint(
                f"     {shares}sh @${entry_price:.2f} cost=${shares*entry_price:,.0f} | "
                f"stop=${stop_loss:.2f}({_risk_pct_entry:.1f}%risk) target=${take_profit:.2f} "
                f"max_hold={MAX_HOLD_DAYS.get(strategy, MAX_HOLD_DAYS.get(time_horizon, 7))}d"
            )
            vprint(
                f"     ADX={ind.get('adx',0):.1f} RSI={ind.get('rsi',0):.1f} "
                f"MACD={ind.get('macd_hist',0):.3f}↑{ind.get('macd_hist_prev1',0):.3f} "
                f"vol={ind.get('volume_ratio',0):.1f}x "
                f"bb%={ind.get('bb_pctb',0) or 0:.2f}"
            )
            if _top_sigs:
                vprint(f"     [dim]signals: {', '.join(_top_sigs)}[/dim]")

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
        "regime_days":  _regime_days,
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
        rd = results.get("regime_days", {})
        return {"total_trades": 0, "regime_days": rd}

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

    # (a) Avg realized risk per trade as % of capital
    # = mean over trades of ((entry - initial_stop)/entry × position_value / start_capital)
    risk_pcts = []
    for t in trades:
        ep   = t.get("entry_price", 0)
        isl  = t.get("initial_stop_loss") or t.get("stop_loss") or ep
        if ep > 0:
            stop_pct   = (ep - isl) / ep
            pos_val    = ep * t.get("shares", 0)
            risk_pcts.append(stop_pct * pos_val / start * 100)
    avg_realized_risk = round(np.mean(risk_pcts), 3) if risk_pcts else 0.0

    # (b) Avg % of capital deployed = 1 - mean(daily_cash / daily_equity)
    deployed_fracs = []
    for e in eq:
        eq_val   = e.get("equity", 0)
        cash_val = e.get("cash",   eq_val)  # fallback: old curve entries without cash field
        if eq_val > 0:
            deployed_fracs.append(1.0 - cash_val / eq_val)
    avg_deployed = round(np.mean(deployed_fracs) * 100, 1) if deployed_fracs else 0.0

    # Expectancy: avg_win*win_rate + avg_loss*(1-win_rate)
    wr  = len(wins) / len(trades) if trades else 0
    exp = round((np.mean(win_pcts) if win_pcts else 0) * wr
               + (np.mean(loss_pcts) if loss_pcts else 0) * (1 - wr), 3)

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
        "avg_realized_risk_pct": avg_realized_risk,
        "avg_deployed_pct":      avg_deployed,
        "expectancy_pct": exp,
        "by_strategy":    {s: {"trades": len(v), "total_pnl": round(sum(v), 2),
                               "win_rate": round(len([x for x in v if x > 0]) / len(v) * 100, 1)}
                           for s, v in by_strategy.items()},
        "monthly_pnl":    {k: round(v, 2) for k, v in sorted(monthly.items())},
        "regime_days":    results.get("regime_days", {}),
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
    t.add_row("Avg win",                f"[green]{_pct(stats.get('avg_win_pct', 0))}[/green]")
    t.add_row("Avg loss",               f"[red]{_pct(stats.get('avg_loss_pct', 0))}[/red]")
    exp = stats.get("expectancy_pct", 0)
    t.add_row("Expectancy",             f"[{'green' if exp >= 0 else 'red'}]{exp:+.3f}%[/{'green' if exp >= 0 else 'red'}]")
    t.add_row("Profit factor",          f"{stats.get('profit_factor', 0):.2f}")
    t.add_row("Max drawdown",           f"[red]{stats.get('max_drawdown_pct', 0):.2f}%[/red]")
    t.add_row("Sharpe ratio",           str(stats.get("sharpe", 0)))
    t.add_row("Avg hold (days)",        str(stats.get("avg_hold_days", 0)))
    t.add_row("Best trade",             f"[green]{_pct(stats.get('best_trade_pct', 0))}[/green]")
    t.add_row("Worst trade",            f"[red]{_pct(stats.get('worst_trade_pct', 0))}[/red]")
    t.add_row("Avg realized risk/trade",f"{stats.get('avg_realized_risk_pct', 0):.3f}% of capital")
    t.add_row("Avg capital deployed",   f"{stats.get('avg_deployed_pct', 0):.1f}%")
    rd = results.get("regime_days", stats.get("regime_days", {}))
    if rd:
        total_d = sum(rd.values()) or 1
        t.add_row("Regime days",
                  f"[green]uptrend={rd.get('confirmed_uptrend',0)}[/green]  "
                  f"[yellow]caution={rd.get('caution',0)}[/yellow]  "
                  f"[red]downtrend={rd.get('downtrend',0)}  "
                  f"shocked={rd.get('shocked',0)}[/red]  "
                  f"({100*rd.get('confirmed_uptrend',0)//total_d}% up)")
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
        writer = csv.DictWriter(f, fieldnames=["date", "equity", "cash"])
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
    # Change-group ablation flags (all default ON)
    parser.add_argument("--no-sizing-fix", action="store_true",
                        help="Change 1: revert to ATR-mult sizing (mismatched with stop)")
    parser.add_argument("--no-breakeven-fix", action="store_true",
                        help="Change 2: revert breakeven to 2.5%/-0.2% buffer")
    parser.add_argument("--no-trail-fix", action="store_true",
                        help="Change 3: revert trail giveback to 5%, ratchet to 50%/25%")
    parser.add_argument("--no-partial", action="store_true",
                        help="Change 3: disable partial profit-taking at +6%")
    parser.add_argument("--no-regime-deploy", action="store_true",
                        help="Change 4: revert downtrend regime_mult to 0.0 (no new longs)")
    parser.add_argument("--no-trend-follow", action="store_true",
                        help="Change 5: exclude trend_follow strategy entirely")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Print detailed entry/exit/rejection reasoning per trade")
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

    filter_bad         = not args.no_strategy_filter
    vol_cap            = not args.no_vol_cap
    relax              = not args.no_reentry_relax
    adx_filt           = not args.no_adx_filter
    fix_sizing         = not args.no_sizing_fix
    fix_breakeven      = not args.no_breakeven_fix
    fix_trail          = not args.no_trail_fix
    enable_partial     = not args.no_partial
    fix_regime_deploy  = not args.no_regime_deploy
    disable_trend_follow = args.no_trend_follow

    def _on(flag): return "[green]ON[/green]" if flag else "[red]OFF[/red]"
    console.print(f"\n[bold cyan]MoneyPrinter Backtester[/bold cyan]")
    console.print(f"Period:  {start} → {end}  ({(end-start).days} calendar days)")
    console.print(f"Capital: ${args.capital:,.0f}")
    console.print(f"Tickers: {len(tickers)} (including SPY for regime)")
    console.print(f"Min net: {args.min_net}")
    console.print(f"Core:    strat_filter={_on(filter_bad)}  vol_cap={_on(vol_cap)}  "
                  f"reentry_relax={_on(relax)}  adx_filter={_on(adx_filt)}")
    console.print(f"Fixes:   sizing={_on(fix_sizing)}  breakeven={_on(fix_breakeven)}  "
                  f"trail={_on(fix_trail)}  partial={_on(enable_partial)}  "
                  f"regime_deploy={_on(fix_regime_deploy)}  "
                  f"trend_follow={'[red]OFF[/red]' if disable_trend_follow else '[green]ON[/green]'}\n")

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
        fix_sizing=fix_sizing,
        fix_breakeven=fix_breakeven,
        fix_trail=fix_trail,
        enable_partial=enable_partial,
        fix_regime_deploy=fix_regime_deploy,
        disable_trend_follow=disable_trend_follow,
        verbose=getattr(args, "verbose", False),
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
