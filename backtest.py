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
import hashlib
import json
import logging
import os
import pickle
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
from bot.scorer import score_ticker, get_velocity_returns
from bot.strategies import classify_strategy

# ── Universe ─────────────────────────────────────────────────────────────────
from bot.discovery import UNIVERSE
from main import STATIC_TICKERS, SECTOR_GROUPS

ALL_TICKERS = list(dict.fromkeys(STATIC_TICKERS + UNIVERSE))  # deduped, order preserved

# Curated liquid subset for `--quick` smoke tests (~20 names, mix of sectors)
QUICK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "AMZN", "META", "GOOGL", "NFLX",
    "JPM", "BAC", "XOM", "CVX", "UNH", "LLY", "AVGO", "MRVL", "SMCI",
    "COIN", "PLTR",
]

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
MIN_CONFIDENCE     = 0.65        # mirrors live bot gate

# Hold-period rules. NOTE: the original code looked up the STRATEGY name in a
# dict keyed by HORIZON ({"scalp","swing","mixed"}), so every non-breakout
# position silently fell to the 5-day default. hold_mode="legacy" reproduces
# that measured behavior; "strategy" and "horizon" are the explicit rules.
HOLD_BY_STRATEGY = {
    "trend_follow":      7,
    "mean_reversion":    5,
    "news_momentum":     3,
    "breakdown":         7,
    "mixed":             5,
    "squeeze_breakout": 21,
}
HOLD_BY_HORIZON = {"scalp": 5, "swing": 20, "position": 45}


def _max_hold_for(pos: dict, hold_mode: str) -> int:
    if hold_mode == "legacy":
        return 5
    if hold_mode == "horizon":
        return HOLD_BY_HORIZON.get(pos.get("horizon", "swing"), 20)
    return HOLD_BY_STRATEGY.get(pos.get("strategy", "mixed"), 5)

# ── Breakout let-run constants ────────────────────────────────────────────────
CHANDELIER_ATR_MULT    = 3.0   # chandelier stop: highest_seen - 3 × ATR
BREAKOUT_SWING_LOOKBACK = 3    # prior bars used for structure stop
BREAKOUT_MAX_HOLD_RUN  = 21    # max hold days when breakout_let_run=True

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
#  Disk cache — bars + indicators are identical across exit-logic test runs,
#  so build once and reload in seconds on every subsequent run.
# ─────────────────────────────────────────────────────────────────────────────

CACHE_DIR = Path("backtest_cache")

# Bump this when indicator/scorer/velocity computation changes, to invalidate
# stale on-disk caches automatically.
CACHE_VERSION = "v1"


def _cache_key(tickers: list[str], start: date, end: date) -> str:
    raw = CACHE_VERSION + "|" + ",".join(sorted(tickers)) + f"|{start}|{end}"
    return hashlib.md5(raw.encode()).hexdigest()[:16]


def load_or_build_cache(
    tickers: list[str], start: date, end: date,
    refresh: bool = False, use_cache: bool = True,
) -> tuple[dict, dict, list[date]]:
    """
    Returns (bars_map, ind_cache, trading_days).
    Reuses an on-disk pickle when available; otherwise builds and saves it.
    """
    CACHE_DIR.mkdir(exist_ok=True)
    key  = _cache_key(tickers, start, end)
    path = CACHE_DIR / f"{key}.pkl"

    if use_cache and not refresh and path.exists():
        try:
            with open(path, "rb") as f:
                blob = pickle.load(f)
            built = blob.get("built_at", "?")
            console.print(f"[green]✓ Loaded cached bars + indicators[/green] "
                          f"[dim]({path.name}, built {built})[/dim]")
            console.print("[dim]  Pass --refresh-cache to rebuild after indicator/scorer changes.[/dim]\n")
            return blob["bars_map"], blob["ind_cache"], blob["trading_days"]
        except Exception as e:
            console.print(f"[yellow]Cache read failed ({e}); rebuilding…[/yellow]")

    # Build fresh
    bars_map     = load_all_bars(tickers, start, end)
    trading_days = get_trading_days(bars_map, start, end)
    if not trading_days:
        return bars_map, {}, trading_days

    sim_tickers = [t for t in tickers if t != "SPY"]
    ind_cache   = _build_indicator_cache(sim_tickers, bars_map, trading_days)

    if use_cache:
        try:
            with open(path, "wb") as f:
                pickle.dump({
                    "bars_map":      bars_map,
                    "ind_cache":     ind_cache,
                    "trading_days":  trading_days,
                    "built_at":      datetime.now().isoformat(timespec="seconds"),
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            console.print(f"[green]✓ Cache saved → {path}[/green] "
                          f"[dim](reused automatically on next run)[/dim]\n")
        except Exception as e:
            console.print(f"[yellow]Cache write failed ({e}); continuing without saving.[/yellow]")

    return bars_map, ind_cache, trading_days


# ─────────────────────────────────────────────────────────────────────────────
#  Indicator computation (historical, no API calls)
# ─────────────────────────────────────────────────────────────────────────────

def _fast_slice(df: pd.DataFrame, as_of: date) -> pd.DataFrame:
    """Slice df to rows with index date <= as_of using binary search (O(log n))."""
    cutoff = pd.Timestamp(as_of) + pd.Timedelta(days=1)
    pos = df.index.searchsorted(cutoff, side="left")
    return df.iloc[:pos]


def _build_ticker_ind_cache(
    ticker: str, df: pd.DataFrame, trading_days: list[date]
) -> tuple[str, dict[date, dict]]:
    """Compute indicators for all trading days for one ticker. Runs in a thread."""
    ind_by_day: dict[date, dict] = {}
    prev_ind: dict = {}
    for day in trading_days:
        sliced = _fast_slice(df, day)
        if len(sliced) < 30:
            ind = {"ticker": ticker, "error": "insufficient_history"}
        else:
            try:
                ind = compute_indicators_from_df(ticker, sliced, intraday=None, realtime_price=False)
            except Exception:
                ind = {"ticker": ticker, "error": "compute_failed"}
        if not ind.get("error"):
            # Inject historical velocity returns so score_ticker skips live API fetch
            vel = get_velocity_returns(ticker, sliced)
            ind.update(vel)
            triggers = compute_entry_triggers(ind, prev_ind)
            ind["entry_triggers"] = triggers
            prev_ind = {k: v for k, v in ind.items() if k != "entry_triggers"}
        ind_by_day[day] = ind
    return ticker, ind_by_day


def _build_indicator_cache(
    tickers: list[str],
    bars_map: dict[str, pd.DataFrame],
    trading_days: list[date],
    max_workers: int = 8,
) -> dict[str, dict[date, dict]]:
    """Pre-compute all (ticker, day) indicators in parallel. Returns {ticker: {day: ind}}."""
    work = [(t, bars_map[t]) for t in tickers if bars_map.get(t) is not None]
    cache: dict[str, dict[date, dict]] = {}
    n_work = len(work)
    n_comp = n_work * len(trading_days)
    console.print(f"[cyan]Pre-computing indicators: {n_work} tickers × "
                  f"{len(trading_days)} days = {n_comp:,} computations "
                  f"({min(max_workers, n_work)} threads)…[/cyan]")
    done = 0
    with ThreadPoolExecutor(max_workers=min(max_workers, n_work)) as ex:
        futs = {ex.submit(_build_ticker_ind_cache, t, df, trading_days): t
                for t, df in work}
        for f in as_completed(futs):
            tk, ibd = f.result()
            cache[tk] = ibd
            done += 1
            if done % 40 == 0 or done == n_work:
                console.print(f"  [dim]{done}/{n_work} tickers done[/dim]")
    console.print(f"[green]Cache ready[/green]\n")
    return cache


def compute_ind_for_day(ticker: str, df: pd.DataFrame, as_of: date) -> dict:
    """Slice df to [start, as_of] and compute indicators — no lookahead."""
    sliced = _fast_slice(df, as_of)
    if len(sliced) < 30:
        return {"ticker": ticker, "error": "insufficient_history"}
    return compute_indicators_from_df(ticker, sliced, intraday=None, realtime_price=False)


# ─────────────────────────────────────────────────────────────────────────────
#  Position sizing (mirrors bot/risk.py)
# ─────────────────────────────────────────────────────────────────────────────

def size_position(portfolio_value: float, confidence: float,
                  atr: float, price: float,
                  risk_pct: float = RISK_PCT,
                  max_pos_pct: float = MAX_POSITION_PCT) -> int:
    if price <= 0 or atr <= 0:
        return 0
    dollar_risk = portfolio_value * risk_pct * confidence
    shares      = floor(dollar_risk / (atr * ATR_STOP_MULT))
    max_shares  = floor(portfolio_value * max_pos_pct / price)
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
    breakout_let_run: bool = True,
    max_open: int = MAX_OPEN_POSITIONS,
    max_pos_pct: float = MAX_POSITION_PCT,
    risk_pct: float = RISK_PCT,
    hold_mode: str = "strategy",          # "strategy" (live default) | "legacy" | "horizon"
    letrun_strats: tuple = ("breakout",), # strategies that get chandelier let-run
    quiet: bool = False,
    _bars_map:  dict | None = None,  # pre-loaded bars — avoids double fetch for A/B
    _ind_cache: dict | None = None,  # pre-built cache — avoids double compute for A/B
    _score_cache: dict | None = None, # shared {(ticker, day): classified score} across variants
) -> dict:
    """Full day-by-day simulation. Returns results dict with trades + equity curve."""
    bars_map     = _bars_map or load_all_bars(tickers, start, end)
    trading_days = get_trading_days(bars_map, start, end)

    if not trading_days:
        console.print("[red]No trading days found — check date range and API credentials[/red]")
        return {}

    if not quiet:
        console.print(f"[cyan]Simulating {len(trading_days)} trading days "
                      f"({trading_days[0]} → {trading_days[-1]})[/cyan]")

    # ── Pre-compute all (ticker, day) indicators in parallel ─────────────────
    if _ind_cache is not None:
        ind_cache = _ind_cache
        if not quiet:
            console.print("[dim]Using shared indicator cache[/dim]")
    else:
        sim_tickers = [t for t in tickers if t != "SPY"]
        ind_cache   = _build_indicator_cache(sim_tickers, bars_map, trading_days)

    # ── State ─────────────────────────────────────────────────────────────────
    cash         = starting_capital
    positions:   dict[str, dict] = {}
    all_trades:  list[dict]      = []
    equity_curve: list[dict]     = []

    spy_bars = bars_map.get("SPY")

    def _spy_regime(as_of: date) -> str:
        if spy_bars is None:
            return "bull"
        try:
            sl = _fast_slice(spy_bars, as_of)
            if len(sl) < 50:
                return "bull"
            price = float(sl["Close"].iloc[-1])
            ema50 = float(sl["Close"].ewm(span=50, adjust=False).mean().iloc[-1])
            return "bull" if price >= ema50 else "caution"
        except Exception:
            return "bull"

    # ── Day loop ──────────────────────────────────────────────────────────────
    for day_idx, today in enumerate(trading_days):

        today_ts = pd.Timestamp(today)
        next_ts  = today_ts + pd.Timedelta(days=1)

        # Portfolio value = cash + mark-to-market open positions
        mkt_value = cash
        for ticker, pos in positions.items():
            df = bars_map.get(ticker)
            if df is None:
                continue
            sl = _fast_slice(df, today)
            if not sl.empty:
                mkt_value += pos["shares"] * float(sl["Close"].iloc[-1])

        equity_curve.append({"date": today, "equity": round(mkt_value, 2)})

        regime = _spy_regime(today)
        macro  = {
            "vix": 18.0,
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
            # Today's bar — O(log n) binary search
            p0 = df.index.searchsorted(today_ts, side="left")
            p1 = df.index.searchsorted(next_ts,  side="left")
            today_bars = df.iloc[p0:p1]
            if today_bars.empty:
                continue

            day_high  = float(today_bars["High"].iloc[-1])
            day_low   = float(today_bars["Low"].iloc[-1])
            day_close = float(today_bars["Close"].iloc[-1])
            day_open  = float(today_bars["Open"].iloc[-1])

            stop     = pos["stop_loss"]
            target   = pos["take_profit"]
            entry    = pos["entry_price"]
            shares   = pos["shares"]
            max_hold = _max_hold_for(pos, hold_mode)
            age_days = (today - pos["entry_date"]).days

            exit_price  = None
            exit_reason = None

            # ── Breakout let-run: dynamic stop from PRIOR bars (no look-ahead) ──
            prior_bars_brk = None
            if breakout_let_run and pos.get("strategy") in letrun_strats:
                max_hold       = max(max_hold, BREAKOUT_MAX_HOLD_RUN)
                prior_bars_brk = _fast_slice(df, today - timedelta(days=1)).tail(
                    BREAKOUT_SWING_LOOKBACK
                )
                if len(prior_bars_brk) >= 1:
                    prior_close_brk = float(prior_bars_brk["Close"].iloc[-1])
                    structure_low   = float(prior_bars_brk["Low"].min()) * 0.995
                    structure_stop  = min(structure_low, prior_close_brk * 0.999)
                    pos_highest     = pos.get("highest", entry)
                    pos_atr         = pos.get("atr") or (entry * 0.02)
                    chandelier      = pos_highest - CHANDELIER_ATR_MULT * pos_atr
                    dyn_stop        = max(stop, structure_stop, chandelier)
                    # Ratchet: only up, never above highest (safety)
                    if dyn_stop > stop and dyn_stop < pos_highest:
                        stop             = dyn_stop
                        pos["stop_loss"] = stop

            # Stop hit — worst case: gap down through stop uses day open
            if day_low <= stop:
                exit_price  = max(min(stop, day_open), day_low)
                exit_reason = "stop"
            # Target hit
            elif target and day_high >= target:
                exit_price  = min(target, day_high)
                exit_reason = "target"
            # Time exit
            elif age_days >= max_hold:
                if (breakout_let_run and pos.get("strategy") in letrun_strats
                        and prior_bars_brk is not None and len(prior_bars_brk) >= 1):
                    brk_lvl         = pos.get("breakout_level", 0.0)
                    prior_close_brk = float(prior_bars_brk["Close"].iloc[-1])
                    if brk_lvl > 0 and prior_close_brk >= brk_lvl:
                        pass  # still above pivot — suppress time exit
                    else:
                        exit_price  = day_close
                        exit_reason = "time_exit"
                else:
                    exit_price  = day_close
                    exit_reason = "time_exit"

            # Track highest for tomorrow's chandelier (no look-ahead)
            if exit_price is None and breakout_let_run and pos.get("strategy") in letrun_strats:
                pos["highest"] = max(pos.get("highest", entry), day_high)

            if exit_price is not None:
                pnl_dollar = (exit_price - entry) * shares
                pnl_pct    = (exit_price - entry) / entry * 100
                cash      += exit_price * shares
                all_trades.append({
                    "ticker":      ticker,
                    "entry_date":  pos["entry_date"].isoformat(),
                    "exit_date":   today.isoformat(),
                    "entry_price": round(entry, 2),
                    "exit_price":  round(exit_price, 2),
                    "shares":      shares,
                    "stop_loss":   round(stop, 2),
                    "take_profit": round(target, 2) if target else None,
                    "pnl_dollar":  round(pnl_dollar, 2),
                    "pnl_pct":     round(pnl_pct, 2),
                    "hold_days":   age_days,
                    "exit_reason": exit_reason,
                    "strategy":    pos.get("strategy", "unknown"),
                    "net_score":   pos.get("net_score", 0),
                    "confidence":  pos.get("confidence", 0),
                    "signals":     pos.get("signals", []),
                })
                closed_tickers.append(ticker)

        for t in closed_tickers:
            del positions[t]

        # ── 2. Generate signals from pre-computed indicator cache ─────────────
        if len(positions) >= max_open:
            continue

        sector_counts: dict[str, int] = {}
        for held in positions:
            sec = _SECTOR_OF.get(held.upper())
            if sec:
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

        new_signals: list[dict] = []

        for ticker in tickers:
            if ticker in positions or ticker == "SPY":
                continue

            # Cache lookup — O(1), no indicator recomputation
            ind = ind_cache.get(ticker, {}).get(today)
            if not ind or ind.get("error"):
                continue

            # Scoring is variant-independent (same indicators/regime/news), so
            # experiment runs share one score cache — only gates/exits differ.
            _sc_key = (ticker, today)
            if _score_cache is not None and _sc_key in _score_cache:
                score = _score_cache[_sc_key]
                if score is None:
                    continue
            else:
                try:
                    score = score_ticker(ticker, ind, {}, macro)
                    score = classify_strategy(score, ind)
                except Exception as e:
                    logger.debug(f"[backtest] scorer failed {ticker} on {today}: {e}")
                    score = None
                if _score_cache is not None:
                    _score_cache[_sc_key] = score
                if score is None:
                    continue

            action     = score.get("action", "hold")
            net        = score.get("net_score", 0)
            confidence = score.get("confidence", 0.0)

            if action != "buy":
                continue
            if net < min_net:
                continue
            if confidence < MIN_CONFIDENCE or score.get("strategy") == "mixed":
                continue
            if macro["bearish_market"]:
                continue

            sec = _SECTOR_OF.get(ticker.upper())
            if sec and sector_counts.get(sec, 0) >= MAX_PER_SECTOR:
                continue

            new_signals.append(score)

        new_signals.sort(key=lambda s: s.get("confidence", 0), reverse=True)
        slots = max_open - len(positions)

        for score in new_signals[:slots]:
            ticker      = score["ticker"]
            ind         = ind_cache.get(ticker, {}).get(today, {})
            atr         = score.get("atr") or ind.get("atr") or 0
            entry_ref   = score.get("entry_price") or ind.get("current_price") or 0
            stop_loss   = score.get("stop_loss") or 0
            take_profit = score.get("take_profit") or 0
            confidence  = score.get("confidence", 0.65)
            net         = score.get("net_score", 0)
            strategy    = score.get("strategy", "swing")

            if entry_ref <= 0 or atr <= 0:
                continue

            # Entry at NEXT DAY's open
            if day_idx + 1 >= len(trading_days):
                continue
            next_day = trading_days[day_idx + 1]
            df = bars_map.get(ticker)
            if df is None:
                continue
            nts  = pd.Timestamp(next_day)
            np0  = df.index.searchsorted(nts,                           side="left")
            np1  = df.index.searchsorted(nts + pd.Timedelta(days=1),    side="left")
            next_bars = df.iloc[np0:np1]
            if next_bars.empty:
                continue

            entry_price = float(next_bars["Open"].iloc[0])
            if entry_price <= 0:
                continue

            shares = size_position(
                portfolio_value=cash + sum(
                    p["shares"] * entry_price for p in positions.values()
                ),
                confidence=confidence,
                atr=atr,
                price=entry_price,
                risk_pct=risk_pct,
                max_pos_pct=max_pos_pct,
            )
            if shares < 1:
                continue

            cost = shares * entry_price
            if cost > cash:
                shares = floor(cash * max_pos_pct / entry_price)
                cost   = shares * entry_price
            if shares < 1:
                continue

            if stop_loss <= 0 or stop_loss >= entry_price:
                stop_loss = round(entry_price - atr * ATR_STOP_MULT, 2)
            if take_profit <= 0 or take_profit <= entry_price:
                risk        = entry_price - stop_loss
                take_profit = round(entry_price + risk * 2.5, 2)

            cash -= cost

            sec = _SECTOR_OF.get(ticker.upper())
            if sec:
                sector_counts[sec] = sector_counts.get(sec, 0) + 1

            if strategy == "breakout":
                r1_val  = float(ind.get("R1") or 0)
                w52_val = float(ind.get("wk52_high") or 0)
                if w52_val > 0 and entry_price >= w52_val * 0.99:
                    brk_lvl_entry = round(w52_val, 2)
                elif r1_val > 0:
                    brk_lvl_entry = round(r1_val, 2)
                else:
                    brk_lvl_entry = round(entry_price * 0.985, 2)
            else:
                brk_lvl_entry = 0.0

            positions[ticker] = {
                "ticker":         ticker,
                "entry_date":     next_day,
                "entry_price":    entry_price,
                "shares":         shares,
                "stop_loss":      stop_loss,
                "take_profit":    take_profit,
                "strategy":       strategy,
                "horizon":        score.get("time_horizon", "swing"),
                "net_score":      net,
                "confidence":     confidence,
                "signals":        score.get("signals_triggered", []),
                "breakout_level": brk_lvl_entry,
                "atr":            atr,
                "highest":        entry_price,
            }

    # ── Close any remaining open positions at end date ────────────────────────
    final_day = trading_days[-1] if trading_days else end
    for ticker, pos in list(positions.items()):
        df = bars_map.get(ticker)
        exit_price = pos["entry_price"]
        if df is not None:
            sl = _fast_slice(df, final_day)
            if not sl.empty:
                exit_price = float(sl["Close"].iloc[-1])
        pnl_dollar = (exit_price - pos["entry_price"]) * pos["shares"]
        pnl_pct    = (exit_price - pos["entry_price"]) / pos["entry_price"] * 100
        cash += exit_price * pos["shares"]
        all_trades.append({
            "ticker":      ticker,
            "entry_date":  pos["entry_date"].isoformat(),
            "exit_date":   final_day.isoformat(),
            "entry_price": round(pos["entry_price"], 2),
            "exit_price":  round(exit_price, 2),
            "shares":      pos["shares"],
            "stop_loss":   round(pos["stop_loss"], 2),
            "take_profit": round(pos["take_profit"], 2),
            "pnl_dollar":  round(pnl_dollar, 2),
            "pnl_pct":     round(pnl_pct, 2),
            "hold_days":   (final_day - pos["entry_date"]).days,
            "exit_reason": "end_of_backtest",
            "strategy":    pos.get("strategy", "unknown"),
            "net_score":   pos.get("net_score", 0),
            "confidence":  pos.get("confidence", 0),
            "signals":     pos.get("signals", []),
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


def compute_strategy_stats(trades: list[dict], strategy_filter: str | None = None) -> dict:
    """Stats for a trade list, optionally filtered to one strategy."""
    if strategy_filter:
        trades = [t for t in trades if t.get("strategy") == strategy_filter]
    if not trades:
        return {
            "total_trades": 0, "win_rate": 0.0,
            "avg_win_pct": 0.0, "avg_loss_pct": 0.0,
            "profit_factor": 0.0, "expectancy": 0.0,
            "avg_win_loss_ratio": 0.0, "total_pnl": 0.0,
        }
    pnls      = [t["pnl_dollar"] for t in trades]
    wins      = [p for p in pnls if p > 0]
    losses    = [p for p in pnls if p <= 0]
    win_pcts  = [t["pnl_pct"] for t in trades if t["pnl_dollar"] > 0]
    loss_pcts = [t["pnl_pct"] for t in trades if t["pnl_dollar"] <= 0]
    n         = len(trades)
    wr        = len(wins) / n
    avg_win_d = float(np.mean(wins))   if wins   else 0.0
    avg_los_d = abs(float(np.mean(losses))) if losses else 0.0
    pf        = abs(sum(wins) / sum(losses)) if losses else float("inf")
    exp       = (wr * avg_win_d) - ((1 - wr) * avg_los_d)
    wl_ratio  = avg_win_d / avg_los_d if avg_los_d > 0 else float("inf")
    return {
        "total_trades":       n,
        "win_rate":           wr * 100,
        "avg_win_pct":        float(np.mean(win_pcts))  if win_pcts  else 0.0,
        "avg_loss_pct":       float(np.mean(loss_pcts)) if loss_pcts else 0.0,
        "profit_factor":      pf,
        "expectancy":         exp,
        "avg_win_loss_ratio": wl_ratio,
        "total_pnl":          sum(pnls),
    }


def print_breakout_comparison(res_on: dict, res_off: dict) -> None:
    """Side-by-side A/B table: breakout_let_run=True vs False."""
    brk_on   = compute_strategy_stats(res_on.get("trades",  []), "breakout")
    brk_off  = compute_strategy_stats(res_off.get("trades", []), "breakout")
    book_on  = compute_strategy_stats(res_on.get("trades",  []))
    book_off = compute_strategy_stats(res_off.get("trades", []))

    def _f(v, is_pct=False, is_dollar=False, is_ratio=False) -> str:
        if v == float("inf"):
            return "∞"
        if is_dollar:
            return f"${v:,.0f}"
        if is_pct:
            return f"{v:+.1f}%"
        if is_ratio:
            return f"{v:.2f}×"
        return f"{v:.2f}"

    console.print()
    console.print(Panel(
        "[bold]Breakout Let-Run: A/B Comparison[/bold]\n"
        "Judge by expectancy, profit factor, and avg W/L ratio — NOT win rate.",
        style="bold yellow",
        expand=False,
    ))

    tbl = Table(box=box.SIMPLE_HEAVY, show_header=True)
    tbl.add_column("Metric",         style="bold", min_width=22)
    tbl.add_column("Let-Run  ON",    justify="right", style="green", min_width=14)
    tbl.add_column("Baseline OFF",   justify="right", style="red",   min_width=14)

    tbl.add_row("[bold dim]── BREAKOUT trades ──[/bold dim]", "", "")
    tbl.add_row("Trade count",
                str(brk_on["total_trades"]),
                str(brk_off["total_trades"]))
    tbl.add_row("Win rate",
                f"{brk_on['win_rate']:.1f}%",
                f"{brk_off['win_rate']:.1f}%")
    tbl.add_row("Avg win %",
                _f(brk_on["avg_win_pct"],  is_pct=True),
                _f(brk_off["avg_win_pct"], is_pct=True))
    tbl.add_row("Avg loss %",
                _f(brk_on["avg_loss_pct"],  is_pct=True),
                _f(brk_off["avg_loss_pct"], is_pct=True))
    tbl.add_row("Avg W/L ratio",
                _f(brk_on["avg_win_loss_ratio"],  is_ratio=True),
                _f(brk_off["avg_win_loss_ratio"], is_ratio=True))
    tbl.add_row("Profit factor",
                _f(brk_on["profit_factor"]),
                _f(brk_off["profit_factor"]))
    tbl.add_row("Expectancy $/trade",
                _f(brk_on["expectancy"],  is_dollar=True),
                _f(brk_off["expectancy"], is_dollar=True))
    tbl.add_row("Total P&L",
                _f(brk_on["total_pnl"],  is_dollar=True),
                _f(brk_off["total_pnl"], is_dollar=True))

    tbl.add_row("[bold dim]── WHOLE BOOK ──[/bold dim]", "", "")
    tbl.add_row("Trade count",
                str(book_on["total_trades"]),
                str(book_off["total_trades"]))
    tbl.add_row("Win rate",
                f"{book_on['win_rate']:.1f}%",
                f"{book_off['win_rate']:.1f}%")
    tbl.add_row("Profit factor",
                _f(book_on["profit_factor"]),
                _f(book_off["profit_factor"]))
    tbl.add_row("Expectancy $/trade",
                _f(book_on["expectancy"],  is_dollar=True),
                _f(book_off["expectancy"], is_dollar=True))
    tbl.add_row("Total P&L",
                _f(book_on["total_pnl"],  is_dollar=True),
                _f(book_off["total_pnl"], is_dollar=True))

    console.print(tbl)
    console.print("[dim]SUCCESS criterion: expectancy rises and/or profit factor rises, "
                  "even if win rate falls.[/dim]\n")


def print_report(results: dict, stats: dict, args, label: str = "") -> None:
    trades = results.get("trades", [])
    start  = results["start_equity"]
    end_eq = results["final_equity"]
    ret    = stats.get("total_return_pct", 0)

    ret_color = "green" if ret >= 0 else "red"

    title_extra = f"  [{label}]" if label else ""
    console.print()
    console.print(Panel(
        f"[bold]MoneyPrinter Backtest Report{title_extra}[/bold]\n"
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
#  Experiment matrix — measured A/B of candidate strategy improvements.
#  All variants share one bars+indicator cache, so each extra variant costs
#  only simulation time (seconds), not data/compute time.
# ─────────────────────────────────────────────────────────────────────────────

EXPERIMENTS: dict[str, dict] = {
    # The pre-fix behavior (5-day holds everywhere due to the
    # strategy/horizon key mismatch, breakout let-run 21d)
    "baseline_legacy": {"hold_mode": "legacy"},
    # Explicit per-strategy holds — measured winner across both windows,
    # now the live + backtest default
    "hold_strategy":   {"hold_mode": "strategy"},
    # Horizon-based holds like main.py's walk-forward (swing=20d)
    "hold_horizon":    {"hold_mode": "horizon"},
    # Entry quality: live bot uses 65; is 60 or 70 better?
    "min_net_65":      {"min_net": 65},
    "min_net_70":      {"min_net": 70},
    # Capital utilization: 5 slots × 10% caps deployment at 50%
    "slots8_pos15":    {"max_open": 8, "max_pos_pct": 0.15},
    "risk3":           {"risk_pct": 0.03},
    # Extend the chandelier let-run (the best-performing exit) beyond breakout
    "letrun_trend":    {"letrun_strats": ("breakout", "squeeze_breakout", "trend_follow")},
    # Combined growth candidate
    "combo_growth":    {"max_open": 8, "max_pos_pct": 0.15,
                        "letrun_strats": ("breakout", "squeeze_breakout", "trend_follow")},
}


def run_experiments(tickers: list[str], windows: list[tuple[date, date]],
                    capital: float, refresh: bool, use_cache: bool) -> None:
    full_start = min(w[0] for w in windows)
    full_end   = max(w[1] for w in windows)

    bars_map, ind_cache, trading_days = load_or_build_cache(
        tickers, full_start, full_end, refresh=refresh, use_cache=use_cache)
    if not trading_days:
        console.print("[red]No trading days — check credentials/date range[/red]")
        sys.exit(1)

    out: dict = {"windows": {}, "experiments": {k: v for k, v in EXPERIMENTS.items()}}

    for (ws, we) in windows:
        wkey = f"{ws}→{we}"
        console.print(Panel(f"[bold]Experiment window {wkey}[/bold]", style="bold cyan", expand=False))

        # SPY buy-and-hold benchmark for the window
        spy_ret = None
        spy_df = bars_map.get("SPY")
        if spy_df is not None:
            s0 = _fast_slice(spy_df, ws)
            s1 = _fast_slice(spy_df, we)
            if len(s0) and len(s1):
                spy_ret = (float(s1["Close"].iloc[-1]) - float(s0["Close"].iloc[-1])) \
                          / float(s0["Close"].iloc[-1]) * 100

        tbl = Table(box=box.SIMPLE_HEAVY)
        for col in ["Variant", "Return %", "Trades", "Win %", "PF",
                    "Max DD %", "Sharpe", "Avg hold"]:
            tbl.add_column(col, justify="right" if col != "Variant" else "left")

        score_cache: dict = {}   # shared across variants within this window
        win_rows = {}
        for name, cfg in EXPERIMENTS.items():
            try:
                res = run_backtest(
                    tickers=tickers, start=ws, end=we,
                    starting_capital=capital, quiet=True,
                    _bars_map=bars_map, _ind_cache=ind_cache,
                    _score_cache=score_cache, **cfg,
                )
                st = compute_stats(res)
                console.print(f"[dim]  {name} done — {st.get('total_trades',0)} trades, "
                              f"{st.get('total_return_pct',0):+.2f}%[/dim]")
            except Exception as e:
                console.print(f"[red]{name} failed: {e}[/red]")
                continue
            ret = st.get("total_return_pct", 0.0)
            color = "green" if ret >= 0 else "red"
            tbl.add_row(
                name,
                f"[{color}]{ret:+.2f}%[/{color}]",
                str(st.get("total_trades", 0)),
                f"{st.get('win_rate', 0):.1f}",
                f"{st.get('profit_factor', 0):.2f}" if st.get("profit_factor") != float("inf") else "∞",
                f"{st.get('max_drawdown_pct', 0):.1f}",
                f"{st.get('sharpe', 0)}",
                f"{st.get('avg_hold_days', 0)}d",
            )
            win_rows[name] = {
                "return_pct":   round(ret, 2),
                "trades":       st.get("total_trades", 0),
                "win_rate":     round(st.get("win_rate", 0), 1),
                "profit_factor": (round(st["profit_factor"], 2)
                                  if st.get("profit_factor") not in (None, float("inf")) else None),
                "max_dd_pct":   round(st.get("max_drawdown_pct", 0), 2),
                "sharpe":       st.get("sharpe", 0),
                "avg_hold":     st.get("avg_hold_days", 0),
                "by_strategy":  st.get("by_strategy", {}),
            }

        console.print(tbl)
        if spy_ret is not None:
            console.print(f"[dim]SPY buy-and-hold over window: {spy_ret:+.2f}%[/dim]\n")
        out["windows"][wkey] = {"spy_return_pct": round(spy_ret, 2) if spy_ret is not None else None,
                                "results": win_rows}

    out_dir = Path("backtest_output")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / "experiments.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    console.print(f"[green]Experiment matrix exported → {path}[/green]")


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
    parser.add_argument("--no-breakout-run", action="store_true",
                        help="Disable breakout let-run mode (wider stops / 21-day hold)")
    parser.add_argument("--quick", action="store_true",
                        help="Fast smoke test: ~20 liquid tickers instead of full universe")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="Rebuild the bars+indicator cache (use after changing indicator/scorer code)")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable the on-disk bars+indicator cache entirely")
    parser.add_argument("--experiments", action="store_true",
                        help="Run the variant experiment matrix instead of a single backtest")
    parser.add_argument("--windows", default="2024-06-01:2025-06-01,2025-06-01:2026-06-01",
                        help="Comma-separated start:end windows for --experiments")
    args = parser.parse_args()

    start = date.fromisoformat(args.start)
    end   = date.fromisoformat(args.end)

    if end <= start:
        console.print("[red]--end must be after --start[/red]")
        sys.exit(1)

    if args.quick:
        # Curated liquid subset for fast smoke tests
        tickers = [t for t in QUICK_TICKERS if t in ALL_TICKERS] or QUICK_TICKERS
    else:
        tickers = args.tickers or ALL_TICKERS
    # Always include SPY for regime detection
    if "SPY" not in tickers:
        tickers = ["SPY"] + tickers

    breakout_let_run = not args.no_breakout_run

    if args.experiments:
        windows = []
        for w in args.windows.split(","):
            ws, we = w.strip().split(":")
            windows.append((date.fromisoformat(ws), date.fromisoformat(we)))
        console.print(f"\n[bold cyan]MoneyPrinter Experiment Matrix[/bold cyan]")
        console.print(f"Windows: {[(str(a), str(b)) for a, b in windows]}")
        console.print(f"Tickers: {len(tickers)}  |  Variants: {len(EXPERIMENTS)}\n")
        run_experiments(tickers, windows, args.capital,
                        refresh=args.refresh_cache, use_cache=not args.no_cache)
        return

    console.print(f"\n[bold cyan]MoneyPrinter Backtester[/bold cyan]")
    console.print(f"Period:  {start} → {end}  ({(end-start).days} calendar days)")
    console.print(f"Capital: ${args.capital:,.0f}")
    qtag = "  [yellow](--quick)[/yellow]" if args.quick else ""
    console.print(f"Tickers: {len(tickers)} (including SPY for regime){qtag}")
    console.print(f"Min net: {args.min_net}")
    brk_status = "[green]ON[/green]" if breakout_let_run else "[red]OFF[/red]"
    console.print(f"Breakout let-run: {brk_status}")
    cache_status = "[red]disabled[/red]" if args.no_cache else (
        "[yellow]refreshing[/yellow]" if args.refresh_cache else "[green]enabled[/green]")
    console.print(f"Disk cache: {cache_status}\n")

    # ── Load (or build) bars + indicator cache once, shared across both A/B runs ─
    bars_map, ind_cache, trading_days = load_or_build_cache(
        tickers, start, end,
        refresh=args.refresh_cache,
        use_cache=not args.no_cache,
    )
    if not trading_days:
        console.print("[red]No trading days found — check date range and API credentials[/red]")
        sys.exit(1)

    results = run_backtest(
        tickers=tickers,
        start=start,
        end=end,
        starting_capital=args.capital,
        min_net=args.min_net,
        breakout_let_run=breakout_let_run,
        _bars_map=bars_map,
        _ind_cache=ind_cache,
    )

    if not results:
        sys.exit(1)

    # ── A/B comparison when let-run is ON (reuses shared cache — no extra fetch) ─
    if breakout_let_run:
        console.print("\n[dim]Running baseline (let-run OFF) for A/B comparison…[/dim]")
        results_baseline = run_backtest(
            tickers=tickers,
            start=start,
            end=end,
            starting_capital=args.capital,
            min_net=args.min_net,
            breakout_let_run=False,
            _bars_map=bars_map,
            _ind_cache=ind_cache,
        )
        if results_baseline:
            print_breakout_comparison(results, results_baseline)

    stats = compute_stats(results)
    label = "breakout let-run ON" if breakout_let_run else "breakout let-run OFF"
    print_report(results, stats, args, label=label)

    # ── Export ────────────────────────────────────────────────────────────────
    out_dir = Path("backtest_output")
    out_dir.mkdir(exist_ok=True)
    suffix  = "" if breakout_let_run else "_no_letrun"
    prefix  = f"{out_dir}/{args.out}{suffix}_{args.start}_{args.end}"
    save_csv(results["trades"], Path(f"{prefix}_trades.csv"))
    save_equity_csv(results["equity_curve"], Path(f"{prefix}_equity.csv"))

    # JSON summary
    summary_path = Path(f"{prefix}_summary.json")
    with open(summary_path, "w") as f:
        json.dump({
            "args": {**vars(args), "breakout_let_run": breakout_let_run},
            "stats": {k: v for k, v in stats.items()
                      if not isinstance(v, (dict,))},
            "by_strategy": stats.get("by_strategy", {}),
            "monthly_pnl": stats.get("monthly_pnl", {}),
        }, f, indent=2)
    console.print(f"[green]Summary exported → {summary_path}[/green]")


if __name__ == "__main__":
    main()
