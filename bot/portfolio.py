"""Portfolio state — pulls live data from Alpaca and cross-references trades DB."""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from bot.logger import get_open_trades, get_trades_today, update_trade_exit, update_trade_stop
from bot.risk import record_trade_pnl

# Time-based exit rules keyed by STRATEGY (mirrors backtest MAX_HOLD_DAYS lookup).
# Breakout positions get 21 days so the chandelier stop can do its job.
MAX_HOLD_DAYS = {
    "scalp":             2,
    "swing":             7,
    "mixed":             5,
    "breakout":         21,
    "squeeze_breakout": 21,
    "trend_follow":      7,
    "mean_reversion":    5,
    "news_momentum":     3,
    "breakdown":         7,
}

logger = logging.getLogger(__name__)



def get_open_positions(alpaca_client=None) -> list[dict]:
    """
    Return open positions from the trades DB, enriched with current price
    from Alpaca if client is provided.
    """
    db_positions = get_open_trades()
    if not alpaca_client:
        return db_positions

    try:
        from bot.trader import get_positions
        live = {p["symbol"]: p for p in get_positions(alpaca_client)}
    except Exception as e:
        logger.warning(f"[portfolio] Could not fetch live positions: {e}")
        live = {}

    enriched = []
    for pos in db_positions:
        ticker = pos["ticker"]
        if ticker in live:
            lp = live[ticker]
            pos["current_price"]  = lp.get("current_price")
            pos["unrealized_pnl"] = lp.get("unrealized_pl")
            pos["unrealized_pct"] = lp.get("unrealized_plpc")
        enriched.append(pos)
    return enriched


def check_stops(alpaca_client) -> list[dict]:
    """
    Compare open positions to their stop loss levels.
    Returns list of positions that have breached their stop loss.
    """
    positions = get_open_positions(alpaca_client)
    breached = []
    for pos in positions:
        sl = pos.get("stop_loss")
        cp = pos.get("current_price") or pos.get("entry_price")
        action = pos.get("action", "buy")
        if sl is None or cp is None:
            continue
        if action == "buy" and float(cp) <= float(sl):
            logger.warning(f"[portfolio] STOP HIT: {pos['ticker']} price={cp} sl={sl}")
            breached.append(pos)
        elif action in ("short", "sell") and float(cp) >= float(sl):
            logger.warning(f"[portfolio] STOP HIT (short): {pos['ticker']} price={cp} sl={sl}")
            breached.append(pos)
    return breached


def check_targets(alpaca_client) -> list[dict]:
    """
    Compare open positions to their take profit targets.
    Returns list of positions that have hit their target.
    """
    positions = get_open_positions(alpaca_client)
    targets_hit = []
    for pos in positions:
        tp = pos.get("take_profit")
        cp = pos.get("current_price") or pos.get("entry_price")
        action = pos.get("action", "buy")
        if tp is None or cp is None:
            continue
        if action == "buy" and float(cp) >= float(tp):
            logger.info(f"[portfolio] TARGET HIT: {pos['ticker']} price={cp} tp={tp}")
            targets_hit.append(pos)
        elif action in ("short", "sell") and float(cp) <= float(tp):
            logger.info(f"[portfolio] TARGET HIT (short): {pos['ticker']} price={cp} tp={tp}")
            targets_hit.append(pos)
    return targets_hit


def check_time_exits(alpaca_client=None, data_client=None) -> list[dict]:
    """
    Return open positions that have exceeded their max hold period.
    Keyed by strategy (mirrors backtest), not time_horizon.

    Breakout positions get stale-exit suppression: if prior close is still
    above the original breakout_level pivot, the time exit is skipped —
    the chandelier stop is managing the position.
    """
    positions = get_open_positions(alpaca_client)
    expired = []
    now = datetime.now(timezone.utc)

    for pos in positions:
        strategy  = pos.get("strategy", "swing")
        max_days  = MAX_HOLD_DAYS.get(strategy, MAX_HOLD_DAYS.get(pos.get("time_horizon", "swing"), 7))
        ts_raw    = pos.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).days
        except Exception:
            continue

        if age_days < max_days:
            continue

        # Stale-exit suppression for breakout positions (mirrors backtest)
        if strategy in ("breakout", "squeeze_breakout"):
            brk_lvl = float(pos.get("breakout_level") or 0)
            if brk_lvl > 0:
                try:
                    from bot.data import fetch_daily_bars
                    df = fetch_daily_bars(pos["ticker"], days=5)
                    if df is not None and len(df) >= 2:
                        prior_close = float(df["Close"].iloc[-2])
                        if prior_close >= brk_lvl:
                            logger.info(
                                f"[portfolio] TIME EXIT suppressed: {pos['ticker']} "
                                f"age={age_days}d prior_close={prior_close:.2f} >= pivot={brk_lvl:.2f}"
                            )
                            continue
                except Exception as _e:
                    logger.warning(f"[portfolio] stale-exit suppression check failed for {pos['ticker']}: {_e}")

        cp     = pos.get("current_price") or pos.get("entry_price") or 0
        entry  = float(pos.get("entry_price") or cp)
        action = pos.get("action", "buy")
        if action == "buy":
            pnl_pct = (float(cp) - entry) / entry * 100 if entry else 0
        else:
            pnl_pct = (entry - float(cp)) / entry * 100 if entry else 0
        pos["age_days"] = age_days
        pos["pnl_pct"]  = round(pnl_pct, 2)
        expired.append(pos)
        logger.info(
            f"[portfolio] TIME EXIT: {pos['ticker']} age={age_days}d "
            f"strategy={strategy} max={max_days}d pnl={pnl_pct:+.1f}%"
        )
    return expired



def calculate_partial_exit(position: dict, current_price: float) -> dict:
    """
    Determine whether to partially or fully exit a profitable position.

    Rules:
      - If price >= entry + 60% of (take_profit - entry): close 50% of shares,
        move logical stop to breakeven.
      - If price >= take_profit: close 100% of remaining shares.

    Returns dict with keys:
      close_pct      : 0.0, 0.5, or 1.0
      shares_to_close: int
      new_stop       : float or None   (breakeven when partial exit fires)
      reason         : str
    """
    empty = {"close_pct": 0.0, "shares_to_close": 0, "new_stop": None, "reason": "hold"}

    entry = float(position.get("entry_price") or 0)
    tp    = float(position.get("take_profit")  or 0)
    qty   = int(position.get("quantity")       or 0)

    if entry <= 0 or tp <= 0 or qty <= 0 or current_price <= entry:
        return empty

    distance      = tp - entry
    if distance <= 0:
        return empty

    progress      = (current_price - entry) / distance   # 0.0 → 1.0+

    if progress >= 1.0:
        return {
            "close_pct":       1.0,
            "shares_to_close": qty,
            "new_stop":        None,
            "reason":          "target_reached",
        }

    if progress >= 0.60:
        shares_to_close = max(1, qty // 2)
        return {
            "close_pct":       0.5,
            "shares_to_close": shares_to_close,
            "new_stop":        round(entry, 2),   # move stop to breakeven
            "reason":          "partial_60pct",
        }

    return empty


CHANDELIER_ATR_MULT     = 3.0
BREAKOUT_SWING_LOOKBACK = 3   # bars for structure stop (matches backtest)


def update_breakout_stops(alpaca_client, data_client=None) -> None:
    """
    Ratchet stop losses upward for open breakout positions using chandelier logic.
    Called each scan cycle so stops trail the highest price seen.

    Logic (mirrors backtest breakout_let_run):
      - chandelier = highest_price_seen - CHANDELIER_ATR_MULT * ATR(14)
      - structure  = min(prior 5 lows) * 0.995, capped below prior close
      - new_stop   = max(current_stop, chandelier, structure)
      - Only raised, never lowered.
    """
    from bot.logger import update_trade_trailing

    positions = get_open_positions(alpaca_client)
    breakout_positions = [
        p for p in positions
        if p.get("strategy") == "breakout" and p.get("action", "buy") == "buy"
    ]
    if not breakout_positions:
        return

    try:
        from bot.data import fetch_daily_bars
    except ImportError:
        logger.warning("[portfolio] fetch_daily_bars not available — skipping breakout stops")
        return

    for pos in breakout_positions:
        ticker    = pos["ticker"]
        trade_id  = pos.get("id")
        entry     = float(pos.get("entry_price") or 0)
        cur_stop  = float(pos.get("stop_loss") or 0)
        cur_price = float(pos.get("current_price") or entry)
        highest   = float(pos.get("highest_price_seen") or entry)

        if entry <= 0 or trade_id is None:
            continue

        # Ratchet highest seen
        new_highest = max(highest, cur_price)

        try:
            df = fetch_daily_bars(ticker, days=30)
            if df is None or len(df) < 5:
                continue

            # ATR(14) from recent bars
            high  = df["High"]
            low   = df["Low"]
            close = df["Close"]
            prev_close = close.shift(1)
            tr = pd.concat([
                high - low,
                (high - prev_close).abs(),
                (low  - prev_close).abs(),
            ], axis=1).max(axis=1)
            atr = float(tr.rolling(14).mean().iloc[-1])
            if atr <= 0:
                atr = entry * 0.02

            # Chandelier: highest_seen - 3 * ATR (uses prior data only — no look-ahead)
            chandelier = new_highest - CHANDELIER_ATR_MULT * atr

            # Structure stop from prior BREAKOUT_SWING_LOOKBACK bars (excluding current)
            prior_bars = df.iloc[-(BREAKOUT_SWING_LOOKBACK + 1):-1]
            if len(prior_bars) >= 1:
                prior_close  = float(prior_bars["Close"].iloc[-1])
                structure_lo = float(prior_bars["Low"].min()) * 0.995
                structure    = min(structure_lo, prior_close * 0.999)
            else:
                structure = cur_stop

            # New stop: raise-only, never push above current price
            new_stop = max(cur_stop, chandelier, structure)
            if new_stop >= cur_price:
                new_stop = cur_stop   # don't set stop above current price

            if new_stop > cur_stop:
                update_trade_stop(trade_id, round(new_stop, 4))
                update_trade_trailing(trade_id, round(new_highest, 4), round(new_stop, 4))
                logger.info(
                    f"[portfolio] BREAKOUT STOP RAISED: {ticker} "
                    f"stop {cur_stop:.2f} → {new_stop:.2f} "
                    f"(chandelier={chandelier:.2f} structure={structure:.2f} highest={new_highest:.2f})"
                )
            elif new_highest > highest:
                update_trade_trailing(trade_id, round(new_highest, 4),
                                      round(pos.get("trailing_stop_price") or cur_stop, 4))

        except Exception as e:
            logger.warning(f"[portfolio] update_breakout_stops failed for {ticker}: {e}")


def close_position_and_log(
    alpaca_client,
    trade: dict,
    current_price: float,
    session: str,
    status: str = "closed",
) -> None:
    """Close a position on Alpaca and update the DB record."""
    from bot.trader import close_position
    ticker   = trade["ticker"]
    entry    = float(trade.get("entry_price") or current_price)
    qty      = int(trade.get("quantity") or 1)
    action   = trade.get("action", "buy")

    try:
        close_position(alpaca_client, ticker)
    except Exception as e:
        logger.error(f"[portfolio] Failed to close {ticker} on Alpaca: {e}")
        return

    if action == "buy":
        pnl_dollar = (current_price - entry) * qty
    else:
        pnl_dollar = (entry - current_price) * qty
    pnl_pct = pnl_dollar / (entry * qty) if (entry * qty) != 0 else 0

    update_trade_exit(
        trade_id=trade["id"],
        exit_price=current_price,
        status=status,
        pnl_dollar=round(pnl_dollar, 2),
        pnl_pct=round(pnl_pct, 4),
    )
    record_trade_pnl(pnl_dollar)
    logger.info(
        f"[portfolio] Closed {ticker}: pnl=${pnl_dollar:.2f} ({pnl_pct*100:.2f}%) status={status}"
    )
