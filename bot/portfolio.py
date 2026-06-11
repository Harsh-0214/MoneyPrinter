"""Portfolio state — pulls live data from Alpaca and cross-references trades DB."""

import logging
from datetime import datetime, timezone
from typing import Optional

import pandas as pd

from bot.logger import get_open_trades, get_trades_today, update_trade_exit, update_trade_stop
from bot.risk import record_trade_pnl

# Time-based exit rules in CALENDAR days, keyed by strategy — exactly the
# backtest's measured "hold_strategy" variant, the only configuration that
# was profitable across both the 2024-25 and 2025-26 test windows
# (+5.9% / +2.5% vs baseline -6.7% / +3.6%). Breakout strategies get 21 days
# so the chandelier stop can do its job (with stale-exit suppression below).
HOLD_BY_STRATEGY = {
    "trend_follow":      7,
    "mean_reversion":    5,
    "news_momentum":     3,
    "breakdown":         7,
    "mixed":             5,
    "swing":             7,   # adopted/reconciled positions default
    "breakout":         21,
    "squeeze_breakout": 21,
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
        max_days  = HOLD_BY_STRATEGY.get(strategy, 5)
        ts_raw    = pos.get("timestamp")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            # Calendar days — matches how the backtest variant was measured
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
        # Breakeven stop means a partial was already taken — don't sell half
        # of the remainder again every cycle.
        sl = float(position.get("stop_loss") or 0)
        if sl >= entry:
            return empty
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
    dry_run: bool = False,
) -> None:
    """Close a position on Alpaca and update the DB record."""
    from bot.trader import close_position
    ticker   = trade["ticker"]
    entry    = float(trade.get("entry_price") or current_price)
    qty      = int(trade.get("quantity") or 1)
    action   = trade.get("action", "buy")

    try:
        close_position(alpaca_client, ticker, dry_run=dry_run)
    except Exception as e:
        # Position already gone on Alpaca (bracket fired, manual close, never
        # filled): still close the DB record so it isn't retried forever.
        msg = str(e).lower()
        if "does not exist" in msg or "not found" in msg or "404" in msg:
            logger.warning(f"[portfolio] {ticker} not on Alpaca — closing DB record only")
        else:
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


# ── Alpaca ⇄ DB reconciliation ────────────────────────────────────────────────

ADOPT_STOP_ATR_MULT = 3.0   # stop = entry − 3×ATR (matches live entry sizing)
ADOPT_RISK_REWARD   = 2.0   # target = entry + 2×(entry − stop)


def reconcile_positions(alpaca_client, data_client=None, dry_run: bool = False,
                        attach_exits: bool = True) -> dict:
    """
    Make the trades DB agree with Alpaca, which is the ground truth.

      1. DB rows marked open whose position no longer exists on Alpaca
         → close the row (exit price from the last sell fill when available).
      2. Alpaca positions with no open DB row (orphans from order timeouts,
         lost commits, manual buys) → adopt: insert an open row with an
         ATR-based stop/target and the real entry time from order history,
         and attach a GTC OCO exit on Alpaca if none is working.

    Returns {"adopted": [...], "closed": [...]}.
    """
    from bot.trader import (get_positions, get_entry_fill_info,
                            get_last_sell_fill_price, has_open_exit_order,
                            submit_oco_exit)
    from bot.logger import log_trade

    summary = {"adopted": [], "closed": []}
    if alpaca_client is None:
        return summary

    try:
        live = {p["symbol"]: p for p in get_positions(alpaca_client)}
    except Exception as e:
        logger.warning(f"[reconcile] could not fetch live positions: {e}")
        return summary

    db_open = get_open_trades()
    db_open_tickers = {t.get("ticker") for t in db_open}

    # ── 1. DB-open rows with no live position → close out ────────────────────
    for trade in db_open:
        ticker = trade.get("ticker")
        if not ticker or ticker in live:
            continue
        entry = float(trade.get("entry_price") or 0)
        qty   = int(trade.get("quantity") or 0)
        exit_price = None
        try:
            exit_price = get_last_sell_fill_price(alpaca_client, ticker)
        except Exception:
            pass
        if not exit_price:
            exit_price = entry
        pnl_dollar = (exit_price - entry) * qty if trade.get("action", "buy") == "buy" else (entry - exit_price) * qty
        pnl_pct    = (pnl_dollar / (entry * qty)) if entry and qty else 0.0
        update_trade_exit(
            trade_id=trade["id"],
            exit_price=exit_price,
            status="closed_external",
            pnl_dollar=round(pnl_dollar, 2),
            pnl_pct=round(pnl_pct, 4),
        )
        record_trade_pnl(pnl_dollar)
        summary["closed"].append(ticker)
        logger.warning(
            f"[reconcile] {ticker} open in DB but gone on Alpaca — "
            f"closed as closed_external @ {exit_price:.2f}"
        )

    # ── 2. Live positions with no open DB row → adopt ─────────────────────────
    for ticker, p in live.items():
        if ticker in db_open_tickers:
            continue
        if str(p.get("side", "")).lower().endswith("short"):
            logger.warning(f"[reconcile] {ticker} is a SHORT position — not adopting (manual review)")
            continue

        qty   = int(float(p.get("qty") or 0))
        entry = float(p.get("avg_entry_price") or 0)
        if qty <= 0 or entry <= 0:
            continue

        # ATR for stop/target
        atr = entry * 0.02
        try:
            from bot.data import fetch_daily_bars
            df = fetch_daily_bars(ticker, days=40)
            if df is not None and len(df) >= 15:
                high, low, close = df["High"], df["Low"], df["Close"]
                prev_close = close.shift(1)
                tr = pd.concat([high - low, (high - prev_close).abs(),
                                (low - prev_close).abs()], axis=1).max(axis=1)
                atr_val = float(tr.rolling(14).mean().iloc[-1])
                if atr_val > 0:
                    atr = atr_val
        except Exception as e:
            logger.warning(f"[reconcile] ATR fetch failed for {ticker}: {e}")

        stop   = round(entry - ADOPT_STOP_ATR_MULT * atr, 2)
        target = round(entry + ADOPT_RISK_REWARD * (entry - stop), 2)

        # Real entry time from order history when available
        entry_ts = None
        fill = get_entry_fill_info(alpaca_client, ticker)
        if fill and fill.get("filled_at"):
            entry_ts = fill["filled_at"]

        trade_id = log_trade(
            session="reconcile",
            ticker=ticker,
            action="buy",
            strategy="swing",
            time_horizon="swing",
            quantity=qty,
            entry_price=entry,
            limit_price=entry,
            stop_loss=stop,
            take_profit=target,
            confidence=0.0,
            net_score=0,
            bull_score=0,
            bear_score=0,
            signals_triggered=[],
            signals_against=[],
            reasoning="Adopted from Alpaca — position existed without a DB record",
            risk_reward=ADOPT_RISK_REWARD,
            macro_bias="unknown",
            vix_level=0,
            alpaca_order_id="adopted",
            status="open",
        )
        # Backdate to the real fill time so time exits count from actual entry
        if entry_ts:
            try:
                import sqlite3
                from bot.logger import DB_PATH
                conn = sqlite3.connect(str(DB_PATH))
                conn.execute("UPDATE trades SET timestamp = ? WHERE id = ?", (entry_ts, trade_id))
                conn.commit()
                conn.close()
            except Exception as e:
                logger.warning(f"[reconcile] backdate failed for {ticker}: {e}")

        # Protect it server-side if no sell order is already working
        if attach_exits:
            try:
                if not has_open_exit_order(alpaca_client, ticker):
                    submit_oco_exit(alpaca_client, ticker, qty, target, stop, dry_run=dry_run)
            except Exception as e:
                logger.warning(f"[reconcile] OCO attach failed for {ticker}: {e}")

        summary["adopted"].append(ticker)
        logger.warning(
            f"[reconcile] ADOPTED {ticker}: {qty} sh @ {entry:.2f} "
            f"stop={stop:.2f} target={target:.2f} entered={entry_ts or 'unknown'}"
        )

    if summary["adopted"] or summary["closed"]:
        logger.info(f"[reconcile] adopted={summary['adopted']} closed={summary['closed']}")
    return summary
