"""Portfolio state — pulls live data from Alpaca and cross-references trades DB."""

import logging
from datetime import datetime, timezone
from typing import Optional

from bot.logger import get_open_trades, get_trades_today, update_trade_exit
from bot.risk import record_trade_pnl

# Time-based exit rules per strategy horizon.
# Bot is designed for short-term and swing trades only — no multi-week holds.
# If a trade reaches this age WITHOUT hitting stop or target, close it.
MAX_HOLD_DAYS = {
    "scalp": 2,   # scalp must resolve by end of next day
    "swing": 7,   # swing trades get one calendar week max
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


def check_time_exits(alpaca_client=None) -> list[dict]:
    """
    Return open positions that have exceeded their max hold period.
    Doesn't close them — caller decides when to act (so dry-run is respected).

    Rules:
      - scalp: close after 2 calendar days regardless of P&L
      - swing: close after 10 calendar days regardless of P&L
    Both directions: if you're profitable take the gain, if you're flat/losing
    cut the position and redeploy capital elsewhere.
    """
    positions = get_open_positions(alpaca_client)
    expired = []
    now = datetime.now(timezone.utc)

    for pos in positions:
        horizon   = pos.get("time_horizon", "swing")
        max_days  = MAX_HOLD_DAYS.get(horizon, 10)
        ts_raw    = pos.get("timestamp")
        if not ts_raw:
            continue
        try:
            # timestamp stored as ISO string, may or may not have tz info
            ts = datetime.fromisoformat(ts_raw)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (now - ts).days
        except Exception:
            continue

        if age_days >= max_days:
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
                f"horizon={horizon} max={max_days}d pnl={pnl_pct:+.1f}%"
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
