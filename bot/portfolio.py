"""Portfolio state — pulls live data from Alpaca and cross-references trades DB."""

import logging
from datetime import datetime, timezone
from typing import Optional

from bot.logger import get_open_trades, get_trades_today, update_trade_exit
from bot.risk import record_trade_pnl

# Time-based exit rules per strategy horizon.
# If a trade reaches this age WITHOUT hitting stop or target, close it.
# Profitable trades are closed to bank the gain.
# Flat/losing trades are closed to free capital.
MAX_HOLD_DAYS = {
    "scalp":    2,    # scalp trades must resolve in 2 days
    "swing":    10,   # swing trades get 10 trading days max
    "position": 45,   # position trades can hold through corrections — ~9 calendar weeks
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
