"""Position sizing, stop/target calculation, and kill switch logic."""

import logging
import os
from math import floor
from typing import Optional

logger = logging.getLogger(__name__)

KILL_SWITCH_ACTIVE = False
DAILY_REALIZED_PNL = 0.0
DAILY_START_VALUE  = 0.0


def init_daily_state(starting_portfolio_value: float) -> None:
    global KILL_SWITCH_ACTIVE, DAILY_REALIZED_PNL, DAILY_START_VALUE
    KILL_SWITCH_ACTIVE  = False
    DAILY_REALIZED_PNL  = 0.0
    DAILY_START_VALUE   = starting_portfolio_value
    logger.info(f"[risk] Daily state initialized. Starting value: ${starting_portfolio_value:,.2f}")


def record_trade_pnl(pnl: float) -> None:
    """Called after each closed trade to accumulate daily P&L and check kill switch."""
    global KILL_SWITCH_ACTIVE, DAILY_REALIZED_PNL, DAILY_START_VALUE
    DAILY_REALIZED_PNL += pnl

    if DAILY_START_VALUE > 0:
        pnl_pct = DAILY_REALIZED_PNL / DAILY_START_VALUE
        if pnl_pct < -0.03 and not KILL_SWITCH_ACTIVE:
            KILL_SWITCH_ACTIVE = True
            logger.critical(
                f"[risk] KILL SWITCH ACTIVATED — daily P&L {pnl_pct*100:.2f}% "
                f"(${DAILY_REALIZED_PNL:,.2f}) exceeds -3% threshold"
            )


def is_kill_switch_active() -> bool:
    return KILL_SWITCH_ACTIVE


def get_vix_multiplier(vix: float) -> float:
    """Return position size multiplier based on VIX level."""
    if vix < 15:
        return 1.0
    elif vix < 20:
        return 0.85
    elif vix < 25:
        return 0.70
    elif vix < 35:
        return 0.50
    else:
        return 0.0   # kill all new longs


def calculate_position(
    portfolio_value: float,
    confidence: float,
    atr: float,
    price: float,
    vix_multiplier: float = 1.0,
    high_vol_flag: bool = False,
) -> dict:
    """
    Compute the number of shares to buy/short.

    Base risk: 2% of portfolio per trade, scaled by confidence + VIX + volatility.
    Hard cap: 12% of portfolio (8% for high-vol / gap-prone tickers).
    """
    if is_kill_switch_active():
        logger.warning("[risk] Kill switch active — position size = 0")
        return {"shares": 0, "dollar_risk": 0, "reason": "kill_switch"}

    if price <= 0 or atr <= 0:
        return {"shares": 0, "dollar_risk": 0, "reason": "invalid_price_or_atr"}

    # High-vol tickers: halve risk budget AND cap at 8% (not 12%)
    # Worst-case gap math: 8% × 17% gap = 1.4% portfolio loss (acceptable)
    #                      12% × 17% gap = 2.0% portfolio loss (acceptable)
    #                      20% × 17% gap = 3.4% portfolio loss (too much)
    vol_adj = 0.60 if high_vol_flag else 1.0
    pos_cap = 0.08 if high_vol_flag else 0.12

    dollar_risk = portfolio_value * 0.02 * confidence * vix_multiplier * vol_adj
    shares = floor(dollar_risk / (atr * 1.5))

    max_val    = portfolio_value * pos_cap
    max_shares = floor(max_val / price)
    shares     = min(shares, max_shares)
    shares     = max(0, shares)

    return {
        "shares": shares,
        "dollar_risk": round(dollar_risk, 2),
        "max_position_value": round(max_val, 2),
        "position_value": round(shares * price, 2),
        "reason": "ok" if shares > 0 else "zero_shares",
    }


def calculate_scale_in(
    existing_position: dict,
    current_price: float,
    confidence: float,
    atr: float,
    portfolio_value: float,
) -> int:
    """
    Return shares to add to a profitable open position (scale-in).

    Conditions that must all be met:
      - Position is profitable by >= 2% unrealised gain
      - confidence > 0.75
      - Total position value after adding would not exceed 15% of portfolio
      - Scale-in size capped at 50% of original entry shares

    Returns 0 if any condition is not met.
    """
    if is_kill_switch_active():
        return 0
    if confidence <= 0.75 or atr <= 0 or current_price <= 0 or portfolio_value <= 0:
        return 0

    entry_price  = float(existing_position.get("entry_price") or 0)
    orig_qty     = int(existing_position.get("quantity") or 0)
    if entry_price <= 0 or orig_qty <= 0:
        return 0

    unrealised_pct = (current_price - entry_price) / entry_price
    if unrealised_pct < 0.02:
        return 0

    # Max allowed total position value: 15% of portfolio
    max_position_value = portfolio_value * 0.15
    current_value      = current_price * orig_qty
    if current_value >= max_position_value:
        return 0

    headroom_dollars = max_position_value - current_value
    max_add_shares   = floor(headroom_dollars / current_price)

    # 50% of original entry size
    scale_in_shares = floor(orig_qty * 0.50)
    scale_in_shares = min(scale_in_shares, max_add_shares)
    scale_in_shares = max(0, scale_in_shares)

    if scale_in_shares > 0:
        logger.info(
            f"[risk] scale-in approved: {scale_in_shares} shares "
            f"(unrealised={unrealised_pct*100:.1f}% conf={confidence:.2f})"
        )
    return scale_in_shares


TRAILING_ACTIVATE_PCT  = 0.06   # wait until +6% before trailing
TRAILING_TRAIL_PCT     = 0.05   # trail 5% below highest seen
TRAIL_TIGHT_PCT        = 0.025  # tighten to 2.5% once PARTIAL_TIGHT_PCT reached
BREAKEVEN_TRIGGER_PCT  = 0.03   # move stop to entry at +3%
PARTIAL_TIGHT_PCT      = 0.08   # highest_price_seen threshold to activate tight trail


def update_trailing_stop(trade_record: dict, current_price: float) -> dict:
    """
    Returns updated trade_record with trailing stop, breakeven, and tight-trail logic.
    Call this every cycle for open long positions.

    Keys added/updated in returned dict:
      - highest_price_seen: float
      - trailing_stop_price: float or None
      - trailing_stop_updated: bool
      - trailing_stop_triggered: bool
      - breakeven_set: bool         (True when stop was moved to entry this cycle)
      - new_stop_loss: float|None   (new stop_loss value if breakeven was set)
    """
    result = dict(trade_record)
    result["trailing_stop_updated"]   = False
    result["trailing_stop_triggered"] = False
    result["breakeven_set"]           = False
    result["new_stop_loss"]           = None

    entry_price = float(trade_record.get("entry_price") or 0)
    if entry_price <= 0 or current_price <= 0:
        return result

    action = trade_record.get("action", "buy")
    if action not in ("buy",):
        return result

    highest = float(trade_record.get("highest_price_seen") or entry_price)
    if current_price > highest:
        highest = current_price
    result["highest_price_seen"] = highest

    gain_pct = (highest - entry_price) / entry_price if entry_price > 0 else 0.0

    # Breakeven: move stop to entry once highest has reached +3%
    current_stop = float(trade_record.get("stop_loss") or 0)
    if gain_pct >= BREAKEVEN_TRIGGER_PCT and current_stop < entry_price:
        result["new_stop_loss"]           = round(entry_price, 2)
        result["breakeven_set"]           = True
        result["trailing_stop_updated"]   = True

    # Tight trail activates once highest has reached +8%
    tight = gain_pct >= PARTIAL_TIGHT_PCT

    if gain_pct < TRAILING_ACTIVATE_PCT:
        return result

    trail_pct   = TRAIL_TIGHT_PCT if tight else TRAILING_TRAIL_PCT
    trail_price = round(highest * (1.0 - trail_pct), 2)
    old_trail   = trade_record.get("trailing_stop_price")

    if old_trail is None or trail_price > float(old_trail):
        result["trailing_stop_price"]   = trail_price
        result["trailing_stop_updated"] = True

    effective_trail = result.get("trailing_stop_price") or trail_price
    if current_price <= float(effective_trail):
        result["trailing_stop_triggered"] = True
        logger.info(
            f"[risk] Trailing stop triggered: price={current_price:.2f} "
            f"trail={effective_trail:.2f} highest={highest:.2f} tight={tight}"
        )

    return result


