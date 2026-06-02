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
    Hard cap: 10% of portfolio in any single position.
    """
    if is_kill_switch_active():
        logger.warning("[risk] Kill switch active — position size = 0")
        return {"shares": 0, "dollar_risk": 0, "reason": "kill_switch"}

    if price <= 0 or atr <= 0:
        return {"shares": 0, "dollar_risk": 0, "reason": "invalid_price_or_atr"}

    # High ATR: reduce by 40%
    vol_adj = 0.60 if high_vol_flag else 1.0

    dollar_risk = portfolio_value * 0.02 * confidence * vix_multiplier * vol_adj
    shares = floor(dollar_risk / (atr * 1.5))

    # Cap at 10% of portfolio
    max_val    = portfolio_value * 0.10
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


def compute_stops(
    action: str,
    entry_price: float,
    atr: float,
    strategy: str = "trend_follow",
    rr_target: float = 2.5,
) -> dict:
    """Compute stop loss and take profit based on strategy ATR multipliers."""
    from bot.strategies import STRATEGY_CONFIGS
    cfg = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["mixed"])
    sl_mult = cfg["sl_atr_mult"]
    rr      = rr_target if rr_target else cfg["tp_rr"]

    if action == "buy":
        sl = round(entry_price - atr * sl_mult, 2)
        tp = round(entry_price + atr * sl_mult * rr, 2)
    elif action in ("short", "sell"):
        sl = round(entry_price + atr * sl_mult, 2)
        tp = round(entry_price - atr * sl_mult * rr, 2)
    else:
        sl = None
        tp = None

    return {"stop_loss": sl, "take_profit": tp, "risk_reward": rr}
