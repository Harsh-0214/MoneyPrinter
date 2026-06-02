"""Strategy classifier — maps scored signals to named trading strategies."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

STRATEGY_CONFIGS = {
    "trend_follow": {
        "description": "EMA aligned + ADX strong + MACD rising + volume confirm",
        "time_horizon": "swing",
        "sl_atr_mult": 1.5,
        "tp_rr": 2.5,
    },
    "mean_reversion": {
        "description": "RSI/BB oversold + volume spike + Stoch RSI turning",
        "time_horizon": "scalp",
        "sl_atr_mult": 2.0,
        "tp_rr": 2.0,
    },
    "breakout": {
        "description": "Price breaks R1 or 52-week high with volume > 1.5x",
        "time_horizon": "swing",
        "sl_atr_mult": 1.0,
        "tp_rr": 3.0,
    },
    "breakdown": {
        "description": "Price breaks S1 or 52-week low with volume",
        "time_horizon": "swing",
        "sl_atr_mult": 1.0,
        "tp_rr": 2.5,
    },
    "squeeze_breakout": {
        "description": "BB squeeze resolved + KC breakout",
        "time_horizon": "swing",
        "sl_atr_mult": 1.5,
        "tp_rr": 2.5,
    },
    "news_momentum": {
        "description": "Sentiment > 0.4 + trend/breakout confirmation",
        "time_horizon": "scalp",
        "sl_atr_mult": 1.0,
        "tp_rr": 2.0,
    },
    "mixed": {
        "description": "Mixed signals — no dominant strategy pattern",
        "time_horizon": "swing",
        "sl_atr_mult": 1.5,
        "tp_rr": 2.0,
    },
}


def classify_strategy(score_result: dict, indicators: dict) -> dict:
    """
    Formally classify the strategy and compute stop/target based on strategy config.

    Takes the output of scorer.score_ticker() and enriches it with
    strategy-specific stop loss and take profit levels.
    """
    sigs    = set(score_result.get("signals_triggered", []))
    action  = score_result.get("action", "hold")
    cp      = score_result.get("entry_price") or 0
    atr     = score_result.get("atr") or (cp * 0.02 if cp else 0)
    vol_ratio = indicators.get("volume_ratio") or 0
    adx       = indicators.get("adx") or 0
    net_score = score_result.get("net_score", 0)

    # --- classify ---
    strategy = _classify(sigs, indicators, score_result)

    cfg = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["mixed"])
    sl_mult = cfg["sl_atr_mult"]
    rr      = cfg["tp_rr"]
    horizon = cfg["time_horizon"]

    # Compute stops
    if action == "buy" and cp:
        stop_loss   = round(cp - atr * sl_mult, 2)
        take_profit = round(cp + atr * sl_mult * rr, 2)
    elif action in ("short", "sell") and cp:
        stop_loss   = round(cp + atr * sl_mult, 2)
        take_profit = round(cp - atr * sl_mult * rr, 2)
    else:
        stop_loss   = score_result.get("stop_loss")
        take_profit = score_result.get("take_profit")

    result = {**score_result}
    result["strategy"]     = strategy
    result["time_horizon"] = horizon
    result["stop_loss"]    = stop_loss
    result["take_profit"]  = take_profit
    result["risk_reward"]  = rr
    result["strategy_description"] = cfg["description"]
    return result


def _classify(sigs: set, ind: dict, score: dict) -> str:
    """Pick strategy by signal hierarchy."""
    squeeze   = "bb_squeeze_detected" in sigs
    kc_break  = "kc_breakout_bull" in sigs or "kc_breakdown_bear" in sigs
    r1_break  = "broke_above_r1_with_volume" in sigs
    wk52_break = "breaking_52wk_high" in sigs
    s1_break  = "broke_below_s1_with_volume" in sigs
    ema_full  = "ema_full_bull_alignment" in sigs or "ema_full_bear_alignment" in sigs
    ema_part  = "ema_partial_bull_alignment" in sigs or "ema_partial_bear_alignment" in sigs
    macd_r    = "macd_hist_rising_2bars" in sigs or "macd_hist_falling_2bars" in sigs
    vol_conf  = "volume_confirm_bull" in sigs or "volume_surge_bull" in sigs
    mean_rev  = "rsi_oversold" in sigs or "bb_deeply_oversold" in sigs or "cci_oversold" in sigs
    stoch_rev = "stochrsi_bull_cross_below30" in sigs
    news_pos  = "news_positive" in sigs or "news_very_positive" in sigs
    news_neg  = "news_negative" in sigs or "news_very_negative" in sigs
    adx       = (ind.get("adx") or 0)

    if squeeze and kc_break:
        return "squeeze_breakout"
    if r1_break or wk52_break:
        return "breakout"
    if s1_break:
        return "breakdown"
    if (ema_full or ema_part) and adx > 25 and macd_r and vol_conf:
        return "trend_follow"
    if mean_rev or stoch_rev:
        return "mean_reversion"
    if (news_pos or news_neg) and (ema_full or ema_part or vol_conf):
        return "news_momentum"
    if ema_full or ema_part:
        return "trend_follow"
    return "mixed"
