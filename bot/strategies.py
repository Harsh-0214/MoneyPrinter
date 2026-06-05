"""Strategy classifier — maps scored signals to named trading strategies."""

import logging

logger = logging.getLogger(__name__)

STRATEGY_CONFIGS = {
    "trend_follow": {
        "description": "EMA9>EMA21>EMA50 + ADX>22 + MACD hist positive + volume confirm — 2-5 day swing",
        "time_horizon": "swing",
        "sl_atr_mult": 2.0,
        "tp_rr": 2.5,
    },
    "mean_reversion": {
        "description": "RSI<35 or >72 AND Bollinger %B extreme — same-day to 2-day scalp",
        "time_horizon": "scalp",
        "sl_atr_mult": 1.5,
        "tp_rr": 2.0,
    },
    "breakout": {
        "description": "Price breaks R1/52wk high + ADX>20 + MACD confirm + volume surge >1.8x — 1-4 day momentum",
        "time_horizon": "swing",
        "sl_atr_mult": 1.2,   # tight: breakout entry should be precise; cut fast if it fails
        "tp_rr": 3.5,          # when they work, breakouts run further than trend trades
    },
    "breakdown": {
        "description": "Price breaks S1 with volume — 1-3 day momentum",
        "time_horizon": "swing",
        "sl_atr_mult": 1.5,
        "tp_rr": 2.5,
    },
    "squeeze_breakout": {
        "description": "BB squeeze + KC breakout + ADX>20 + vol>1.5x + MACD confirm — 2-4 day expansion play",
        "time_horizon": "swing",
        "sl_atr_mult": 1.5,   # was 2.0 — tighter stop, real squeeze expansions move fast
        "tp_rr": 3.0,          # was 2.5 — squeeze breakouts that work run further
    },
    "news_momentum": {
        "description": "Catalyst-driven move with trend confirmation — same-day scalp",
        "time_horizon": "scalp",
        "sl_atr_mult": 1.5,
        "tp_rr": 2.0,
    },
    "mixed": {
        "description": "Mixed signals — no dominant pattern, short hold only",
        "time_horizon": "swing",
        "sl_atr_mult": 2.0,
        "tp_rr": 2.0,
    },
}

# Confidence penalty when no clean strategy is identifiable
MIXED_CONFIDENCE_PENALTY = 0.05


def classify_strategy(score_result: dict, indicators: dict) -> dict:
    """
    Formally classify the strategy and compute stop/target based on strategy config.
    When strategy resolves to 'mixed', confidence is reduced by MIXED_CONFIDENCE_PENALTY.
    """
    sigs    = set(score_result.get("signals_triggered", []))
    action  = score_result.get("action", "hold")
    cp      = score_result.get("entry_price") or 0
    atr     = score_result.get("atr") or (cp * 0.02 if cp else 0)

    strategy = _classify(sigs, indicators, score_result)

    cfg     = STRATEGY_CONFIGS.get(strategy, STRATEGY_CONFIGS["mixed"])
    sl_mult = cfg["sl_atr_mult"]
    rr      = cfg["tp_rr"]
    horizon = cfg["time_horizon"]

    # Compute stops using strategy-specific ATR multiplier
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
    result["strategy"]              = strategy
    result["time_horizon"]          = horizon
    result["stop_loss"]             = stop_loss
    result["take_profit"]           = take_profit
    result["risk_reward"]           = rr
    result["strategy_description"]  = cfg["description"]

    # Penalise mixed: reduce confidence to discourage weak trades
    if strategy == "mixed":
        old_conf = result.get("confidence", 0.0)
        result["confidence"] = max(0.0, old_conf - MIXED_CONFIDENCE_PENALTY)
        if action != "hold":
            logger.info(
                f"[strategies] {score_result.get('ticker')}: mixed strategy — "
                f"confidence reduced {old_conf:.2f} -> {result['confidence']:.2f}"
            )

    return result


def _classify(sigs: set, ind: dict, score: dict) -> str:
    """
    Pick the most appropriate strategy using strict signal conditions.

    Hierarchy (first match wins):
      squeeze_breakout > breakout > breakdown > trend_follow > mean_reversion > news_momentum > mixed
    """
    ema_full_bull = score.get("ema_full_bull", False)
    adx           = float(ind.get("adx") or 0)
    rsi           = float(ind.get("rsi") or 50)
    bb_pctb       = ind.get("bb_pctb")          # may be None
    macd_hist     = float(ind.get("macd_hist") or 0)
    vol_ratio     = float(ind.get("volume_ratio") or 0)

    squeeze    = "bb_squeeze_detected"          in sigs
    kc_break   = "kc_breakout_bull"             in sigs or "kc_breakdown_bear" in sigs
    s1_break   = "broke_below_s1_with_volume"   in sigs
    ema_full   = "ema_full_bull_alignment"       in sigs or "ema_full_bear_alignment"  in sigs
    ema_part   = "ema_partial_bull_alignment"    in sigs or "ema_partial_bear_alignment" in sigs
    vol_conf   = "volume_confirm_bull"           in sigs or "volume_surge_bull" in sigs
    news_sig   = "news_positive" in sigs or "news_very_positive" in sigs or \
                 "news_negative" in sigs or "news_very_negative" in sigs

    # Breakout: within 2% above R1 or 52wk high with STRONG volume (>1.8x) + ADX + MACD confirm.
    # Raising volume threshold from 1.3x→1.8x eliminates low-conviction "near resistance" noise.
    cp   = float(score.get("entry_price") or 0)
    R1   = float(ind.get("R1") or 0)
    w52h = float(ind.get("wk52_high") or 0)
    at_r1_break   = R1   > 0 and cp > R1   and cp <= R1   * 1.02 and vol_ratio > 1.8
    at_52wk_break = w52h > 0 and cp >= w52h * 0.99 and vol_ratio > 1.8
    r1_break = at_r1_break or at_52wk_break or "broke_above_r1_with_volume" in sigs or "breaking_52wk_high" in sigs
    # Breakout must also have ADX trend + MACD momentum — prevents false breakouts in chop
    breakout_confirmed = r1_break and adx > 20 and macd_hist > 0

    # Trend follow: EMA9>EMA21>EMA50, ADX>18, MACD hist positive
    ema9_gt_ema21_gt_ema50 = (
        "ema_full_bull_alignment" in sigs or "ema_partial_bull_alignment" in sigs
    )
    trend_follow_ok = ema9_gt_ema21_gt_ema50 and adx > 18 and macd_hist > 0

    # Mean reversion: (RSI < 38 OR RSI > 68) OR (bb_pctb extreme) AND NOT full bull
    bb_extreme  = bb_pctb is not None and (bb_pctb < 0.15 or bb_pctb > 0.85)
    rsi_extreme = rsi < 38 or rsi > 68
    mean_rev_ok = (rsi_extreme or bb_extreme) and not ema_full_bull

    # Classify — squeeze_breakout now requires real momentum, not just pattern detection
    if squeeze and kc_break and adx > 20 and macd_hist > 0 and vol_ratio > 1.5:
        return "squeeze_breakout"
    if breakout_confirmed:
        return "breakout"
    if s1_break:
        return "breakdown"
    if trend_follow_ok and vol_conf:
        return "trend_follow"
    if mean_rev_ok:
        return "mean_reversion"
    if news_sig and (ema_full or ema_part or vol_conf):
        return "news_momentum"
    if ema_full or ema_part:
        return "trend_follow"    # EMA aligned but missing some conditions — still trend
    return "mixed"
