"""Strategy classifier — maps scored signals to named trading strategies."""

import logging

logger = logging.getLogger(__name__)

STRATEGY_CONFIGS = {
    "trend_follow": {
        "description": "EMA9>EMA21>EMA50 + ADX>22 + MACD hist positive + volume confirm — 2-5 day swing",
        "time_horizon": "swing",
        "sl_atr_mult": 5.0,
        "tp_rr": 3.0,
    },
    "mean_reversion": {
        "description": "RSI<38 or >68 AND BB extreme, ADX<22, not in downtrend — same-day to 2-day scalp",
        "time_horizon": "scalp",
        "sl_atr_mult": 4.0,
        "tp_rr": 2.0,
    },
    "breakout": {
        "description": "Level break + in_uptrend + vol>=2x + ADX>=25 — 1-4 day momentum",
        "time_horizon": "swing",
        "sl_atr_mult": 3.6,
        "tp_rr": 3.5,
    },
    "breakdown": {
        "description": "Price breaks S1 with volume — 1-3 day momentum",
        "time_horizon": "swing",
        "sl_atr_mult": 4.0,
        "tp_rr": 2.5,
    },
    "squeeze_breakout": {
        "description": "BB squeeze + KC breakout + vol>=1.5x + not in downtrend — 2-4 day expansion play",
        "time_horizon": "swing",
        "sl_atr_mult": 4.0,
        "tp_rr": 3.0,
    },
    "news_momentum": {
        "description": "Catalyst-driven move with trend confirmation — same-day scalp",
        "time_horizon": "scalp",
        "sl_atr_mult": 4.0,
        "tp_rr": 2.0,
    },
    "mixed": {
        "description": "Mixed signals — no dominant pattern, short hold only",
        "time_horizon": "swing",
        "sl_atr_mult": 5.0,
        "tp_rr": 2.0,
    },
}

# Confidence penalty when no clean strategy is identifiable
MIXED_CONFIDENCE_PENALTY = 0.15

ATR_STOP_FLOOR = 0.025  # never stop tighter than 2.5% — avoids noise exits on low-ATR stocks
ATR_STOP_CAP   = 0.095  # never stop wider than 9.5% — limits max loss on extremely volatile stocks


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

    # ATR-based stop: uses per-strategy sl_atr_mult with floor/cap to fit actual volatility.
    # R:R is preserved — take profit scales proportionally with the stop distance.
    atr_pct = (atr / cp) if cp else 0.02
    sl_pct  = max(ATR_STOP_FLOOR, min(atr_pct * sl_mult, ATR_STOP_CAP))

    if action == "buy" and cp:
        stop_loss   = round(cp * (1 - sl_pct), 2)
        take_profit = round(cp * (1 + sl_pct * rr), 2)
    elif action in ("short", "sell") and cp:
        stop_loss   = round(cp * (1 + sl_pct), 2)
        take_profit = round(cp * (1 - sl_pct * rr), 2)
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
    # ── Market structure (computed once, shared by all strategy checks) ────────
    price  = float(ind.get("current_price") or score.get("entry_price") or 0)
    ema50  = float(ind.get("ema50")  or 0)
    ema200 = float(ind.get("ema200") or 0)
    in_downtrend = ema50 > 0 and ema200 > 0 and price > 0 and price < ema50 and price < ema200
    in_uptrend   = ema50 > 0 and ema200 > 0 and price > 0 and price > ema50 and price > ema200

    ema_full_bull = score.get("ema_full_bull", False)
    adx           = float(ind.get("adx") or 0)
    rsi           = float(ind.get("rsi") or 50)
    bb_pctb       = ind.get("bb_pctb")          # may be None
    macd_hist     = float(ind.get("macd_hist") or 0)
    macd_hist_p1  = float(ind.get("macd_hist_prev1") or 0)
    vol_ratio     = float(ind.get("volume_ratio") or 0)

    squeeze  = "bb_squeeze_detected"          in sigs
    kc_bull  = "kc_breakout_bull"             in sigs   # bullish direction only — bearish break ≠ long entry
    s1_break = "broke_below_s1_with_volume"   in sigs
    ema_full = "ema_full_bull_alignment"       in sigs or "ema_full_bear_alignment" in sigs
    ema_part = "ema_partial_bull_alignment"    in sigs or "ema_partial_bear_alignment" in sigs
    vol_conf = "volume_confirm_bull"           in sigs or "volume_surge_bull" in sigs
    news_sig = ("news_positive" in sigs or "news_very_positive" in sigs or
                "news_negative" in sigs or "news_very_negative" in sigs)

    # ── Breakout: must be breaking a real level in confirmed uptrend with strong momentum
    cp   = float(score.get("entry_price") or price)
    R1   = float(ind.get("R1") or 0)
    w52h = float(ind.get("wk52_high") or 0)
    at_r1_break   = R1   > 0 and cp > R1   and cp <= R1   * 1.02
    at_52wk_break = w52h > 0 and cp >= w52h * 0.99
    r1_break = (at_r1_break or at_52wk_break
                or "broke_above_r1_with_volume" in sigs
                or "breaking_52wk_high" in sigs)
    # Require uptrend context + 2x volume + ADX ≥ 25 — consistent with backtest guards
    breakout_confirmed = r1_break and in_uptrend and vol_ratio >= 2.0 and adx >= 25

    # ── Trend follow: ADX>22, volume confirmed, RSI not extended.
    # in_uptrend + MACD accelerating + vol >= 1.5x + RSI not overbought.
    ema_aligned     = "ema_full_bull_alignment" in sigs or "ema_partial_bull_alignment" in sigs
    trend_follow_ok = (ema_aligned and in_uptrend
                       and adx > 22
                       and macd_hist > 0
                       and macd_hist > macd_hist_p1
                       and vol_ratio >= 1.5
                       and rsi < 68)

    # ── Mean reversion: both RSI AND BB extreme + MACD improving
    # MACD improving (hist rising toward zero) confirms the bottom is forming —
    # filters out entries into ongoing free-falls that keep going lower.
    bb_extreme    = bb_pctb is not None and (bb_pctb < 0.15 or bb_pctb > 0.85)
    rsi_extreme   = rsi < 30 or rsi > 70
    macd_turning  = macd_hist > macd_hist_p1   # histogram moving in right direction
    mean_rev_ok   = (rsi_extreme and bb_extreme and macd_turning
                     and not ema_full_bull and adx < 20 and not in_downtrend)

    # ── Squeeze breakout: BB compression releasing UPWARD — requires confirmed uptrend
    # Kept broad so squeeze setups are labeled correctly (not reclassified as
    # trend_follow). Blocking happens at the entry gate via BAD_STRATEGIES.
    squeeze_ok = squeeze and kc_bull and in_uptrend and vol_ratio >= 1.5

    # ── Classification (first match wins) ─────────────────────────────────────
    if squeeze_ok:
        return "squeeze_breakout"
    if breakout_confirmed:
        return "breakout"
    if s1_break:
        return "breakdown"
    if trend_follow_ok and vol_conf:
        return "trend_follow"
    # Scorer-validated oversold bounce (set in score_ticker's mean-rev override).
    # The full setup was already checked there, so trust the flag here.
    if "mean_reversion_long_setup" in sigs:
        return "mean_reversion"
    if mean_rev_ok:
        return "mean_reversion"
    if news_sig and (ema_full or ema_part or vol_conf):
        return "news_momentum"
    return "mixed"   # EMA aligned but without uptrend+ADX+MACD+vol = no coherent edge
