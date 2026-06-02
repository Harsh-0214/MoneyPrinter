"""Rules-based decision engine. Pure deterministic Python — no AI API."""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def _v(val, default=0.0):
    """Safe value getter with default."""
    return val if val is not None else default


def score_ticker(
    ticker: str,
    indicators: dict,
    news_sentiment: dict,
    macro_context: dict,
) -> dict:
    """
    Score a ticker and return a full signal dict.

    Returns dict with action, confidence, entry/stop/target, reasoning, etc.
    """
    bull = 0.0
    bear = 0.0
    signals_triggered = []
    signals_against   = []
    reasoning_parts   = []

    ind = indicators
    cp = _v(ind.get("current_price"))
    if cp == 0:
        return _no_signal(ticker, "no_price_data")

    # ──────────────────────────────────────────────────────────────
    # TREND SIGNALS
    # ──────────────────────────────────────────────────────────────
    e9   = _v(ind.get("ema9"))
    e21  = _v(ind.get("ema21"))
    e50  = _v(ind.get("ema50"))
    e200 = _v(ind.get("ema200"))
    adx  = _v(ind.get("adx"))

    ema_bull_base = 0.0
    ema_bear_base = 0.0
    if e9 > 0 and e21 > 0 and e50 > 0 and e200 > 0:
        if e9 > e21 > e50 > e200:
            ema_bull_base = 25
            signals_triggered.append("ema_full_bull_alignment")
            reasoning_parts.append("EMA9>EMA21>EMA50>EMA200 full bull alignment")
        elif e9 < e21 < e50 < e200:
            ema_bear_base = 25
            signals_triggered.append("ema_full_bear_alignment")
            reasoning_parts.append("EMA9<EMA21<EMA50<EMA200 full bear alignment")
        elif e9 > e21 > e50:
            ema_bull_base = 18
            signals_triggered.append("ema_partial_bull_alignment")
            reasoning_parts.append("EMA9>EMA21>EMA50 partial bull alignment")
        elif e9 < e21 < e50:
            ema_bear_base = 18
            signals_triggered.append("ema_partial_bear_alignment")
            reasoning_parts.append("EMA9<EMA21<EMA50 partial bear alignment")
    elif e9 > 0 and e21 > 0 and e50 > 0:
        if e9 > e21 > e50:
            ema_bull_base = 18
            signals_triggered.append("ema_partial_bull_alignment")
        elif e9 < e21 < e50:
            ema_bear_base = 18
            signals_triggered.append("ema_partial_bear_alignment")

    # ADX modifier
    adx_mult = 1.0
    if adx > 30:
        adx_mult = 1.3
        reasoning_parts.append(f"ADX {adx:.1f}>30 strong trend (1.3x multiplier)")
    elif adx < 20:
        adx_mult = 0.5
        reasoning_parts.append(f"ADX {adx:.1f}<20 weak trend (0.5x multiplier)")

    bull += ema_bull_base * adx_mult
    bear += ema_bear_base * adx_mult

    # DI signals
    di_plus  = _v(ind.get("adx_di_plus"))
    di_minus = _v(ind.get("adx_di_minus"))
    if adx > 25 and di_plus > 0 and di_minus > 0:
        if di_plus > di_minus:
            bull += 8
            signals_triggered.append("di_plus_dominant")
        else:
            bear += 8
            signals_triggered.append("di_minus_dominant")

    # MACD
    macd_hist       = _v(ind.get("macd_hist"))
    macd_hist_prev1 = _v(ind.get("macd_hist_prev1"))
    macd_hist_prev2 = _v(ind.get("macd_hist_prev2"))
    macd_bull_cross = ind.get("macd_bull_cross") or False
    macd_bear_cross = ind.get("macd_bear_cross") or False

    if macd_hist > 0 and macd_hist_prev1 > 0 and macd_hist > macd_hist_prev1 and macd_hist_prev1 > macd_hist_prev2:
        bull += 15
        signals_triggered.append("macd_hist_rising_2bars")
        reasoning_parts.append("MACD histogram positive and rising 2+ bars")
    elif macd_hist < 0 and macd_hist_prev1 < 0 and macd_hist < macd_hist_prev1 and macd_hist_prev1 < macd_hist_prev2:
        bear += 15
        signals_triggered.append("macd_hist_falling_2bars")
        reasoning_parts.append("MACD histogram negative and falling 2+ bars")

    if macd_bull_cross:
        bull += 12
        signals_triggered.append("macd_bullish_cross")
        reasoning_parts.append("MACD bullish crossover (last 3 bars)")
    if macd_bear_cross:
        bear += 12
        signals_triggered.append("macd_bearish_cross")

    # Fading momentum
    if macd_hist > 0 and macd_hist_prev1 > 0 and macd_hist < macd_hist_prev1:
        bull -= 5
        signals_against.append("macd_hist_fading_bull")
    if macd_hist < 0 and macd_hist_prev1 < 0 and macd_hist > macd_hist_prev1:
        bear -= 5
        signals_against.append("macd_hist_fading_bear")

    # Parabolic SAR
    psar_bullish = ind.get("psar_bullish")
    if psar_bullish is True:
        bull += 8
        signals_triggered.append("psar_bullish")
    elif psar_bullish is False:
        bear += 8
        signals_triggered.append("psar_bearish")

    # VWAP
    vwap = _v(ind.get("vwap"))
    if vwap > 0 and cp > 0:
        vwap_diff_pct = (cp - vwap) / vwap * 100
        if vwap_diff_pct > 0.3:
            bull += 8
            signals_triggered.append("price_above_vwap")
            reasoning_parts.append(f"Price {vwap_diff_pct:.2f}% above VWAP")
        elif vwap_diff_pct < -0.3:
            bear += 8
            signals_triggered.append("price_below_vwap")
            reasoning_parts.append(f"Price {abs(vwap_diff_pct):.2f}% below VWAP")

    # ──────────────────────────────────────────────────────────────
    # MOMENTUM SIGNALS
    # ──────────────────────────────────────────────────────────────
    rsi = _v(ind.get("rsi"))
    if rsi > 0:
        if rsi < 20:
            bull += 25
            signals_triggered.append("rsi_extremely_oversold")
            reasoning_parts.append(f"RSI {rsi:.1f} extremely oversold")
        elif rsi < 30:
            bull += 15
            signals_triggered.append("rsi_oversold")
            reasoning_parts.append(f"RSI {rsi:.1f} oversold")
        elif rsi < 40:
            bear += 5
            signals_triggered.append("rsi_weak")
        elif 40 < rsi <= 50:
            bear += 8
            signals_triggered.append("rsi_bearish_momentum")
        elif 50 < rsi <= 60:
            bull += 8
            signals_triggered.append("rsi_healthy_momentum")
            reasoning_parts.append(f"RSI {rsi:.1f} healthy bull momentum")
        elif 60 < rsi <= 70:
            bull += 5
            signals_triggered.append("rsi_strong")
        elif rsi > 80:
            bear += 25
            signals_against.append("rsi_extremely_overbought")
            reasoning_parts.append(f"RSI {rsi:.1f} extremely overbought")
        elif rsi > 70:
            bear += 15
            signals_against.append("rsi_overbought")
            reasoning_parts.append(f"RSI {rsi:.1f} overbought")

    # Stochastic RSI
    sk = _v(ind.get("stoch_k"))
    sd = _v(ind.get("stoch_d"))
    sk_prev = _v(ind.get("stoch_k_prev"))
    sd_prev = _v(ind.get("stoch_d_prev"))

    if sk > 0 and sd > 0:
        if sk < 20 and sk > sd:
            bull += 10
            signals_triggered.append("stochrsi_turning_up_oversold")
        if sk > 80 and sk < sd:
            bear += 10
            signals_triggered.append("stochrsi_turning_down_overbought")
        # Strong reversal: K crossed above D below 30 in last 2 bars
        if sk_prev > 0 and sd_prev > 0:
            if sk > sd and sk_prev <= sd_prev and sk < 30:
                bull += 12
                signals_triggered.append("stochrsi_bull_cross_below30")
                reasoning_parts.append("Stoch RSI bullish cross below 30 — strong reversal")
            if sk < sd and sk_prev >= sd_prev and sk > 70:
                bear += 12
                signals_triggered.append("stochrsi_bear_cross_above70")

    # CCI
    cci = _v(ind.get("cci"))
    if cci != 0:
        if cci > 200:
            bear += 15
            signals_against.append("cci_extremely_overbought")
        elif cci > 100:
            bear += 8
            signals_against.append("cci_overbought")
        elif cci < -200:
            bull += 15
            signals_triggered.append("cci_extremely_oversold")
        elif cci < -100:
            bull += 8
            signals_triggered.append("cci_oversold")

    # Williams %R
    willr = _v(ind.get("willr"), default=None)
    if willr is not None:
        if willr < -80:
            bull += 8
            signals_triggered.append("willr_oversold")
        elif willr > -20:
            bear += 8
            signals_against.append("willr_overbought")

    # Rate of Change
    roc = _v(ind.get("roc"))
    if roc > 3:
        bull += 6
        signals_triggered.append("roc_positive")
    elif roc < -3:
        bear += 6
        signals_triggered.append("roc_negative")

    # ──────────────────────────────────────────────────────────────
    # VOLATILITY SIGNALS
    # ──────────────────────────────────────────────────────────────
    bb_upper   = _v(ind.get("bb_upper"))
    bb_lower   = _v(ind.get("bb_lower"))
    bb_pctb    = _v(ind.get("bb_pctb"), default=None)
    bb_squeeze = ind.get("bb_squeeze") or False
    bb_bw_exp  = ind.get("bb_bw_expanding")

    if not bb_squeeze:
        if bb_lower > 0 and cp <= bb_lower * 1.001:
            bull += 12
            signals_triggered.append("bb_price_at_lower_band")
            reasoning_parts.append("Price at/below BB lower band — mean reversion setup")
        elif bb_upper > 0 and cp >= bb_upper * 0.999:
            bear += 12
            signals_against.append("bb_price_at_upper_band")

        if bb_pctb is not None:
            if bb_pctb < 0.1:
                bull += 18
                signals_triggered.append("bb_deeply_oversold")
                reasoning_parts.append(f"%B={bb_pctb:.2f} deeply oversold")
            elif bb_pctb > 0.9:
                bear += 18
                signals_against.append("bb_deeply_overbought")
                reasoning_parts.append(f"%B={bb_pctb:.2f} deeply overbought")

        if bb_bw_exp:
            net_so_far = bull - bear
            if net_so_far > 0:
                bull += 8
                signals_triggered.append("bb_bandwidth_expanding_bull")
            elif net_so_far < 0:
                bear += 8
                signals_triggered.append("bb_bandwidth_expanding_bear")
    else:
        signals_triggered.append("bb_squeeze_detected")
        reasoning_parts.append("Bollinger Band squeeze — breakout watch")

    # Keltner Channel breakout (especially powerful after squeeze)
    kc_upper = _v(ind.get("kc_upper"))
    kc_lower = _v(ind.get("kc_lower"))
    if kc_upper > 0 and cp > kc_upper and bb_squeeze:
        bull += 15
        signals_triggered.append("kc_breakout_bull")
        reasoning_parts.append("KC breakout above upper channel after BB squeeze")
    elif kc_lower > 0 and cp < kc_lower:
        bear += 15
        signals_triggered.append("kc_breakdown_bear")

    # ATR note
    atr_pct = _v(ind.get("atr_pct"), default=None)
    high_vol_flag = atr_pct is not None and atr_pct > 4
    if high_vol_flag:
        signals_against.append("high_atr_volatility")
        reasoning_parts.append(f"ATR% {atr_pct:.2f}% — high volatility, position size reduced 40%")

    # ──────────────────────────────────────────────────────────────
    # VOLUME SIGNALS
    # ──────────────────────────────────────────────────────────────
    vol_ratio = _v(ind.get("volume_ratio"), default=None)
    price_up = cp > _v(ind.get("prev_close"), default=cp)

    vol_mult = 1.0
    if vol_ratio is not None:
        if vol_ratio < 0.7:
            vol_mult = 0.8
            signals_against.append("low_volume_conviction")
            reasoning_parts.append(f"Volume ratio {vol_ratio:.2f} — low conviction")
        elif vol_ratio >= 2.0:
            if price_up:
                bull += 20
                signals_triggered.append("volume_surge_bull")
                reasoning_parts.append(f"Volume {vol_ratio:.1f}x avg with price up — strong conviction")
            else:
                bear += 20
                signals_triggered.append("volume_surge_bear")
        elif vol_ratio >= 1.5:
            if price_up:
                bull += 12
                signals_triggered.append("volume_confirm_bull")
                reasoning_parts.append(f"Volume {vol_ratio:.1f}x avg confirms upward move")
            else:
                bear += 12
                signals_triggered.append("volume_confirm_bear")

    # Apply low-volume multiplier
    if vol_mult != 1.0:
        bull *= vol_mult
        bear *= vol_mult

    # OBV
    obv_rising = ind.get("obv_rising")
    if obv_rising is True:
        bull += 8
        signals_triggered.append("obv_rising")
    elif obv_rising is False:
        bear += 8
        signals_triggered.append("obv_falling")

    if ind.get("obv_bull_divergence"):
        bull += 10
        signals_triggered.append("obv_bull_divergence")
        reasoning_parts.append("OBV making new highs while price is not — accumulation divergence")
    if ind.get("obv_bear_divergence"):
        bear += 10
        signals_triggered.append("obv_bear_divergence")

    # MFI
    mfi = _v(ind.get("mfi"), default=None)
    if mfi is not None:
        if mfi < 20:
            bull += 10
            signals_triggered.append("mfi_oversold")
        elif mfi > 80:
            bear += 10
            signals_against.append("mfi_overbought")

    # ──────────────────────────────────────────────────────────────
    # SUPPORT / RESISTANCE
    # ──────────────────────────────────────────────────────────────
    def near(price, level, pct=0.003):
        if not level or not price:
            return False
        return abs(price - level) / level <= pct

    P  = _v(ind.get("P"))
    R1 = _v(ind.get("R1"))
    R2 = _v(ind.get("R2"))
    S1 = _v(ind.get("S1"))
    S2 = _v(ind.get("S2"))

    if near(cp, S1):
        bull += 10
        signals_triggered.append("near_s1_support")
    if near(cp, S2):
        bull += 15
        signals_triggered.append("near_s2_strong_support")
    if near(cp, R1):
        bear += 8
        signals_against.append("near_r1_resistance")
        reasoning_parts.append(f"Near R1 resistance at {R1:.2f}")
    if near(cp, R2):
        bear += 12
        signals_against.append("near_r2_strong_resistance")

    # Pivot breakouts (need volume)
    if vol_ratio and vol_ratio > 1.5:
        if R1 > 0 and cp > R1:
            bull += 15
            signals_triggered.append("broke_above_r1_with_volume")
            reasoning_parts.append(f"Broke above R1 ({R1:.2f}) with volume {vol_ratio:.1f}x")
        if S1 > 0 and cp < S1:
            bear += 15
            signals_triggered.append("broke_below_s1_with_volume")

    # 52-week range
    pct_52h = _v(ind.get("pct_from_52wk_high"), default=None)
    pct_52l = _v(ind.get("pct_from_52wk_low"),  default=None)
    wk52_h  = _v(ind.get("wk52_high"))

    if pct_52h is not None:
        if pct_52h >= -2:
            if vol_ratio and vol_ratio > 1.5:
                bull += 20
                signals_triggered.append("breaking_52wk_high")
                reasoning_parts.append("Breaking above 52-week high with volume — major breakout")
            else:
                bear += 10
                signals_against.append("near_52wk_high_resistance")
        if pct_52l is not None and pct_52l <= 2:
            bull += 10
            signals_triggered.append("near_52wk_low_support")

    # ──────────────────────────────────────────────────────────────
    # NEWS SENTIMENT
    # ──────────────────────────────────────────────────────────────
    news = news_sentiment or {}
    avg_pol = _v(news.get("avg_polarity"))
    sec_8k  = news.get("sec_8k_flag") or False
    earn_risk = news.get("earnings_risk") or False

    if avg_pol > 0.5:
        bull += 22
        signals_triggered.append("news_very_positive")
        reasoning_parts.append(f"News sentiment very positive (polarity={avg_pol:.2f})")
    elif avg_pol > 0.3:
        bull += 15
        signals_triggered.append("news_positive")
        reasoning_parts.append(f"News sentiment positive (polarity={avg_pol:.2f})")
    elif avg_pol < -0.5:
        bear += 22
        signals_triggered.append("news_very_negative")
    elif avg_pol < -0.3:
        bear += 15
        signals_triggered.append("news_negative")

    if sec_8k:
        net_so_far = bull - bear
        if net_so_far >= 0:
            bull += 20
        else:
            bear += 20
        signals_triggered.append("sec_8k_catalyst")
        reasoning_parts.append("SEC 8-K filing detected — catalyst amplifier")

    # ──────────────────────────────────────────────────────────────
    # MACRO FILTER
    # ──────────────────────────────────────────────────────────────
    vix       = _v(macro_context.get("vix"), default=15)
    spy_regime = macro_context.get("spy_regime", "bull")
    vix_mult  = macro_context.get("vix_multiplier", 1.0)

    if spy_regime == "caution":
        bull *= 0.80
        reasoning_parts.append("SPY in caution zone — bull signals discounted 20%")
    elif spy_regime == "bear":
        bull *= 0.60
        reasoning_parts.append("SPY bear regime — bull signals discounted 40%")

    # ──────────────────────────────────────────────────────────────
    # FINAL SCORING
    # ──────────────────────────────────────────────────────────────
    bull = max(0, round(bull))
    bear = max(0, round(bear))
    net  = bull - bear

    confidence_raw = net / 100.0
    confidence     = max(-1.0, min(1.0, confidence_raw))

    # Earnings risk: reduce confidence
    if earn_risk:
        confidence *= 0.85
        signals_against.append("earnings_within_3_days")
        reasoning_parts.append("Earnings within 3 days — confidence reduced, binary risk")

    # VIX high fear gates
    if vix > 35 and net > 0:
        return _no_signal(ticker, "vix_extreme_fear_no_longs")

    if vix > 25 and abs(confidence) < 0.80:
        return _no_signal(ticker, "vix_high_below_confidence_threshold")

    # Determine action
    if net > 30 and confidence >= 0.30:
        action = "buy"
    elif net < -30 and abs(confidence) >= 0.30:
        action = "short" if vix < 25 else "sell"
    else:
        action = "hold"

    # ATR for stop/target
    atr    = _v(ind.get("atr"), default=cp * 0.02)
    rr     = 2.5

    if action in ("buy",):
        stop_loss   = round(cp - atr * 1.5, 2)
        take_profit = round(cp + atr * 1.5 * rr, 2)
    elif action in ("short", "sell"):
        stop_loss   = round(cp + atr * 1.5, 2)
        take_profit = round(cp - atr * 1.5 * rr, 2)
    else:
        stop_loss   = None
        take_profit = None

    # Determine strategy hint (strategies.py will classify formally)
    strategy = _pick_strategy_hint(signals_triggered, ind, vol_ratio)

    # Build reasoning string
    reasoning = ". ".join(reasoning_parts[:8]) if reasoning_parts else "No strong directional signals."

    return {
        "ticker": ticker,
        "bull_score": bull,
        "bear_score": bear,
        "net_score": net,
        "signals_triggered": signals_triggered,
        "signals_against": signals_against,
        "strategy": strategy,
        "action": action,
        "confidence": round(abs(confidence), 4),
        "time_horizon": _time_horizon(strategy),
        "reasoning": reasoning,
        "entry_price": round(cp, 2),
        "stop_loss": stop_loss,
        "take_profit": take_profit,
        "risk_reward": rr,
        "atr": round(atr, 4),
        "high_vol_flag": high_vol_flag,
        "earnings_risk": earn_risk,
        "vix": vix,
        "macro_bias": spy_regime,
    }


def _no_signal(ticker: str, reason: str) -> dict:
    return {
        "ticker": ticker,
        "bull_score": 0,
        "bear_score": 0,
        "net_score": 0,
        "signals_triggered": [],
        "signals_against": [reason],
        "strategy": "none",
        "action": "hold",
        "confidence": 0.0,
        "time_horizon": "none",
        "reasoning": reason,
        "entry_price": None,
        "stop_loss": None,
        "take_profit": None,
        "risk_reward": 0,
        "atr": None,
        "high_vol_flag": False,
        "earnings_risk": False,
        "vix": None,
        "macro_bias": "unknown",
    }


def _pick_strategy_hint(signals: list, ind: dict, vol_ratio) -> str:
    sigs = set(signals)
    ema_aligned = "ema_full_bull_alignment" in sigs or "ema_partial_bull_alignment" in sigs
    adx = _v(ind.get("adx"))
    macd_rising = "macd_hist_rising_2bars" in sigs
    vol_confirm = "volume_confirm_bull" in sigs or "volume_surge_bull" in sigs
    squeeze = "bb_squeeze_detected" in sigs
    kc_break = "kc_breakout_bull" in sigs
    news_pos = "news_positive" in sigs or "news_very_positive" in sigs
    r1_break = "broke_above_r1_with_volume" in sigs or "breaking_52wk_high" in sigs
    mean_rev = ("rsi_oversold" in sigs or "bb_deeply_oversold" in sigs or "cci_oversold" in sigs)

    if squeeze and kc_break:
        return "squeeze_breakout"
    if r1_break:
        return "breakout"
    if "broke_below_s1_with_volume" in sigs:
        return "breakdown"
    if ema_aligned and adx > 25 and macd_rising and vol_confirm:
        return "trend_follow"
    if mean_rev:
        return "mean_reversion"
    if news_pos and (ema_aligned or vol_confirm):
        return "news_momentum"
    if ema_aligned:
        return "trend_follow"
    return "mixed"


def _time_horizon(strategy: str) -> str:
    mapping = {
        "trend_follow": "swing",
        "mean_reversion": "scalp",
        "breakout": "swing",
        "breakdown": "swing",
        "squeeze_breakout": "swing",
        "news_momentum": "scalp",
        "mixed": "swing",
        "none": "none",
    }
    return mapping.get(strategy, "swing")
