"""Rules-based decision engine. Pure deterministic Python — no AI API."""

import logging
from typing import Optional

import yfinance as yf

logger = logging.getLogger(__name__)

# ── Fundamental quality cache ──────────────────────────────────────────────────
_fund_cache: dict = {}


def get_velocity_returns(ticker: str, df) -> dict:
    """Compute multi-timeframe returns from a daily Close series."""
    result = {"return_1d": None, "return_5d": None, "return_1m": None, "return_3m": None}
    try:
        if df is None or df.empty:
            return result
        close = df["Close"]
        n = len(close)
        if n >= 2:
            result["return_1d"] = float((close.iloc[-1] - close.iloc[-2]) / close.iloc[-2])
        if n >= 6:
            result["return_5d"] = float((close.iloc[-1] - close.iloc[-6]) / close.iloc[-6])
        if n >= 22:
            result["return_1m"] = float((close.iloc[-1] - close.iloc[-22]) / close.iloc[-22])
        if n >= 64:
            result["return_3m"] = float((close.iloc[-1] - close.iloc[-64]) / close.iloc[-64])
    except Exception as e:
        logger.warning(f"[scorer] velocity returns failed for {ticker}: {e}")
    return result


def _fetch_velocity(ticker: str) -> dict:
    """Fetch 90-day daily data and compute velocity returns."""
    try:
        from bot.data import fetch_daily_bars
        df = fetch_daily_bars(ticker, days=90)
        if df is None or df.empty:
            return {"return_1d": None, "return_5d": None, "return_1m": None, "return_3m": None}
        return get_velocity_returns(ticker, df)
    except Exception as e:
        logger.warning(f"[scorer] _fetch_velocity failed for {ticker}: {e}")
        return {"return_1d": None, "return_5d": None, "return_1m": None, "return_3m": None}


def get_fundamental_quality(ticker: str) -> dict:
    """Fetch yfinance .info and compute a fundamental quality score."""
    if ticker in _fund_cache:
        return _fund_cache[ticker]

    bull_pts = 0
    bear_pts = 0
    no_revenue = False
    revenue_growth = None
    short_pct = None
    institutional_pct = None
    eps_beat = False
    eps_surprise_pct = None
    eps_actual = None

    def _sf(v):
        """Safe float — yfinance occasionally returns strings for numeric fields."""
        if v is None:
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    try:
        info = yf.Ticker(ticker).info or {}

        revenue_growth      = _sf(info.get("revenueGrowth"))
        short_pct           = _sf(info.get("shortPercentOfFloat"))
        institutional_pct   = _sf(info.get("institutionsPercentHeld")
                                   or info.get("institutionPercentHeld"))
        eps_actual          = _sf(info.get("trailingEps"))
        eps_estimate        = _sf(info.get("epsCurrentYear"))
        forward_pe          = _sf(info.get("forwardPE"))
        trailing_pe         = _sf(info.get("trailingPE"))
        total_revenue       = _sf(info.get("totalRevenue"))

        # Revenue growth signals
        if revenue_growth is not None:
            if revenue_growth > 0.40:
                bull_pts += 18   # >40%: +10 base + +8 extra
            elif revenue_growth > 0.20:
                bull_pts += 10   # >20%: +10
            elif revenue_growth < 0:
                bear_pts += 10

        # No revenue guard
        if total_revenue is None or total_revenue == 0:
            no_revenue = True

        # EPS beat
        if eps_actual is not None and eps_estimate is not None:
            if eps_actual > eps_estimate:
                bull_pts += 8
                eps_beat = True
                denom = abs(eps_estimate) if eps_estimate != 0 else 1
                eps_surprise_pct = (eps_actual - eps_estimate) / denom
                if eps_surprise_pct > 0.10:
                    bull_pts += 4

        # PE compression
        if (forward_pe is not None and trailing_pe is not None
                and forward_pe > 0 and trailing_pe > 0
                and forward_pe < trailing_pe):
            bull_pts += 8

        # Institutional ownership
        if institutional_pct is not None:
            if institutional_pct > 0.60:
                bull_pts += 8

        # Short interest
        if short_pct is not None:
            if short_pct < 0.05:
                bull_pts += 5
            elif short_pct > 0.20:
                bear_pts += 15

        # Overvalued with no growth
        if (trailing_pe is not None and trailing_pe > 200
                and (revenue_growth is None or revenue_growth <= 0.05)):
            bear_pts += 10

    except Exception as e:
        logger.warning(f"[scorer] fundamental quality failed for {ticker}: {e}")

    # Determine breakout quality
    rev_ok  = revenue_growth is not None and revenue_growth > 0.20
    inst_ok = institutional_pct is not None and institutional_pct > 0.50
    if rev_ok and eps_beat and inst_ok:
        breakout_quality = "fundamental"
    elif rev_ok or eps_beat:
        breakout_quality = "technical"
    else:
        all_none = (revenue_growth is None and short_pct is None
                    and institutional_pct is None and eps_actual is None)
        breakout_quality = "unknown" if all_none else "technical"

    result = {
        "fund_score":        bull_pts - bear_pts,
        "bull_pts":          bull_pts,
        "bear_pts":          bear_pts,
        "breakout_quality":  breakout_quality,
        "no_revenue":        no_revenue,
        "revenue_growth":    revenue_growth,
        "short_pct":         short_pct,
        "institutional_pct": institutional_pct,
        "eps_beat":          eps_beat,
        "eps_surprise_pct":  eps_surprise_pct,
    }
    _fund_cache[ticker] = result
    return result

# ── Trading thresholds ─────────────────────────────────────────────────────
MIN_NET_SCORE_BUY      = 60    # matches backtest MIN_NET_SCORE; quality gates provide additional filtering
MIN_CONFIDENCE_BUY     = 0.60  # conf = net/100, matches net>=60
MIN_NET_SCORE_SHORT    = 70    # shorts need strong conviction, especially in bull markets
MIN_CONFIDENCE_SHORT   = 0.70  # shorts are riskier

# High-volatility tickers that need extra stop room (4x ATR instead of strategy default)
HIGH_VOLATILITY_TICKERS = {
    "NVDA", "TSLA", "COIN", "MSTR", "SMCI", "PLTR", "AMD", "SOFI", "LI", "MDB",
    "MRNA", "AFRM", "AMC", "PLUG", "GME", "RIVN", "LCID", "ARM",
    "AVGO", "PANW", "CRM", "SNOW", "NET", "DDOG", "CRWD", "ZS", "TEAM",
    # Medical device / diagnostics — event-driven, gap-prone
    "TMDX", "GMED", "PODD", "DXCM", "IRTC", "INSP", "NVST",
    # Small/mid-cap momentum with high ATR
    "SAIA", "AXON", "KTOS", "JOBY", "ACHR", "SPCE",
}


def _v(val, default=0.0):
    """Safe value getter with default."""
    return val if val is not None else default


def score_ticker(
    ticker: str,
    indicators: dict,
    news_sentiment: dict,
    macro_context: dict,
    historical_context: dict = None,
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

    # ── Category score buckets ─────────────────────────────────────────
    # WITHIN a category: signals don't stack — only the strongest fires.
    # ACROSS categories: they sum (each measures a different price dimension).
    # Structure: [bull, bear]
    c_trend      = [0.0, 0.0]   # EMA alignment + ADX baked in
    c_momentum   = [0.0, 0.0]   # MACD (confirms/denies trend direction)
    c_anchor     = [0.0, 0.0]   # PSAR + VWAP + ROC (can stack up to cap of 14)
    c_oscillator = [0.0, 0.0]   # RSI, StochRSI, CCI, Williams %R — take max
    c_vol_struct = [0.0, 0.0]   # Bollinger Bands + Keltner Channel
    c_volume     = [0.0, 0.0]   # Volume ratio + OBV + MFI bonus
    c_structure  = [0.0, 0.0]   # Support/resistance levels + 52wk range

    def _b(cat, pts):
        if pts > cat[0]: cat[0] = pts
    def _r(cat, pts):
        if pts > cat[1]: cat[1] = pts

    # ── TREND — EMA alignment with ADX built into point value ──────────
    e9   = _v(ind.get("ema9"))
    e21  = _v(ind.get("ema21"))
    e50  = _v(ind.get("ema50"))
    e200 = _v(ind.get("ema200"))
    adx  = _v(ind.get("adx"))
    di_plus  = _v(ind.get("adx_di_plus"))
    di_minus = _v(ind.get("adx_di_minus"))

    ema_full_bull = False
    if e9 > 0 and e21 > 0 and e50 > 0 and e200 > 0:
        if e9 > e21 > e50 > e200:
            ema_full_bull = True
            signals_triggered.append("ema_full_bull_alignment")
            reasoning_parts.append("EMA9>EMA21>EMA50>EMA200 full bull alignment")
            pts = 30 if adx > 25 else (24 if adx > 18 else 16)
            _b(c_trend, pts)
        elif e9 < e21 < e50 < e200:
            signals_triggered.append("ema_full_bear_alignment")
            reasoning_parts.append("EMA9<EMA21<EMA50<EMA200 full bear alignment")
            pts = 30 if adx > 25 else (24 if adx > 18 else 16)
            _r(c_trend, pts)
        elif e9 > e21 > e50:
            signals_triggered.append("ema_partial_bull_alignment")
            reasoning_parts.append("EMA9>EMA21>EMA50 partial bull alignment")
            pts = 20 if adx > 22 else 13
            _b(c_trend, pts)
        elif e9 < e21 < e50:
            signals_triggered.append("ema_partial_bear_alignment")
            reasoning_parts.append("EMA9<EMA21<EMA50 partial bear alignment")
            pts = 20 if adx > 22 else 13
            _r(c_trend, pts)
    elif e9 > 0 and e21 > 0 and e50 > 0:
        if e9 > e21 > e50:
            signals_triggered.append("ema_partial_bull_alignment")
            _b(c_trend, 13)
        elif e9 < e21 < e50:
            signals_triggered.append("ema_partial_bear_alignment")
            _r(c_trend, 13)

    # DI+ / DI-: signal only (strategy classifier uses these); no separate score
    # — ADX is already incorporated in the EMA point values above.
    if adx > 25 and di_plus > 0 and di_minus > 0:
        if di_plus > di_minus:
            signals_triggered.append("di_plus_dominant")
        else:
            signals_triggered.append("di_minus_dominant")

    # ── MOMENTUM — MACD (strongest tier wins) ──────────────────────────
    macd_hist       = _v(ind.get("macd_hist"))
    macd_hist_prev1 = _v(ind.get("macd_hist_prev1"))
    macd_hist_prev2 = _v(ind.get("macd_hist_prev2"))
    macd_bull_cross = ind.get("macd_bull_cross") or False
    macd_bear_cross = ind.get("macd_bear_cross") or False

    if macd_hist > 0 and macd_hist_prev1 > 0 and macd_hist > macd_hist_prev1 and macd_hist_prev1 > macd_hist_prev2:
        signals_triggered.append("macd_hist_rising_2bars")
        reasoning_parts.append("MACD histogram positive and rising 2+ bars")
        _b(c_momentum, 20)
    elif macd_bull_cross:
        signals_triggered.append("macd_bullish_cross")
        reasoning_parts.append("MACD bullish crossover (last 3 bars)")
        _b(c_momentum, 18)
    elif macd_hist > 0 and macd_hist > macd_hist_prev1:
        _b(c_momentum, 12)
    elif macd_hist > 0:
        _b(c_momentum, 6)

    if macd_hist < 0 and macd_hist_prev1 < 0 and macd_hist < macd_hist_prev1 and macd_hist_prev1 < macd_hist_prev2:
        signals_triggered.append("macd_hist_falling_2bars")
        _r(c_momentum, 20)
    elif macd_bear_cross:
        signals_triggered.append("macd_bearish_cross")
        _r(c_momentum, 18)
    elif macd_hist < 0 and macd_hist < macd_hist_prev1:
        _r(c_momentum, 12)
    elif macd_hist < 0:
        _r(c_momentum, 6)

    # Fading momentum: reduce this category's contribution
    if macd_hist > 0 and macd_hist_prev1 > 0 and macd_hist < macd_hist_prev1:
        c_momentum[0] = max(0.0, c_momentum[0] - 5)
        signals_against.append("macd_hist_fading_bull")
    if macd_hist < 0 and macd_hist_prev1 < 0 and macd_hist > macd_hist_prev1:
        c_momentum[1] = max(0.0, c_momentum[1] - 5)
        signals_against.append("macd_hist_fading_bear")

    # ── ANCHOR — PSAR + VWAP + ROC (stacks up to cap 14) ──────────────
    anchor_bull = 0.0
    anchor_bear = 0.0

    psar_bullish = ind.get("psar_bullish")
    if psar_bullish is True:
        anchor_bull += 8
        signals_triggered.append("psar_bullish")
    elif psar_bullish is False:
        anchor_bear += 8
        signals_triggered.append("psar_bearish")

    vwap = _v(ind.get("vwap"))
    if vwap > 0 and cp > 0:
        vwap_diff_pct = (cp - vwap) / vwap * 100
        if vwap_diff_pct > 0.3:
            anchor_bull += 6
            signals_triggered.append("price_above_vwap")
            reasoning_parts.append(f"Price {vwap_diff_pct:.2f}% above VWAP")
        elif vwap_diff_pct < -0.3:
            anchor_bear += 6
            signals_triggered.append("price_below_vwap")
            reasoning_parts.append(f"Price {abs(vwap_diff_pct):.2f}% below VWAP")

    roc = _v(ind.get("roc"))
    if roc > 3:
        anchor_bull = min(anchor_bull + 3, 14)
        signals_triggered.append("roc_positive")
    elif roc < -3:
        anchor_bear = min(anchor_bear + 3, 14)
        signals_triggered.append("roc_negative")

    c_anchor[0] = min(anchor_bull, 14)
    c_anchor[1] = min(anchor_bear, 14)

    # ── OSCILLATOR — RSI, StochRSI, CCI, Williams %R (take max) ────────
    rsi = _v(ind.get("rsi"))
    osc_bull = 0.0
    osc_bear = 0.0

    if rsi > 0:
        if rsi < 20:
            osc_bull = max(osc_bull, 20)
            signals_triggered.append("rsi_extremely_oversold")
            reasoning_parts.append(f"RSI {rsi:.1f} extremely oversold")
        elif rsi < 30:
            osc_bull = max(osc_bull, 15)
            signals_triggered.append("rsi_oversold")
            reasoning_parts.append(f"RSI {rsi:.1f} oversold")
        elif rsi < 40:
            osc_bear = max(osc_bear, 5)
            signals_triggered.append("rsi_weak")
        elif 40 < rsi <= 50:
            osc_bear = max(osc_bear, 8)
            signals_triggered.append("rsi_bearish_momentum")
        elif 50 < rsi <= 60:
            osc_bull = max(osc_bull, 8)
            signals_triggered.append("rsi_healthy_momentum")
            reasoning_parts.append(f"RSI {rsi:.1f} healthy bull momentum")
        elif 60 < rsi <= 70:
            osc_bull = max(osc_bull, 5)
            signals_triggered.append("rsi_strong")
        elif rsi > 80:
            osc_bear = max(osc_bear, 20)
            signals_against.append("rsi_extremely_overbought")
            reasoning_parts.append(f"RSI {rsi:.1f} extremely overbought")
        elif rsi > 70:
            osc_bear = max(osc_bear, 15)
            signals_against.append("rsi_overbought")
            reasoning_parts.append(f"RSI {rsi:.1f} overbought")

    sk = _v(ind.get("stoch_k"))
    sd = _v(ind.get("stoch_d"))
    sk_prev = _v(ind.get("stoch_k_prev"))
    sd_prev = _v(ind.get("stoch_d_prev"))

    if sk > 0 and sd > 0:
        if sk < 20 and sk > sd:
            osc_bull = max(osc_bull, 12)
            signals_triggered.append("stochrsi_turning_up_oversold")
        if sk > 80 and sk < sd:
            osc_bear = max(osc_bear, 12)
            signals_triggered.append("stochrsi_turning_down_overbought")
        if sk_prev > 0 and sd_prev > 0:
            if sk > sd and sk_prev <= sd_prev and sk < 30:
                osc_bull = max(osc_bull, 14)
                signals_triggered.append("stochrsi_bull_cross_below30")
                reasoning_parts.append("Stoch RSI bullish cross below 30 — strong reversal")
            if sk < sd and sk_prev >= sd_prev and sk > 70:
                osc_bear = max(osc_bear, 14)
                signals_triggered.append("stochrsi_bear_cross_above70")

    cci = _v(ind.get("cci"))
    if cci != 0:
        if cci > 200:
            osc_bear = max(osc_bear, 15)
            signals_against.append("cci_extremely_overbought")
        elif cci > 100:
            osc_bear = max(osc_bear, 8)
            signals_against.append("cci_overbought")
        elif cci < -200:
            osc_bull = max(osc_bull, 15)
            signals_triggered.append("cci_extremely_oversold")
        elif cci < -100:
            osc_bull = max(osc_bull, 8)
            signals_triggered.append("cci_oversold")

    willr = _v(ind.get("willr"), default=None)
    if willr is not None:
        if willr < -80:
            osc_bull = max(osc_bull, 8)
            signals_triggered.append("willr_oversold")
        elif willr > -20:
            osc_bear = max(osc_bear, 8)
            signals_against.append("willr_overbought")

    c_oscillator[0] = osc_bull
    c_oscillator[1] = osc_bear

    # ── VOLATILITY STRUCTURE — BB + KC (exclusive) ─────────────────────
    bb_upper   = _v(ind.get("bb_upper"))
    bb_lower   = _v(ind.get("bb_lower"))
    bb_pctb    = _v(ind.get("bb_pctb"), default=None)
    bb_squeeze = ind.get("bb_squeeze") or False
    bb_bw_exp  = ind.get("bb_bw_expanding")

    vs_bull = 0.0
    vs_bear = 0.0
    if not bb_squeeze:
        if bb_pctb is not None:
            if bb_pctb < 0.10:
                vs_bull = 20
                signals_triggered.append("bb_deeply_oversold")
                reasoning_parts.append(f"%B={bb_pctb:.2f} deeply oversold")
            elif bb_lower > 0 and cp <= bb_lower * 1.001:
                vs_bull = 12
                signals_triggered.append("bb_price_at_lower_band")
                reasoning_parts.append("Price at/below BB lower band — mean reversion setup")
            elif bb_pctb > 0.90:
                vs_bear = 20
                signals_against.append("bb_deeply_overbought")
                reasoning_parts.append(f"%B={bb_pctb:.2f} deeply overbought")
            elif bb_upper > 0 and cp >= bb_upper * 0.999:
                vs_bear = 12
                signals_against.append("bb_price_at_upper_band")
        elif bb_lower > 0 and cp <= bb_lower * 1.001:
            vs_bull = 12
            signals_triggered.append("bb_price_at_lower_band")
            reasoning_parts.append("Price at/below BB lower band — mean reversion setup")
        elif bb_upper > 0 and cp >= bb_upper * 0.999:
            vs_bear = 12
            signals_against.append("bb_price_at_upper_band")

        if bb_bw_exp:
            if vs_bull > 0:
                vs_bull = min(vs_bull + 5, 20)
                signals_triggered.append("bb_bandwidth_expanding_bull")
            elif vs_bear > 0:
                vs_bear = min(vs_bear + 5, 20)
                signals_triggered.append("bb_bandwidth_expanding_bear")
    else:
        signals_triggered.append("bb_squeeze_detected")
        reasoning_parts.append("Bollinger Band squeeze — breakout watch")
        kc_upper = _v(ind.get("kc_upper"))
        kc_lower = _v(ind.get("kc_lower"))
        if kc_upper > 0 and cp > kc_upper:
            vs_bull = 18
            signals_triggered.append("kc_breakout_bull")
            reasoning_parts.append("KC breakout above upper channel after BB squeeze")
        elif kc_lower > 0 and cp < kc_lower:
            vs_bear = 18
            signals_triggered.append("kc_breakdown_bear")

    c_vol_struct[0] = vs_bull
    c_vol_struct[1] = vs_bear

    # ATR note — flag by ATR% threshold OR known gap-prone tickers
    atr_pct = _v(ind.get("atr_pct"), default=None)
    high_vol_flag = (
        (atr_pct is not None and atr_pct > 4)
        or ticker.upper() in HIGH_VOLATILITY_TICKERS
    )
    if high_vol_flag:
        signals_against.append("high_atr_volatility")
        reasoning_parts.append(f"ATR% {atr_pct:.2f}% — high volatility, position size reduced 40%")

    # ── VOLUME — ratio (exclusive tiers) + OBV/MFI bonus ───────────────
    vol_ratio = _v(ind.get("volume_ratio"), default=None)
    price_up  = cp > _v(ind.get("prev_close"), default=cp)

    vol_bull = 0.0
    vol_bear = 0.0
    if vol_ratio is not None:
        if vol_ratio >= 3.0:
            if price_up:
                vol_bull = 22
                signals_triggered.append("volume_surge_extreme_bull")
                reasoning_parts.append(f"Volume {vol_ratio:.1f}x avg — extreme surge with price up")
            else:
                vol_bear = 22
                signals_triggered.append("volume_surge_extreme_bear")
        elif vol_ratio >= 2.0:
            if price_up:
                vol_bull = 16
                signals_triggered.append("volume_surge_bull")
                reasoning_parts.append(f"Volume {vol_ratio:.1f}x avg with price up — strong conviction")
            else:
                vol_bear = 16
                signals_triggered.append("volume_surge_bear")
        elif vol_ratio >= 1.5:
            if price_up:
                vol_bull = 10
                signals_triggered.append("volume_confirm_bull")
                reasoning_parts.append(f"Volume {vol_ratio:.1f}x avg confirms upward move")
            else:
                vol_bear = 10
                signals_triggered.append("volume_confirm_bear")
        # vol_ratio < 0.7: no conviction — zero volume contribution

    # OBV bonus (independent from vol_ratio — accumulation vs raw volume)
    obv_rising = ind.get("obv_rising")
    if obv_rising is True:
        vol_bull += 5
        signals_triggered.append("obv_rising")
    elif obv_rising is False:
        vol_bear += 5
        signals_triggered.append("obv_falling")

    if ind.get("obv_bull_divergence"):
        vol_bull += 5
        signals_triggered.append("obv_bull_divergence")
        reasoning_parts.append("OBV making new highs while price is not — accumulation divergence")
    if ind.get("obv_bear_divergence"):
        vol_bear += 5
        signals_triggered.append("obv_bear_divergence")

    # MFI bonus
    mfi = _v(ind.get("mfi"), default=None)
    if mfi is not None:
        if mfi < 20:
            vol_bull += 4
            signals_triggered.append("mfi_oversold")
        elif mfi > 80:
            vol_bear += 4
            signals_against.append("mfi_overbought")

    c_volume[0] = min(vol_bull, 25)
    c_volume[1] = min(vol_bear, 25)

    # ── STRUCTURE — S/R levels + 52wk range (take max) ─────────────────
    def near(price, level, pct=0.003):
        if not level or not price:
            return False
        return abs(price - level) / level <= pct

    R1 = _v(ind.get("R1"))
    R2 = _v(ind.get("R2"))
    S1 = _v(ind.get("S1"))
    S2 = _v(ind.get("S2"))

    struct_bull = 0.0
    struct_bear = 0.0

    if near(cp, S2):
        struct_bull = max(struct_bull, 15)
        signals_triggered.append("near_s2_strong_support")
    if near(cp, S1):
        struct_bull = max(struct_bull, 10)
        signals_triggered.append("near_s1_support")
    if near(cp, R1):
        struct_bear = max(struct_bear, 8)
        signals_against.append("near_r1_resistance")
        reasoning_parts.append(f"Near R1 resistance at {R1:.2f}")
    if near(cp, R2):
        struct_bear = max(struct_bear, 12)
        signals_against.append("near_r2_strong_resistance")

    if vol_ratio and vol_ratio > 1.5:
        if R1 > 0 and cp > R1:
            struct_bull = max(struct_bull, 15)
            signals_triggered.append("broke_above_r1_with_volume")
            reasoning_parts.append(f"Broke above R1 ({R1:.2f}) with volume {vol_ratio:.1f}x")
        if S1 > 0 and cp < S1:
            struct_bear = max(struct_bear, 15)
            signals_triggered.append("broke_below_s1_with_volume")

    pct_52h = _v(ind.get("pct_from_52wk_high"), default=None)
    pct_52l = _v(ind.get("pct_from_52wk_low"),  default=None)

    if pct_52h is not None:
        if pct_52h >= -2:
            if vol_ratio and vol_ratio > 1.5:
                struct_bull = max(struct_bull, 20)
                signals_triggered.append("breaking_52wk_high")
                reasoning_parts.append("Breaking above 52-week high with volume — major breakout")
            else:
                struct_bear = max(struct_bear, 10)
                signals_against.append("near_52wk_high_resistance")
        if pct_52l is not None and pct_52l <= 2:
            struct_bull = max(struct_bull, 10)
            signals_triggered.append("near_52wk_low_support")

    c_structure[0] = struct_bull
    c_structure[1] = struct_bear

    # ── Sum categories → bull / bear ───────────────────────────────────
    bull = (c_trend[0] + c_momentum[0] + c_anchor[0] + c_oscillator[0]
            + c_vol_struct[0] + c_volume[0] + c_structure[0])
    bear = (c_trend[1] + c_momentum[1] + c_anchor[1] + c_oscillator[1]
            + c_vol_struct[1] + c_volume[1] + c_structure[1])

    # ── Short filter: price above EMA200 ───────────────────────────────
    if e200 > 0 and cp > e200:
        bear = max(0, bear - 20)
        signals_against.append("above_ema200_short_penalty")

    # ──────────────────────────────────────────────────────────────
    # NEWS SENTIMENT
    # ──────────────────────────────────────────────────────────────
    news = news_sentiment or {}
    avg_pol   = _v(news.get("avg_polarity"))
    sec_8k    = news.get("sec_8k_flag") or False
    _earn_raw = news.get("earnings_risk") or {}
    # Support both old bool format and new dict format from _check_earnings_proximity
    if isinstance(_earn_raw, dict):
        earn_risk_level = _earn_raw.get("risk_level", "clear")
        earn_risk       = earn_risk_level == "block"
        earn_warn       = earn_risk_level in ("warn", "warn3")
    else:
        earn_risk  = bool(_earn_raw)
        earn_warn  = False
        earn_risk_level = "block" if earn_risk else "clear"

    # Earnings warn: reduce confidence at scoring time
    # earn_risk_level: "block" (within 1 day), "warn3" (within 3 days), "warn" (within 7 days)
    if earn_warn:
        signals_against.append("earnings_proximity")

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

    # Keyword amplifier boosts from news
    bull += news.get("bull_keyword_boost", 0)
    bear += news.get("bear_keyword_boost", 0)
    if news.get("bull_keyword_boost", 0) > 0:
        signals_triggered.append("keyword_bull_boost")
    if news.get("bear_keyword_boost", 0) > 0:
        signals_triggered.append("keyword_bear_boost")

    # ──────────────────────────────────────────────────────────────
    # MACRO FILTER
    # ──────────────────────────────────────────────────────────────
    vix        = _v(macro_context.get("vix"), default=15)
    spy_regime = macro_context.get("spy_regime", "bull")

    # In a confirmed bull market, heavily discount bear signals — shorts rarely work
    if spy_regime == "bull":
        bear *= 0.50
        reasoning_parts.append("SPY bull regime — bear signals discounted 50%")

    # ──────────────────────────────────────────────────────────────
    # FINAL SCORING
    # ──────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────
    # INTRADAY 15-MIN SIGNALS
    # ──────────────────────────────────────────────────────────────
    intraday_vwap    = _v(ind.get("intraday_vwap"))
    intraday_rsi     = _v(ind.get("intraday_rsi"))
    intraday_vs_vwap = _v(ind.get("intraday_vs_vwap"))

    if intraday_vwap > 0 and cp > 0:
        if intraday_vs_vwap > 0:
            bull += 5
            signals_triggered.append("price_above_intraday_vwap")
        elif intraday_vs_vwap < 0:
            bear += 5
            signals_triggered.append("price_below_intraday_vwap")

    if intraday_rsi > 0:
        if intraday_rsi < 35:
            bull += 8
            signals_triggered.append("intraday_rsi_oversold")
        elif intraday_rsi > 65:
            bear += 8
            signals_triggered.append("intraday_rsi_overbought")

    # ──────────────────────────────────────────────────────────────
    # FUNDAMENTAL QUALITY — add bull/bear pts before net calculation
    # ──────────────────────────────────────────────────────────────
    fq = get_fundamental_quality(ticker)
    bull += fq["bull_pts"]
    bear += fq["bear_pts"]

    bull = max(0, round(bull))
    bear = max(0, round(bear))
    net  = bull - bear

    confidence_raw = net / 100.0
    confidence     = max(-1.0, min(1.0, confidence_raw))

    # Earnings warn: reduce confidence based on proximity
    # earn_risk_level: "block"=within 1 day, "warn3"=within 3 days, "warn"=within 7 days
    if earn_risk_level == "warn3":
        confidence = max(-1.0, min(1.0, confidence - 0.30))
        reasoning_parts.append("Earnings within 3 days — confidence reduced 0.30 (heavy penalty)")
    elif earn_warn:
        confidence = max(-1.0, min(1.0, confidence - 0.15))
        reasoning_parts.append("Earnings within 7 days — confidence reduced 0.15 (light warning)")

    # Earnings risk: block entry entirely — binary outcome, not tradeable (within 1 day)
    if earn_risk:
        return _no_signal(ticker, "earnings_within_1_day")

    # ──────────────────────────────────────────────────────────────
    # VELOCITY RETURNS + HYPE DETECTION
    # ──────────────────────────────────────────────────────────────
    # If the backtest pre-computed returns from the historical slice, use them directly
    # and skip the live network fetch (eliminates forward-looking bias in replay).
    if indicators.get("return_1d") is not None:
        vel = {
            "return_1d": indicators.get("return_1d"),
            "return_5d": indicators.get("return_5d"),
            "return_1m": indicators.get("return_1m"),
            "return_3m": indicators.get("return_3m"),
        }
    else:
        vel = _fetch_velocity(ticker)
    r1d = vel.get("return_1d") or 0.0
    r5d = vel.get("return_5d") or 0.0
    r1m = vel.get("return_1m") or 0.0
    r3m = vel.get("return_3m") or 0.0

    # Import hype_penalty locally to avoid circular import
    try:
        from bot.news import hype_penalty as _hype_penalty_fn
        hype = _hype_penalty_fn(news.get("top_headlines", []), ticker)
    except Exception as e:
        logger.warning(f"[scorer] hype_penalty import/call failed for {ticker}: {e}")
        hype = {
            "hype_penalty": 0.0, "catalyst_boost": 0.0,
            "hype_signals": [], "catalyst_signals": [], "net_confidence_adj": 0.0,
        }

    # Velocity penalty
    vel_penalty = 0.0
    _earn_raw_dict = news.get("earnings_risk") or {}
    _earn_risk_level = _earn_raw_dict.get("risk_level", "clear") if isinstance(_earn_raw_dict, dict) else ("block" if _earn_raw_dict else "clear")
    has_earnings_event = _earn_risk_level in ("block", "warn", "warn3") or fq.get("eps_beat")

    # 1d penalties (skip if earnings event)
    if not has_earnings_event:
        if r1d > 0.25:
            vel_penalty += 0.20
        elif r1d > 0.15:
            vel_penalty += 0.10

    # 5d penalties
    if r5d > 0.40:
        vel_penalty += 0.25
        signals_against.append("5d_severely_extended")
    elif r5d > 0.30:
        vel_penalty += 0.15
    elif r5d > 0.20:
        vel_penalty += 0.08
        signals_against.append("5d_extended")

    # 1m penalties
    if r1m > 0.50:
        vel_penalty += 0.18
    elif r1m > 0.35:
        vel_penalty += 0.10

    # 3m penalties (waive if revenue_growth > 20% or earnings beat)
    waive_3m = (fq.get("revenue_growth") or 0) > 0.20 or fq.get("eps_beat")
    if not waive_3m:
        if r3m > 1.00:
            vel_penalty += 0.20
        elif r3m > 0.60:
            vel_penalty += 0.10

    # Exception: waive/halve if one big news day (1d > 50% of 5d run)
    if r5d and abs(r1d) > abs(r5d) * 0.50 and has_earnings_event:
        vel_penalty *= 0.0   # full waive
    elif r5d and abs(r1d) > abs(r5d) * 0.50:
        vel_penalty *= 0.5   # halve

    # breakout_quality refinement with hype
    bq = fq.get("breakout_quality", "unknown")
    if hype.get("hype_penalty", 0) > 0.10:
        bq = "hype"

    # Velocity penalty multiplier by breakout quality
    if bq == "fundamental":
        vel_penalty *= 0.5
    elif bq == "hype":
        vel_penalty *= 2.0
    vel_penalty = min(vel_penalty, 0.45)  # hard cap

    total_conf_adj = hype.get("net_confidence_adj", 0) - vel_penalty

    # Hype signals → signals_against; catalyst signals → signals_triggered
    for sig in hype.get("hype_signals", []):
        signals_against.append(f"hype:{sig}")
    for sig in hype.get("catalyst_signals", []):
        signals_triggered.append(f"catalyst:{sig}")

    # Apply confidence adjustments
    confidence = max(0.0, confidence + total_conf_adj)
    if fq.get("no_revenue"):
        confidence = min(confidence, 0.65)

    logger.info(
        f"[{ticker}] velocity: 1d={r1d:.1%} 5d={r5d:.1%} 1m={r1m:.1%} 3m={r3m:.1%} "
        f"vel_penalty={vel_penalty:.2f} hype_penalty={hype.get('hype_penalty', 0):.2f} bq={bq}"
    )

    # ── Historical context (multi-day progression) ────────────────────────
    if historical_context:
        try:
            maturity  = historical_context.get("maturity_label", "none")
            days_conf = historical_context.get("days_of_confluence", 0)
            bull_adj  = 0

            if maturity == "strong":
                bull_adj = 15
                signals_triggered.append("3day_confluence_strong")
                reasoning_parts.append(
                    f"3-day confluence strong ({days_conf}/4 signals building)"
                )
            elif maturity == "developing":
                bull_adj = 8
                signals_triggered.append("2day_confluence_building")
                reasoning_parts.append(
                    f"Setup building over 2+ days ({days_conf}/4 signals)"
                )
            elif maturity == "weak":
                bull_adj = -5
                signals_against.append("setup_immature")
            else:  # "none"
                bull_adj = -10
                signals_against.append("single_day_spike_no_history")
                reasoning_parts.append(
                    "Single-day spike — no multi-day confirmation"
                )

            bull = max(0, bull + bull_adj)
            net  = bull - bear
            # Re-derive confidence with the updated net so velocity/hype adj
            # anchors correctly for the trigger multiplier
            confidence_raw = net / 100.0
            confidence = max(0.0, min(1.0, confidence_raw) + total_conf_adj)
            if fq.get("no_revenue"):
                confidence = min(confidence, 0.65)

            logger.info(
                f"[{ticker}] hist_ctx: maturity={maturity} ({days_conf}/4) "
                f"bull_adj={bull_adj:+d}  net={net}"
            )
        except Exception as _hc_err:
            logger.warning(f"[{ticker}] hist_ctx scoring failed: {_hc_err}")

    # ── Entry Trigger Multiplier ───────────────────────────────────────────
    try:
        triggers = ind.get("entry_triggers") or {}
        fresh_count = triggers.get("fresh_trigger_count", 0)
        trigger_names = triggers.get("fresh_trigger_names", [])

        # Only bullish triggers should drive the bull-boost; bearish triggers in
        # fresh_trigger_names (e.g. ema9_just_crossed_ema21_bearish) must not count.
        _BEARISH_TRIGGER_KEYWORDS = ("bearish", "bear", "broke_s1", "crossed_70_down")
        bullish_trigger_names = [n for n in trigger_names
                                  if not any(kw in n for kw in _BEARISH_TRIGGER_KEYWORDS)]
        bullish_fresh_count = len(bullish_trigger_names)

        if bullish_fresh_count >= 1:
            net = round(net * 1.25)
            bull = round(bull * 1.25)
            for name in trigger_names:
                signals_triggered.append(f"{name}_fresh")
            logger.info(
                f"[{ticker}] FRESH TRIGGER x{bullish_fresh_count} bullish "
                f"({fresh_count} total): {bullish_trigger_names} → net boosted to {net}"
            )

            # Don't let fresh triggers override a strong hype/velocity signal
            if total_conf_adj < -0.15:
                net = round(net / 1.25)
                bull = round(bull / 1.25)
                logger.info(f"[{ticker}] Hype override: trigger boost cancelled due to velocity penalty {total_conf_adj:.2f}")
        else:
            # Mild penalty for no same-day trigger — ongoing trends don't fire fresh
            # crossovers every day but are still valid entries. 0.65 was a near-block
            # (needed raw ≥108), 0.80 still too harsh (raw ≥88). 0.90 requires raw ≥73
            # for min=65 (too strict for late-stage trends). 0.95 requires raw ≥64.
            net = round(net * 0.95)
            bull = round(bull * 0.95)
            signals_against.append("no_fresh_trigger")
            logger.info(
                f"[{ticker}] NO fresh bullish triggers "
                f"(had {fresh_count} total, {fresh_count - bullish_fresh_count} bearish-only) "
                f"→ net discounted to {net}"
            )

        # Recompute confidence after trigger adjustment
        confidence_raw = net / 100.0
        confidence = max(-1.0, min(1.0, confidence_raw))
        # Re-apply velocity/hype confidence adjustments
        confidence = max(0.0, confidence + total_conf_adj)
        if fq.get("no_revenue"):
            confidence = min(confidence, 0.65)
    except Exception as e:
        logger.warning(f"[{ticker}] trigger multiplier failed: {e}")

    # VIX high fear gates
    if vix > 35 and net > 0:
        return _no_signal(ticker, "vix_extreme_fear_no_longs")
    if vix > 25 and abs(confidence) < 0.80:
        return _no_signal(ticker, "vix_high_below_confidence_threshold")

    # ── Intraday move filter — don't chase already-extended stocks ───────────
    intraday_move_pct = _v(ind.get("intraday_move_pct"), default=0.0)
    INTRADAY_HARD_BLOCK = 15.0   # >15% move → no buy regardless of score
    INTRADAY_HIGH_BAR   = 10.0   # >10% move → require net>=90 to buy
    INTRADAY_HIGH_NET   = 90

    if intraday_move_pct > INTRADAY_HARD_BLOCK:
        return _no_signal(ticker, "intraday_move_too_large")

    effective_min_net = MIN_NET_SCORE_BUY
    if intraday_move_pct > INTRADAY_HIGH_BAR:
        effective_min_net = INTRADAY_HIGH_NET
        signals_against.append(
            f"intraday_move_{intraday_move_pct:.1f}pct_requires_net{INTRADAY_HIGH_NET}"
        )

    # ── Determine action with raised thresholds ────────────────────────────
    if net >= effective_min_net and confidence >= MIN_CONFIDENCE_BUY:
        action = "buy"
    elif net <= -MIN_NET_SCORE_SHORT and abs(confidence) >= MIN_CONFIDENCE_SHORT:
        action = "short" if vix < 25 else "sell"
    else:
        action = "hold"

    # Short selling disabled — requires dedicated short-specific indicator set
    # All short signals are converted to hold
    if action == "short":
        action = "hold"
        signals_against.append("shorts_disabled")

    # ── Short-specific filters (after action is tentatively set) ───────────
    if action in ("short", "sell"):
        # Hard block: never short in a confirmed bull market — it's fighting the tide
        if spy_regime == "bull":
            action = "hold"
            signals_against.append("short_blocked_bull_market")
            logger.info(f"[scorer] {ticker}: short blocked — SPY in bull regime")
        else:
            # Must have at least one extreme condition to justify a short
            short_extreme = (
                (e50 > 0 and cp < e50) or       # price already below EMA50
                (rsi > 75) or                    # RSI deeply overbought
                (bb_pctb is not None and bb_pctb > 0.95)  # BB deeply overbought
            )
            if not short_extreme:
                action = "hold"
                signals_against.append("short_blocked_no_extreme_condition")
                logger.info(f"[scorer] {ticker}: short blocked — no extreme overbought condition present")
            elif adx > 30 and di_plus > 0 and di_plus > di_minus:
                # Strong confirmed uptrend — never short into it
                action = "hold"
                signals_against.append("short_blocked")
                reasoning_parts.append(
                    f"Short blocked: ADX {adx:.1f}>30 with +DI {di_plus:.1f}>-DI {di_minus:.1f} (strong uptrend)"
                )
                logger.info(
                    f"[scorer] {ticker}: short blocked — strong uptrend "
                    f"ADX={adx:.1f} +DI={di_plus:.1f} > -DI={di_minus:.1f}"
                )

    # ── Stops and targets ─────────────────────────────────────────────────
    atr = _v(ind.get("atr"), default=cp * 0.02)
    rr  = 2.5

    # Always compute stops/targets — for holds these represent "if you were to enter"
    if action in ("short", "sell"):
        stop_loss   = round(cp + atr * 1.5, 2)
        take_profit = round(cp - atr * 1.5 * rr, 2)
    else:
        # buy or hold — long-side levels
        stop_loss   = round(cp - atr * 1.5, 2)
        take_profit = round(cp + atr * 1.5 * rr, 2)

    strategy  = _pick_strategy_hint(signals_triggered, ind, vol_ratio, ema_full_bull)
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
        "earnings_warn": earn_warn,
        "vix": vix,
        "macro_bias": spy_regime,
        "ema_full_bull": ema_full_bull,
        # Multi-timeframe velocity
        "return_1d":               round(r1d, 4) if r1d else None,
        "return_5d":               round(r5d, 4) if r5d else None,
        "return_1m":               round(r1m, 4) if r1m else None,
        "return_3m":               round(r3m, 4) if r3m else None,
        "velocity_penalty_applied": round(vel_penalty, 4),
        # Fundamental quality
        "fundamental_score":        fq.get("fund_score", 0),
        # Hype detection
        "hype_penalty_applied":     round(hype.get("hype_penalty", 0), 4),
        "breakout_quality":         bq,
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
        "ema_full_bull": False,
        "return_1d": None,
        "return_5d": None,
        "return_1m": None,
        "return_3m": None,
        "velocity_penalty_applied": 0.0,
        "fundamental_score": 0,
        "hype_penalty_applied": 0.0,
        "breakout_quality": "unknown",
    }


def _pick_strategy_hint(signals: list, ind: dict, vol_ratio, ema_full_bull: bool) -> str:
    sigs = set(signals)
    ema_aligned = "ema_full_bull_alignment" in sigs or "ema_partial_bull_alignment" in sigs
    adx         = _v(ind.get("adx"))
    macd_hist   = _v(ind.get("macd_hist"))
    squeeze     = "bb_squeeze_detected" in sigs
    kc_break    = "kc_breakout_bull" in sigs
    news_pos    = "news_positive" in sigs or "news_very_positive" in sigs
    r1_break    = "broke_above_r1_with_volume" in sigs or "breaking_52wk_high" in sigs
    vol_confirm = "volume_confirm_bull" in sigs or "volume_surge_bull" in sigs
    mean_rev    = "rsi_oversold" in sigs or "bb_deeply_oversold" in sigs or "cci_oversold" in sigs

    if squeeze and kc_break:
        return "squeeze_breakout"
    if r1_break:
        return "breakout"
    if "broke_below_s1_with_volume" in sigs:
        return "breakdown"
    if ema_aligned and adx > 22 and macd_hist > 0 and vol_confirm:
        return "trend_follow"
    if mean_rev and not ema_full_bull:   # never assign mean_rev to a full-bull-aligned stock
        return "mean_reversion"
    if news_pos and (ema_aligned or vol_confirm):
        return "news_momentum"
    if ema_aligned:
        return "trend_follow"
    return "mixed"


def _time_horizon(strategy: str) -> str:
    return {
        "trend_follow":    "swing",
        "mean_reversion":  "scalp",
        "breakout":        "swing",
        "breakdown":       "swing",
        "squeeze_breakout":"swing",
        "news_momentum":   "scalp",
        "mixed":           "swing",
        "none":            "none",
    }.get(strategy, "swing")  # default to swing — no long-term position holds
