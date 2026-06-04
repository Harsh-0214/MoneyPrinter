"""Technical indicator calculations using yfinance + ta library."""

import json
import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import ta.trend as ta_trend
import ta.momentum as ta_momentum
import ta.volatility as ta_volatility
import ta.volume as ta_volume
from bot.data import fetch_daily_bars, fetch_intraday_bars, fetch_snapshot, fetch_vix

logger = logging.getLogger(__name__)

_cache: dict = {}

_PREV_IND_PATH = Path(__file__).parent.parent / "data" / "prev_indicators.json"


def load_prev_indicators() -> dict:
    """Load previous-cycle indicators from data/prev_indicators.json."""
    try:
        if _PREV_IND_PATH.exists():
            with open(_PREV_IND_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.debug(f"[indicators] load_prev_indicators failed: {e}")
    return {}


def save_prev_indicators(data: dict) -> None:
    """Save current indicators to data/prev_indicators.json."""
    try:
        _PREV_IND_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_PREV_IND_PATH, "w") as f:
            json.dump(data, f)
    except Exception as e:
        logger.warning(f"[indicators] save_prev_indicators failed: {e}")


def compute_entry_triggers(current_ind: dict, prev_ind: dict) -> dict:
    """
    Detect FRESH entry trigger events by comparing current vs previous cycle indicators.
    Returns a dict of boolean trigger flags plus summary counts.
    """
    triggers = {}

    def _g(d, k):
        return (d.get(k) if d else None)

    try:
        # ── Trend ───────────────────────────────────────────────────────────
        cur_macd_line  = _g(current_ind, "macd_line")
        cur_macd_sig   = _g(current_ind, "macd_signal")
        prev_macd_line = _g(prev_ind, "macd_line")
        prev_macd_sig  = _g(prev_ind, "macd_signal")
        cur_macd_hist  = _g(current_ind, "macd_hist")
        prev_macd_hist = _g(prev_ind, "macd_hist")

        if None not in (cur_macd_line, cur_macd_sig, prev_macd_line, prev_macd_sig):
            triggers["macd_just_crossed_bullish"] = (
                prev_macd_line <= prev_macd_sig and cur_macd_line > cur_macd_sig
            )
            triggers["macd_just_crossed_bearish"] = (
                prev_macd_line >= prev_macd_sig and cur_macd_line < cur_macd_sig
            )
        else:
            triggers["macd_just_crossed_bullish"] = False
            triggers["macd_just_crossed_bearish"] = False

        # MACD histogram crossed zero from below (momentum turned positive)
        if None not in (cur_macd_hist, prev_macd_hist):
            triggers["macd_hist_just_turned_positive"] = (
                prev_macd_hist <= 0 and cur_macd_hist > 0
            )
        else:
            triggers["macd_hist_just_turned_positive"] = False

        cur_e9   = _g(current_ind, "ema9")
        cur_e21  = _g(current_ind, "ema21")
        prev_e9  = _g(prev_ind, "ema9")
        prev_e21 = _g(prev_ind, "ema21")

        if None not in (cur_e9, cur_e21, prev_e9, prev_e21):
            triggers["ema9_just_crossed_ema21_bullish"] = (
                prev_e9 <= prev_e21 and cur_e9 > cur_e21
            )
            triggers["ema9_just_crossed_ema21_bearish"] = (
                prev_e9 >= prev_e21 and cur_e9 < cur_e21
            )
        else:
            triggers["ema9_just_crossed_ema21_bullish"] = False
            triggers["ema9_just_crossed_ema21_bearish"] = False

        cur_price   = _g(current_ind, "current_price")
        prev_price  = _g(prev_ind, "current_price")
        cur_r1      = _g(current_ind, "R1")
        prev_r1     = _g(prev_ind, "R1")
        cur_vol_rat = _g(current_ind, "volume_ratio")
        cur_s1      = _g(current_ind, "S1")
        prev_s1     = _g(prev_ind, "S1")
        cur_ema50   = _g(current_ind, "ema50")
        prev_ema50  = _g(prev_ind, "ema50")

        # Price crossed above EMA50 (was below, now above)
        if None not in (cur_price, prev_price, cur_ema50, prev_ema50):
            triggers["price_just_crossed_ema50_up"] = (
                prev_price < prev_ema50 and cur_price >= cur_ema50
            )
        else:
            triggers["price_just_crossed_ema50_up"] = False

        # R1 breakout requires 2x volume confirmation
        if None not in (cur_price, prev_price, cur_r1, prev_r1, cur_vol_rat):
            triggers["price_just_broke_r1_with_volume"] = (
                prev_price <= prev_r1 and cur_price > cur_r1 and cur_vol_rat >= 2.0
            )
        else:
            triggers["price_just_broke_r1_with_volume"] = False

        if None not in (cur_price, prev_price, cur_s1, prev_s1):
            triggers["price_just_broke_s1"] = (
                prev_price >= prev_s1 and cur_price < cur_s1
            )
        else:
            triggers["price_just_broke_s1"] = False

        cur_pct52h  = _g(current_ind, "pct_from_52wk_high")
        prev_pct52h = _g(prev_ind, "pct_from_52wk_high")

        # 52-week high breakout requires 1.5x volume confirmation
        if None not in (cur_pct52h, prev_pct52h, cur_vol_rat):
            triggers["price_just_broke_52wk_high"] = (
                prev_pct52h < 0 and cur_pct52h >= 0 and cur_vol_rat >= 1.5
            )
        else:
            triggers["price_just_broke_52wk_high"] = False

        # ── Momentum ────────────────────────────────────────────────────────
        cur_rsi  = _g(current_ind, "rsi")
        prev_rsi = _g(prev_ind, "rsi")

        if None not in (cur_rsi, prev_rsi):
            triggers["rsi_just_crossed_50_up"]  = prev_rsi < 50  and cur_rsi >= 50
            triggers["rsi_just_crossed_30_up"]  = prev_rsi < 30  and cur_rsi >= 30
            triggers["rsi_just_crossed_70_down"] = prev_rsi > 70 and cur_rsi <= 70
        else:
            triggers["rsi_just_crossed_50_up"]  = False
            triggers["rsi_just_crossed_30_up"]  = False
            triggers["rsi_just_crossed_70_down"] = False

        cur_sk  = _g(current_ind, "stoch_k")
        cur_sd  = _g(current_ind, "stoch_d")
        prev_sk = _g(prev_ind, "stoch_k")
        prev_sd = _g(prev_ind, "stoch_d")

        if None not in (cur_sk, cur_sd, prev_sk, prev_sd):
            triggers["stochrsi_just_crossed_bullish"] = (
                prev_sk <= prev_sd and cur_sk > cur_sd and prev_sk < 30
            )
            triggers["stochrsi_just_crossed_bearish"] = (
                prev_sk >= prev_sd and cur_sk < cur_sd and prev_sk > 70
            )
        else:
            triggers["stochrsi_just_crossed_bullish"] = False
            triggers["stochrsi_just_crossed_bearish"] = False

        # ── Volume ──────────────────────────────────────────────────────────
        cur_vol_today = _g(current_ind, "volume_today")
        cur_vol_avg20 = _g(current_ind, "volume_avg20")

        if None not in (cur_vol_today, cur_vol_avg20) and cur_vol_avg20 > 0:
            triggers["volume_surge_this_candle"] = cur_vol_today > 2.0 * cur_vol_avg20
        else:
            triggers["volume_surge_this_candle"] = False

    except Exception as e:
        logger.warning(f"[indicators] compute_entry_triggers failed: {e}")
        for k in ["macd_just_crossed_bullish", "macd_just_crossed_bearish",
                  "macd_hist_just_turned_positive",
                  "ema9_just_crossed_ema21_bullish", "ema9_just_crossed_ema21_bearish",
                  "price_just_crossed_ema50_up",
                  "price_just_broke_r1_with_volume", "price_just_broke_s1",
                  "price_just_broke_52wk_high",
                  "rsi_just_crossed_50_up", "rsi_just_crossed_30_up", "rsi_just_crossed_70_down",
                  "stochrsi_just_crossed_bullish", "stochrsi_just_crossed_bearish",
                  "volume_surge_this_candle"]:
            triggers.setdefault(k, False)

    # Summary
    bool_keys = [k for k, v in triggers.items() if isinstance(v, bool) and v]
    triggers["fresh_trigger_count"] = len(bool_keys)
    triggers["fresh_trigger_names"] = bool_keys
    return triggers


def _safe(val) -> Optional[float]:
    """Convert NaN/inf to None."""
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _fetch_daily(ticker: str) -> Optional[pd.DataFrame]:
    return fetch_daily_bars(ticker, days=730)


def _fetch_intraday(ticker: str) -> Optional[pd.DataFrame]:
    return fetch_intraday_bars(ticker, days=2)


def _fetch_realtime_price(ticker: str, df: pd.DataFrame) -> tuple[Optional[float], str]:
    """
    Get the most accurate current price available.
    Tries fast_info.last_price first (real-time), falls back to last daily close.
    Applies a 40% sanity check against the 5-day average close.
    Returns (price, source) or (None, "sanity_fail") to signal skip.
    """
    avg5 = float(df["Close"].iloc[-5:].mean()) if len(df) >= 5 else float(df["Close"].iloc[-1])

    # Attempt real-time price from Alpaca snapshot
    try:
        snap = fetch_snapshot(ticker)
        if snap and snap.get("price"):
            last = float(snap["price"])
            if last > 0 and (avg5 <= 0 or abs(last - avg5) / avg5 <= 0.40):
                return last, "alpaca_snapshot"
            elif avg5 > 0:
                logger.warning(
                    f"[indicators] {ticker}: snapshot ${last:.2f} is "
                    f"{abs(last-avg5)/avg5*100:.1f}% from 5d-avg ${avg5:.2f} — ignoring"
                )
    except Exception as e:
        logger.debug(f"[indicators] {ticker}: snapshot unavailable: {e}")

    # Fall back to last close
    fallback = float(df["Close"].iloc[-1])
    if avg5 > 0 and abs(fallback - avg5) / avg5 > 0.40:
        logger.warning(
            f"[indicators] {ticker}: last_close ${fallback:.2f} also "
            f"{abs(fallback-avg5)/avg5*100:.1f}% from 5d-avg ${avg5:.2f} — skipping ticker"
        )
        return None, "sanity_fail"

    return fallback, "last_close"


def compute_vwap(intraday: pd.DataFrame) -> Optional[float]:
    """Compute VWAP from intraday 5-min data for today."""
    try:
        today = datetime.now().date()
        mask = intraday.index.date == today
        df = intraday[mask].copy()
        if df.empty:
            df = intraday.copy()
        if df.empty:
            return None
        typical = (df["High"] + df["Low"] + df["Close"]) / 3
        vwap = (typical * df["Volume"]).cumsum() / df["Volume"].cumsum()
        return _safe(vwap.iloc[-1])
    except Exception as e:
        logger.warning(f"[indicators] VWAP computation failed: {e}")
        return None


def compute_pivot_points(df: pd.DataFrame) -> dict:
    """Classic daily pivot points using previous day's OHLC."""
    try:
        prev = df.iloc[-2]
        H = float(prev["High"])
        L = float(prev["Low"])
        C = float(prev["Close"])
        P  = (H + L + C) / 3
        R1 = 2 * P - L
        R2 = P + (H - L)
        S1 = 2 * P - H
        S2 = P - (H - L)
        return {"P": P, "R1": R1, "R2": R2, "S1": S1, "S2": S2}
    except Exception:
        return {"P": None, "R1": None, "R2": None, "S1": None, "S2": None}


def compute_indicators_from_df(ticker: str, df: pd.DataFrame,
                                intraday: Optional[pd.DataFrame] = None,
                                realtime_price: bool = True) -> dict:
    """
    Compute all indicators from a pre-fetched daily DataFrame.
    Used by both get_indicators() and the backtest engine.
    When realtime_price=False (backtest), uses df's last close directly.
    """
    result = {"ticker": ticker, "error": None}

    if df is None or len(df) < 30:
        result["error"] = "insufficient_data"
        return result

    close = df["Close"]
    high  = df["High"]
    low   = df["Low"]
    vol   = df["Volume"]

    # ── Current price — real-time or historical ────────────────────────────
    if realtime_price:
        current_price, price_source = _fetch_realtime_price(ticker, df)
        if current_price is None:
            result["error"] = "price_sanity_fail"
            return result
        logger.info(f"[indicators] {ticker}: price=${current_price:.2f} source={price_source}")
    else:
        current_price = _safe(close.iloc[-1])
        price_source  = "historical_close"

    prev_close = _safe(close.iloc[-2]) if len(close) >= 2 else None
    open_today = _safe(df["Open"].iloc[-1])

    result["current_price"] = current_price
    result["open_today"]    = open_today
    result["prev_close"]    = prev_close
    result["price_source"]  = price_source

    if open_today and prev_close and prev_close != 0:
        result["gap_pct"] = (open_today - prev_close) / prev_close * 100
    else:
        result["gap_pct"] = None

    if current_price and prev_close and prev_close != 0:
        result["intraday_move_pct"] = (current_price - prev_close) / prev_close * 100
    else:
        result["intraday_move_pct"] = None

    # ── TREND: EMAs ────────────────────────────────────────────────────────
    for period in [9, 21, 50, 200]:
        try:
            ema_ind = ta_trend.EMAIndicator(close=close, window=period, fillna=False)
            result[f"ema{period}"] = _safe(ema_ind.ema_indicator().iloc[-1])
        except Exception:
            result[f"ema{period}"] = None

    # ── TREND: MACD ────────────────────────────────────────────────────────
    try:
        macd_ind    = ta_trend.MACD(close=close, window_slow=26, window_fast=12,
                                    window_sign=9, fillna=False)
        macd_line   = macd_ind.macd()
        macd_signal = macd_ind.macd_signal()
        macd_hist   = macd_ind.macd_diff()

        result["macd_line"]   = _safe(macd_line.iloc[-1])
        result["macd_signal"] = _safe(macd_signal.iloc[-1])
        result["macd_hist"]   = _safe(macd_hist.iloc[-1])

        h = macd_hist.dropna()
        result["macd_hist_prev1"] = _safe(h.iloc[-2]) if len(h) >= 2 else None
        result["macd_hist_prev2"] = _safe(h.iloc[-3]) if len(h) >= 3 else None

        ml = macd_line.dropna()
        ms = macd_signal.dropna()
        min_len = min(len(ml), len(ms))
        if min_len >= 3:
            result["macd_bull_cross"] = any(
                ml.iloc[-i] > ms.iloc[-i] and ml.iloc[-i - 1] <= ms.iloc[-i - 1]
                for i in range(1, 3)
            )
            result["macd_bear_cross"] = any(
                ml.iloc[-i] < ms.iloc[-i] and ml.iloc[-i - 1] >= ms.iloc[-i - 1]
                for i in range(1, 3)
            )
        else:
            result["macd_bull_cross"] = False
            result["macd_bear_cross"] = False
    except Exception as e:
        logger.warning(f"[indicators] MACD failed for {ticker}: {e}")
        for k in ["macd_line", "macd_signal", "macd_hist",
                  "macd_hist_prev1", "macd_hist_prev2",
                  "macd_bull_cross", "macd_bear_cross"]:
            result[k] = None

    # ── TREND: ADX ─────────────────────────────────────────────────────────
    try:
        adx_ind = ta_trend.ADXIndicator(high=high, low=low, close=close,
                                         window=14, fillna=False)
        result["adx"]          = _safe(adx_ind.adx().iloc[-1])
        result["adx_di_plus"]  = _safe(adx_ind.adx_pos().iloc[-1])
        result["adx_di_minus"] = _safe(adx_ind.adx_neg().iloc[-1])
    except Exception as e:
        logger.warning(f"[indicators] ADX failed for {ticker}: {e}")
        result["adx"] = result["adx_di_plus"] = result["adx_di_minus"] = None

    # ── TREND: Parabolic SAR ───────────────────────────────────────────────
    try:
        psar_ind  = ta_trend.PSARIndicator(high=high, low=low, close=close,
                                            step=0.02, max_step=0.2, fillna=False)
        psar_up_val   = _safe(psar_ind.psar_up().iloc[-1])
        psar_down_val = _safe(psar_ind.psar_down().iloc[-1])
        if psar_up_val is not None:
            result["psar"]         = psar_up_val
            result["psar_bullish"] = True
        elif psar_down_val is not None:
            result["psar"]         = psar_down_val
            result["psar_bullish"] = False
        else:
            result["psar"] = None
            result["psar_bullish"] = None
    except Exception as e:
        logger.warning(f"[indicators] PSAR failed for {ticker}: {e}")
        result["psar"] = None
        result["psar_bullish"] = None

    # ── MOMENTUM: RSI ──────────────────────────────────────────────────────
    try:
        result["rsi"] = _safe(
            ta_momentum.RSIIndicator(close=close, window=14, fillna=False).rsi().iloc[-1]
        )
    except Exception:
        result["rsi"] = None

    # ── MOMENTUM: Stochastic RSI ───────────────────────────────────────────
    try:
        srsi_ind = ta_momentum.StochRSIIndicator(close=close, window=14,
                                                  smooth1=3, smooth2=3, fillna=False)
        k_series = srsi_ind.stochrsi_k()
        d_series = srsi_ind.stochrsi_d()
        result["stoch_k"] = _safe(k_series.iloc[-1])
        result["stoch_d"] = _safe(d_series.iloc[-1])
        k_clean = k_series.dropna()
        d_clean = d_series.dropna()
        result["stoch_k_prev"] = _safe(k_clean.iloc[-2]) if len(k_clean) >= 2 else None
        result["stoch_d_prev"] = _safe(d_clean.iloc[-2]) if len(d_clean) >= 2 else None
    except Exception as e:
        logger.warning(f"[indicators] StochRSI failed for {ticker}: {e}")
        result["stoch_k"] = result["stoch_d"] = \
            result["stoch_k_prev"] = result["stoch_d_prev"] = None

    # ── MOMENTUM: CCI ──────────────────────────────────────────────────────
    try:
        result["cci"] = _safe(
            ta_trend.CCIIndicator(high=high, low=low, close=close,
                                  window=20, constant=0.015, fillna=False).cci().iloc[-1]
        )
    except Exception:
        result["cci"] = None

    # ── MOMENTUM: Williams %R ──────────────────────────────────────────────
    try:
        result["willr"] = _safe(
            ta_momentum.WilliamsRIndicator(high=high, low=low, close=close,
                                           lbp=14, fillna=False).williams_r().iloc[-1]
        )
    except Exception:
        result["willr"] = None

    # ── MOMENTUM: Rate of Change ───────────────────────────────────────────
    try:
        result["roc"] = _safe(
            ta_momentum.ROCIndicator(close=close, window=10, fillna=False).roc().iloc[-1]
        )
    except Exception:
        result["roc"] = None

    # ── VOLATILITY: Bollinger Bands ────────────────────────────────────────
    try:
        bb_ind   = ta_volatility.BollingerBands(close=close, window=20, window_dev=2,
                                                 fillna=False)
        bb_upper = bb_ind.bollinger_hband()
        bb_mid   = bb_ind.bollinger_mavg()
        bb_lower = bb_ind.bollinger_lband()
        bb_pctb  = bb_ind.bollinger_pband()
        bb_wband = bb_ind.bollinger_wband()

        result["bb_upper"] = _safe(bb_upper.iloc[-1])
        result["bb_mid"]   = _safe(bb_mid.iloc[-1])
        result["bb_lower"] = _safe(bb_lower.iloc[-1])
        result["bb_pctb"]  = _safe(bb_pctb.iloc[-1])
        result["bb_bw"]    = _safe(bb_wband.iloc[-1])

        bw_clean = bb_wband.dropna()
        if len(bw_clean) >= 20:
            bw_min = bw_clean.iloc[-20:].min()
            bw_max = bw_clean.iloc[-20:].max()
            bw_pos = (bw_clean.iloc[-1] - bw_min) / (bw_max - bw_min + 1e-9)
            result["bb_squeeze"] = bool(bw_pos < 0.2)
        else:
            result["bb_squeeze"] = False

        result["bb_bw_expanding"] = (
            bool(bw_clean.iloc[-1] > bw_clean.iloc[-2]) if len(bw_clean) >= 2 else None
        )
    except Exception as e:
        logger.warning(f"[indicators] BBands failed for {ticker}: {e}")
        for k in ["bb_upper", "bb_mid", "bb_lower", "bb_pctb",
                  "bb_bw", "bb_squeeze", "bb_bw_expanding"]:
            result[k] = None

    # ── VOLATILITY: ATR ────────────────────────────────────────────────────
    try:
        atr_val = _safe(
            ta_volatility.AverageTrueRange(high=high, low=low, close=close,
                                           window=14, fillna=False)
            .average_true_range().iloc[-1]
        )
        result["atr"] = atr_val
        result["atr_pct"] = (
            atr_val / current_price * 100
            if (atr_val and current_price and current_price != 0)
            else None
        )
    except Exception:
        result["atr"] = None
        result["atr_pct"] = None

    # ── VOLATILITY: Keltner Channel ────────────────────────────────────────
    try:
        kc_ind = ta_volatility.KeltnerChannel(high=high, low=low, close=close,
                                               window=20, window_atr=10,
                                               multiplier=2, fillna=False)
        result["kc_upper"] = _safe(kc_ind.keltner_channel_hband().iloc[-1])
        result["kc_lower"] = _safe(kc_ind.keltner_channel_lband().iloc[-1])
    except Exception:
        result["kc_upper"] = result["kc_lower"] = None

    # ── VOLUME: VWAP ───────────────────────────────────────────────────────
    result["vwap"] = (
        compute_vwap(intraday)
        if intraday is not None and not intraday.empty
        else None
    )

    # ── VOLUME: OBV ────────────────────────────────────────────────────────
    try:
        obv_series = ta_volume.OnBalanceVolumeIndicator(
            close=close, volume=vol, fillna=False
        ).on_balance_volume()
        result["obv"] = _safe(obv_series.iloc[-1])

        obv_clean = obv_series.dropna()
        if len(obv_clean) >= 10:
            slope = float(np.polyfit(range(10), obv_clean.iloc[-10:].values, 1)[0])
            result["obv_slope"] = slope
            result["obv_rising"] = slope > 0
        else:
            result["obv_slope"] = None
            result["obv_rising"] = None

        if len(obv_clean) >= 20:
            obv_hi_20   = float(obv_clean.iloc[-20:].max())
            obv_lo_20   = float(obv_clean.iloc[-20:].min())
            price_hi_20 = float(close.iloc[-20:].max())
            price_lo_20 = float(close.iloc[-20:].min())
            obv_now     = float(obv_clean.iloc[-1])
            price_now   = float(close.iloc[-1])
            result["obv_bull_divergence"] = (
                obv_now >= obv_hi_20 * 0.9999 and price_now < price_hi_20 * 0.99
            )
            result["obv_bear_divergence"] = (
                obv_now <= obv_lo_20 * 1.0001 and price_now > price_lo_20 * 1.01
            )
        else:
            result["obv_bull_divergence"] = False
            result["obv_bear_divergence"] = False
    except Exception as e:
        logger.warning(f"[indicators] OBV failed for {ticker}: {e}")
        for k in ["obv", "obv_slope", "obv_rising",
                  "obv_bull_divergence", "obv_bear_divergence"]:
            result[k] = None

    # ── VOLUME: Volume Ratio ───────────────────────────────────────────────
    try:
        today_vol  = float(vol.iloc[-1])
        avg_vol_20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
        result["volume_today"] = today_vol
        result["volume_avg20"] = avg_vol_20
        result["volume_ratio"] = today_vol / avg_vol_20 if avg_vol_20 > 0 else None
    except Exception:
        result["volume_today"] = result["volume_avg20"] = result["volume_ratio"] = None

    # ── VOLUME: MFI ────────────────────────────────────────────────────────
    try:
        result["mfi"] = _safe(
            ta_volume.MFIIndicator(high=high, low=low, close=close,
                                   volume=vol, window=14, fillna=False)
            .money_flow_index().iloc[-1]
        )
    except Exception:
        result["mfi"] = None

    # ── PRICE CONTEXT: 52-week range ───────────────────────────────────────
    try:
        year_data = close.iloc[-252:] if len(close) >= 252 else close
        wk52_high = float(year_data.max())
        wk52_low  = float(year_data.min())
        result["wk52_high"] = wk52_high
        result["wk52_low"]  = wk52_low
        if current_price:
            result["pct_from_52wk_high"] = (current_price - wk52_high) / wk52_high * 100
            result["pct_from_52wk_low"]  = (current_price - wk52_low)  / wk52_low  * 100
        else:
            result["pct_from_52wk_high"] = None
            result["pct_from_52wk_low"]  = None
    except Exception:
        result["wk52_high"] = result["wk52_low"] = None
        result["pct_from_52wk_high"] = result["pct_from_52wk_low"] = None

    # ── PRICE CONTEXT: % from EMA200 ──────────────────────────────────────
    if result.get("ema200") and current_price:
        result["pct_from_ema200"] = (current_price - result["ema200"]) / result["ema200"] * 100
    else:
        result["pct_from_ema200"] = None

    # ── PRICE CONTEXT: Pivot Points ───────────────────────────────────────
    result.update(compute_pivot_points(df))

    return result


def get_indicators(ticker: str) -> dict:
    """Fetch data and compute all indicators for a ticker. Returns clean dict."""

    _now = datetime.now()
    _bucket = (_now.minute // 15) * 15
    cache_key = f"{ticker}_{_now.strftime('%Y%m%d_%H')}_{_bucket:02d}"
    if cache_key in _cache:
        logger.debug(f"[indicators] cache hit for {ticker}")
        return _cache[cache_key]

    daily    = _fetch_daily(ticker)
    intraday = _fetch_intraday(ticker)

    result = compute_indicators_from_df(ticker, daily, intraday, realtime_price=True)

    # Merge intraday 15-min indicators
    intraday_ind = get_intraday_indicators(ticker)
    result.update(intraday_ind)

    if not result.get("error"):
        _cache[cache_key] = result
    return result


def get_intraday_indicators(ticker: str) -> dict:
    """
    Fetch 15-minute intraday data and compute intraday VWAP, RSI, and MACD.
    Returns dict with: intraday_vwap, intraday_rsi, intraday_macd_hist, intraday_vs_vwap
    """
    result = {
        "intraday_vwap": None,
        "intraday_rsi": None,
        "intraday_macd_hist": None,
        "intraday_vs_vwap": None,
    }
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import Adjustment, DataFeed
        from bot.data import get_data_client
        from datetime import timezone as _tz
        client = get_data_client()
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(15, TimeFrameUnit.Minute),
            start=datetime.now(_tz.utc) - timedelta(days=5),
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        bar_list = (bars.data or {}).get(ticker) if bars and hasattr(bars, "data") else None
        if not bar_list:
            return result
        from bot.data import _bars_to_df
        df = _bars_to_df(bar_list)
        if df is None or df.empty:
            return result

        # Filter to today's bars for VWAP
        today = datetime.now().date()
        today_mask = df.index.date == today
        today_df = df[today_mask].copy()
        if today_df.empty:
            today_df = df.copy()

        # Intraday VWAP (reset daily)
        typical = (today_df["High"] + today_df["Low"] + today_df["Close"]) / 3
        cum_vol = today_df["Volume"].cumsum()
        if cum_vol.iloc[-1] > 0:
            vwap_val = _safe((typical * today_df["Volume"]).cumsum().iloc[-1] / cum_vol.iloc[-1])
            result["intraday_vwap"] = vwap_val

        # Intraday RSI(14) on all 15-min closes (need enough bars)
        if len(df) >= 15:
            rsi_ind = ta_momentum.RSIIndicator(close=df["Close"], window=14, fillna=False)
            result["intraday_rsi"] = _safe(rsi_ind.rsi().iloc[-1])

        # Intraday MACD histogram on 15-min closes
        if len(df) >= 26:
            macd_ind  = ta_trend.MACD(close=df["Close"], window_slow=26,
                                      window_fast=12, window_sign=9, fillna=False)
            result["intraday_macd_hist"] = _safe(macd_ind.macd_diff().iloc[-1])

        # % distance from intraday VWAP
        if result["intraday_vwap"] and result["intraday_vwap"] > 0:
            last_price = _safe(df["Close"].iloc[-1])
            if last_price:
                result["intraday_vs_vwap"] = round(
                    (last_price - result["intraday_vwap"]) / result["intraday_vwap"] * 100, 3
                )
    except Exception as e:
        logger.warning(f"[indicators] intraday_indicators failed for {ticker}: {e}")
    return result


def get_indicators_batch(tickers: list, max_workers: int = 8) -> dict:
    """Fetch indicators for multiple tickers in parallel using ThreadPoolExecutor."""
    import time as _time
    results = {}
    t0 = _time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(get_indicators, t): t for t in tickers}
        for fut, ticker in futures.items():
            try:
                results[ticker] = fut.result()
            except Exception as e:
                logger.warning(f"[indicators] batch failed for {ticker}: {e}")
                results[ticker] = {"ticker": ticker, "error": str(e)}
    elapsed = _time.time() - t0
    logger.info(f"[perf] Fetched {len(tickers)} tickers in {elapsed:.1f}s")

    # Compute entry triggers by comparing against previous cycle
    try:
        prev_all = load_prev_indicators()
        new_prev = {}
        for ticker, ind in results.items():
            if ind.get("error"):
                continue
            prev_ind = prev_all.get(ticker, {})
            triggers = compute_entry_triggers(ind, prev_ind)
            ind["entry_triggers"] = triggers
            # Store current for next cycle (strip large nested dicts to keep file small)
            new_prev[ticker] = {k: v for k, v in ind.items()
                                if k not in ("entry_triggers", "_indicators", "_news", "_position")}
        save_prev_indicators(new_prev)
    except Exception as e:
        logger.warning(f"[indicators] entry trigger batch failed: {e}")

    return results
