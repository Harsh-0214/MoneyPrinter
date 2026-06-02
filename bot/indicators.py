"""Technical indicator calculations using yfinance + pandas-ta."""

import logging
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf

logger = logging.getLogger(__name__)

_cache: dict = {}


def _safe(val):
    """Convert NaN/inf to None."""
    if val is None:
        return None
    try:
        if np.isnan(val) or np.isinf(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _fetch_daily(ticker: str) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="90d", interval="1d", auto_adjust=True)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logger.warning(f"[indicators] daily fetch failed for {ticker}: {e}")
        return None


def _fetch_intraday(ticker: str) -> Optional[pd.DataFrame]:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period="2d", interval="5m", auto_adjust=True)
        if df is None or df.empty:
            return None
        df.index = pd.to_datetime(df.index)
        return df
    except Exception as e:
        logger.warning(f"[indicators] intraday fetch failed for {ticker}: {e}")
        return None


def compute_vwap(intraday: pd.DataFrame) -> Optional[float]:
    """Compute VWAP from intraday 5-min data for today."""
    try:
        today = datetime.now().date()
        # filter to today only
        mask = intraday.index.date == today
        df = intraday[mask].copy()
        if df.empty:
            # fallback: use last available day
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
        P = (H + L + C) / 3
        R1 = 2 * P - L
        R2 = P + (H - L)
        S1 = 2 * P - H
        S2 = P - (H - L)
        return {"P": P, "R1": R1, "R2": R2, "S1": S1, "S2": S2}
    except Exception:
        return {"P": None, "R1": None, "R2": None, "S1": None, "S2": None}


def get_indicators(ticker: str) -> dict:
    """Fetch and compute all indicators for a ticker. Returns clean dict."""

    cache_key = f"{ticker}_{datetime.now().strftime('%Y%m%d_%H')}"
    if cache_key in _cache:
        logger.debug(f"[indicators] cache hit for {ticker}")
        return _cache[cache_key]

    result = {"ticker": ticker, "error": None}

    daily = _fetch_daily(ticker)
    intraday = _fetch_intraday(ticker)

    if daily is None or len(daily) < 30:
        result["error"] = "insufficient_data"
        return result

    df = daily.copy()

    # Current price context
    close = df["Close"]
    current_price = _safe(close.iloc[-1])
    prev_close = _safe(close.iloc[-2]) if len(close) >= 2 else None
    open_today = _safe(df["Open"].iloc[-1])

    result["current_price"] = current_price
    result["open_today"] = open_today
    result["prev_close"] = prev_close

    # Today's gap
    if open_today and prev_close and prev_close != 0:
        result["gap_pct"] = (open_today - prev_close) / prev_close * 100
    else:
        result["gap_pct"] = None

    # --- TREND: EMAs ---
    for period in [9, 21, 50, 200]:
        ema = ta.ema(close, length=period)
        result[f"ema{period}"] = _safe(ema.iloc[-1]) if ema is not None and not ema.empty else None

    # --- TREND: MACD ---
    try:
        macd_df = ta.macd(close, fast=12, slow=26, signal=9)
        if macd_df is not None and not macd_df.empty:
            cols = macd_df.columns.tolist()
            macd_col = [c for c in cols if "MACD_" in c and "MACDh" not in c and "MACDs" not in c]
            hist_col = [c for c in cols if "MACDh" in c]
            sig_col  = [c for c in cols if "MACDs" in c]

            macd_line   = macd_df[macd_col[0]] if macd_col else None
            macd_hist   = macd_df[hist_col[0]] if hist_col else None
            macd_signal = macd_df[sig_col[0]]  if sig_col  else None

            result["macd_line"]   = _safe(macd_line.iloc[-1])   if macd_line   is not None else None
            result["macd_signal"] = _safe(macd_signal.iloc[-1]) if macd_signal is not None else None
            result["macd_hist"]   = _safe(macd_hist.iloc[-1])   if macd_hist   is not None else None

            # last 3 histogram values for crossover detection
            if macd_hist is not None and len(macd_hist.dropna()) >= 3:
                h = macd_hist.dropna()
                result["macd_hist_prev1"] = _safe(h.iloc[-2])
                result["macd_hist_prev2"] = _safe(h.iloc[-3])
            else:
                result["macd_hist_prev1"] = None
                result["macd_hist_prev2"] = None

            # detect crossover in last 3 bars
            if macd_line is not None and macd_signal is not None:
                ml = macd_line.dropna()
                ms = macd_signal.dropna()
                min_len = min(len(ml), len(ms))
                if min_len >= 3:
                    cross_bull = any(
                        ml.iloc[-i] > ms.iloc[-i] and ml.iloc[-i-1] <= ms.iloc[-i-1]
                        for i in range(1, 3)
                    )
                    cross_bear = any(
                        ml.iloc[-i] < ms.iloc[-i] and ml.iloc[-i-1] >= ms.iloc[-i-1]
                        for i in range(1, 3)
                    )
                    result["macd_bull_cross"] = cross_bull
                    result["macd_bear_cross"] = cross_bear
                else:
                    result["macd_bull_cross"] = False
                    result["macd_bear_cross"] = False
            else:
                result["macd_bull_cross"] = False
                result["macd_bear_cross"] = False
        else:
            for k in ["macd_line","macd_signal","macd_hist","macd_hist_prev1","macd_hist_prev2","macd_bull_cross","macd_bear_cross"]:
                result[k] = None
    except Exception as e:
        logger.warning(f"[indicators] MACD failed for {ticker}: {e}")
        for k in ["macd_line","macd_signal","macd_hist","macd_hist_prev1","macd_hist_prev2","macd_bull_cross","macd_bear_cross"]:
            result[k] = None

    # --- TREND: ADX ---
    try:
        adx_df = ta.adx(df["High"], df["Low"], close, length=14)
        if adx_df is not None and not adx_df.empty:
            adx_col  = [c for c in adx_df.columns if c.startswith("ADX_")]
            dmp_col  = [c for c in adx_df.columns if c.startswith("DMP_")]
            dmn_col  = [c for c in adx_df.columns if c.startswith("DMN_")]
            result["adx"]    = _safe(adx_df[adx_col[0]].iloc[-1])  if adx_col  else None
            result["adx_di_plus"]  = _safe(adx_df[dmp_col[0]].iloc[-1]) if dmp_col else None
            result["adx_di_minus"] = _safe(adx_df[dmn_col[0]].iloc[-1]) if dmn_col else None
        else:
            result["adx"] = result["adx_di_plus"] = result["adx_di_minus"] = None
    except Exception as e:
        logger.warning(f"[indicators] ADX failed for {ticker}: {e}")
        result["adx"] = result["adx_di_plus"] = result["adx_di_minus"] = None

    # --- TREND: Parabolic SAR ---
    try:
        psar_df = ta.psar(df["High"], df["Low"], close)
        if psar_df is not None and not psar_df.empty:
            bull_col = [c for c in psar_df.columns if "PSARl" in c]
            bear_col = [c for c in psar_df.columns if "PSARs" in c]
            if bull_col and bear_col:
                bull_val = psar_df[bull_col[0]].iloc[-1]
                bear_val = psar_df[bear_col[0]].iloc[-1]
                if not np.isnan(bull_val):
                    result["psar"] = float(bull_val)
                    result["psar_bullish"] = True
                else:
                    result["psar"] = float(bear_val) if not np.isnan(bear_val) else None
                    result["psar_bullish"] = False
            else:
                result["psar"] = None
                result["psar_bullish"] = None
        else:
            result["psar"] = None
            result["psar_bullish"] = None
    except Exception as e:
        logger.warning(f"[indicators] PSAR failed for {ticker}: {e}")
        result["psar"] = None
        result["psar_bullish"] = None

    # --- MOMENTUM: RSI ---
    try:
        rsi = ta.rsi(close, length=14)
        result["rsi"] = _safe(rsi.iloc[-1]) if rsi is not None and not rsi.empty else None
    except Exception:
        result["rsi"] = None

    # --- MOMENTUM: Stochastic RSI ---
    try:
        stochrsi_df = ta.stochrsi(close, length=14, rsi_length=14, k=3, d=3)
        if stochrsi_df is not None and not stochrsi_df.empty:
            k_col = [c for c in stochrsi_df.columns if "STOCHRSIk" in c]
            d_col = [c for c in stochrsi_df.columns if "STOCHRSId" in c]
            result["stoch_k"] = _safe(stochrsi_df[k_col[0]].iloc[-1]) if k_col else None
            result["stoch_d"] = _safe(stochrsi_df[d_col[0]].iloc[-1]) if d_col else None
            if k_col and d_col and len(stochrsi_df) >= 3:
                k = stochrsi_df[k_col[0]].dropna()
                d = stochrsi_df[d_col[0]].dropna()
                min_len = min(len(k), len(d))
                if min_len >= 2:
                    result["stoch_k_prev"] = _safe(k.iloc[-2])
                    result["stoch_d_prev"] = _safe(d.iloc[-2])
                else:
                    result["stoch_k_prev"] = result["stoch_d_prev"] = None
            else:
                result["stoch_k_prev"] = result["stoch_d_prev"] = None
        else:
            result["stoch_k"] = result["stoch_d"] = result["stoch_k_prev"] = result["stoch_d_prev"] = None
    except Exception as e:
        logger.warning(f"[indicators] StochRSI failed for {ticker}: {e}")
        result["stoch_k"] = result["stoch_d"] = result["stoch_k_prev"] = result["stoch_d_prev"] = None

    # --- MOMENTUM: CCI ---
    try:
        cci = ta.cci(df["High"], df["Low"], close, length=20)
        result["cci"] = _safe(cci.iloc[-1]) if cci is not None and not cci.empty else None
    except Exception:
        result["cci"] = None

    # --- MOMENTUM: Williams %R ---
    try:
        willr = ta.willr(df["High"], df["Low"], close, length=14)
        result["willr"] = _safe(willr.iloc[-1]) if willr is not None and not willr.empty else None
    except Exception:
        result["willr"] = None

    # --- MOMENTUM: Rate of Change ---
    try:
        roc = ta.roc(close, length=10)
        result["roc"] = _safe(roc.iloc[-1]) if roc is not None and not roc.empty else None
    except Exception:
        result["roc"] = None

    # --- VOLATILITY: Bollinger Bands ---
    try:
        bb_df = ta.bbands(close, length=20, std=2)
        if bb_df is not None and not bb_df.empty:
            upper_col = [c for c in bb_df.columns if "BBU" in c]
            mid_col   = [c for c in bb_df.columns if "BBM" in c]
            lower_col = [c for c in bb_df.columns if "BBL" in c]
            pctb_col  = [c for c in bb_df.columns if "BBP" in c]
            bw_col    = [c for c in bb_df.columns if "BBB" in c]
            result["bb_upper"] = _safe(bb_df[upper_col[0]].iloc[-1]) if upper_col else None
            result["bb_mid"]   = _safe(bb_df[mid_col[0]].iloc[-1])   if mid_col   else None
            result["bb_lower"] = _safe(bb_df[lower_col[0]].iloc[-1]) if lower_col else None
            result["bb_pctb"]  = _safe(bb_df[pctb_col[0]].iloc[-1]) if pctb_col  else None
            result["bb_bw"]    = _safe(bb_df[bw_col[0]].iloc[-1])   if bw_col    else None

            # Squeeze: bandwidth in bottom 20% of recent range
            if bw_col:
                bw_series = bb_df[bw_col[0]].dropna()
                if len(bw_series) >= 20:
                    bw_pct = (bw_series.iloc[-1] - bw_series.iloc[-20:].min()) / (
                        bw_series.iloc[-20:].max() - bw_series.iloc[-20:].min() + 1e-9
                    )
                    result["bb_squeeze"] = bw_pct < 0.2
                else:
                    result["bb_squeeze"] = False
            else:
                result["bb_squeeze"] = False

            # Bandwidth expanding: current > prev
            if bw_col and len(bb_df[bw_col[0]].dropna()) >= 2:
                bw = bb_df[bw_col[0]].dropna()
                result["bb_bw_expanding"] = bw.iloc[-1] > bw.iloc[-2]
            else:
                result["bb_bw_expanding"] = None
        else:
            for k in ["bb_upper","bb_mid","bb_lower","bb_pctb","bb_bw","bb_squeeze","bb_bw_expanding"]:
                result[k] = None
    except Exception as e:
        logger.warning(f"[indicators] BBands failed for {ticker}: {e}")
        for k in ["bb_upper","bb_mid","bb_lower","bb_pctb","bb_bw","bb_squeeze","bb_bw_expanding"]:
            result[k] = None

    # --- VOLATILITY: ATR ---
    try:
        atr = ta.atr(df["High"], df["Low"], close, length=14)
        atr_val = _safe(atr.iloc[-1]) if atr is not None and not atr.empty else None
        result["atr"] = atr_val
        if atr_val and current_price and current_price != 0:
            result["atr_pct"] = atr_val / current_price * 100
        else:
            result["atr_pct"] = None
    except Exception:
        result["atr"] = None
        result["atr_pct"] = None

    # --- VOLATILITY: Keltner Channel ---
    try:
        kc_df = ta.kc(df["High"], df["Low"], close, length=20, scalar=2)
        if kc_df is not None and not kc_df.empty:
            upper_col = [c for c in kc_df.columns if "KCUe" in c]
            lower_col = [c for c in kc_df.columns if "KCLe" in c]
            result["kc_upper"] = _safe(kc_df[upper_col[0]].iloc[-1]) if upper_col else None
            result["kc_lower"] = _safe(kc_df[lower_col[0]].iloc[-1]) if lower_col else None
        else:
            result["kc_upper"] = result["kc_lower"] = None
    except Exception:
        result["kc_upper"] = result["kc_lower"] = None

    # --- VOLUME: VWAP ---
    result["vwap"] = compute_vwap(intraday) if intraday is not None and not intraday.empty else None

    # --- VOLUME: OBV ---
    try:
        obv = ta.obv(close, df["Volume"])
        if obv is not None and not obv.empty:
            result["obv"] = _safe(obv.iloc[-1])
            obv_clean = obv.dropna()
            if len(obv_clean) >= 10:
                slope = float(np.polyfit(range(10), obv_clean.iloc[-10:].values, 1)[0])
                result["obv_slope"] = slope
                result["obv_rising"] = slope > 0
            else:
                result["obv_slope"] = None
                result["obv_rising"] = None

            # divergence: OBV new highs/lows vs price
            if len(obv_clean) >= 20:
                obv_high_20 = obv_clean.iloc[-20:].max()
                obv_low_20  = obv_clean.iloc[-20:].min()
                price_high_20 = close.iloc[-20:].max()
                price_low_20  = close.iloc[-20:].min()
                # OBV new high but price not
                result["obv_bull_divergence"] = (
                    _safe(obv_clean.iloc[-1]) == _safe(obv_high_20) and
                    _safe(close.iloc[-1]) < _safe(price_high_20) * 0.99
                )
                result["obv_bear_divergence"] = (
                    _safe(obv_clean.iloc[-1]) == _safe(obv_low_20) and
                    _safe(close.iloc[-1]) > _safe(price_low_20) * 1.01
                )
            else:
                result["obv_bull_divergence"] = False
                result["obv_bear_divergence"] = False
        else:
            for k in ["obv","obv_slope","obv_rising","obv_bull_divergence","obv_bear_divergence"]:
                result[k] = None
    except Exception as e:
        logger.warning(f"[indicators] OBV failed for {ticker}: {e}")
        for k in ["obv","obv_slope","obv_rising","obv_bull_divergence","obv_bear_divergence"]:
            result[k] = None

    # --- VOLUME: Volume Ratio ---
    try:
        vol = df["Volume"]
        today_vol = float(vol.iloc[-1])
        avg_vol_20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())
        result["volume_today"] = today_vol
        result["volume_avg20"] = avg_vol_20
        result["volume_ratio"] = today_vol / avg_vol_20 if avg_vol_20 > 0 else None
    except Exception:
        result["volume_today"] = result["volume_avg20"] = result["volume_ratio"] = None

    # --- VOLUME: MFI ---
    try:
        mfi = ta.mfi(df["High"], df["Low"], close, df["Volume"], length=14)
        result["mfi"] = _safe(mfi.iloc[-1]) if mfi is not None and not mfi.empty else None
    except Exception:
        result["mfi"] = None

    # --- PRICE CONTEXT: 52-week range ---
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

    # --- PRICE CONTEXT: % from EMA200 ---
    if result.get("ema200") and current_price:
        result["pct_from_ema200"] = (current_price - result["ema200"]) / result["ema200"] * 100
    else:
        result["pct_from_ema200"] = None

    # --- PRICE CONTEXT: Pivot Points ---
    pivots = compute_pivot_points(df)
    result.update(pivots)

    _cache[cache_key] = result
    return result


def get_indicators_batch(tickers: list, max_workers: int = 5) -> dict:
    """Fetch indicators for multiple tickers concurrently."""
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as exe:
        futures = {exe.submit(get_indicators, t): t for t in tickers}
        for fut, ticker in futures.items():
            try:
                results[ticker] = fut.result()
            except Exception as e:
                logger.warning(f"[indicators] batch failed for {ticker}: {e}")
                results[ticker] = {"ticker": ticker, "error": str(e)}
    return results
