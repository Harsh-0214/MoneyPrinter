"""
Real-time multi-day indicator context.

Fetches the last ~80 days of daily OHLCV bars fresh from the API (no JSON
history files) and computes a 3-day progression analysis to tell the scorer
and Claude whether a setup has been building over multiple days or is a
single-day spike.

Public API:
    get_historical_context(ticker, data_client=None) -> dict
        Standalone per-ticker call with a 10-second hard timeout.

    get_historical_context_batch(tickers, data_client=None) -> dict[str, dict]
        Efficient batch version used by run_full_scan — one API call for all
        tickers instead of one per ticker.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Days of history to fetch — enough warm-up for MACD(26), EMA50, RSI(14)
_FETCH_DAYS = 80


def _safe(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
        if np.isnan(f) or np.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None


def _compute_context_from_df(ticker: str, df: pd.DataFrame) -> dict:
    """
    Core computation. Expects a capitalised OHLCV DataFrame indexed by date.
    Returns the full historical_context dict, or {} on any failure.
    """
    try:
        if df is None or len(df) < 5:
            return {}

        close = df["Close"]
        vol   = df["Volume"]

        # ── Indicator series ──────────────────────────────────────────────────
        try:
            import ta.momentum as ta_momentum
            rsi_series = ta_momentum.RSIIndicator(
                close=close, window=14, fillna=False
            ).rsi()
        except Exception:
            rsi_series = pd.Series([np.nan] * len(close), index=close.index)

        try:
            import ta.trend as ta_trend
            macd_hist_series = ta_trend.MACD(
                close=close, window_slow=26, window_fast=12,
                window_sign=9, fillna=False
            ).macd_diff()
        except Exception:
            macd_hist_series = pd.Series([np.nan] * len(close), index=close.index)

        try:
            import ta.trend as ta_trend
            ema50_series = ta_trend.EMAIndicator(
                close=close, window=50, fillna=False
            ).ema_indicator()
        except Exception:
            ema50_series = pd.Series([np.nan] * len(close), index=close.index)

        # 20-day average volume (exclude today)
        vol_avg20 = float(vol.iloc[-21:-1].mean()) if len(vol) >= 21 else float(vol.mean())

        # ── Per-day records for the last 3 trading days ───────────────────────
        if len(close) < 3:
            return {}

        day_records = []
        for offset in (-3, -2, -1):   # day_minus_2, day_minus_1, day_0
            try:
                price     = _safe(close.iloc[offset])
                prev_p    = _safe(close.iloc[offset - 1])
                pct_chg   = (
                    round((price - prev_p) / prev_p * 100, 2)
                    if price and prev_p and prev_p != 0 else None
                )
                day_vol   = _safe(vol.iloc[offset])
                vol_ratio = (
                    round(day_vol / vol_avg20, 2)
                    if day_vol and vol_avg20 > 0 else None
                )
                rsi_val   = _safe(rsi_series.iloc[offset])
                macd_h    = _safe(macd_hist_series.iloc[offset])
                ema50_val = _safe(ema50_series.iloc[offset])
                above50   = bool(price and ema50_val and price > ema50_val)

                day_records.append({
                    "close":        round(price, 2) if price is not None else None,
                    "pct_change":   pct_chg,
                    "rsi":          round(rsi_val, 1) if rsi_val is not None else None,
                    "macd_hist":    round(macd_h, 4) if macd_h is not None else None,
                    "volume_ratio": vol_ratio,
                    "above_ema50":  above50,
                })
            except Exception as e:
                logger.debug(f"[hist_ctx] {ticker} day offset {offset}: {e}")
                day_records.append({})

        d2, d1, d0 = day_records[0], day_records[1], day_records[2]

        # ── Progression signals ───────────────────────────────────────────────
        rsi_vals = [
            d["rsi"] for d in (d2, d1, d0)
            if d.get("rsi") is not None
        ]
        rsi_trending_up = bool(
            len(rsi_vals) >= 2 and
            all(rsi_vals[i] < rsi_vals[i + 1] for i in range(len(rsi_vals) - 1))
        )

        macd_vals = [
            d["macd_hist"] for d in (d2, d1, d0)
            if d.get("macd_hist") is not None
        ]
        macd_hist_rising = bool(
            len(macd_vals) >= 2 and
            all(macd_vals[i] < macd_vals[i + 1] for i in range(len(macd_vals) - 1))
        )

        vol_ratios = [d.get("volume_ratio") for d in (d2, d1, d0)]
        volume_sustained = sum(
            1 for v in vol_ratios if v is not None and v >= 1.2
        ) >= 2

        above_flags = [d.get("above_ema50", False) for d in (d2, d1, d0)]
        price_above_ema50_holding = sum(1 for f in above_flags if f) >= 2

        days_of_confluence = sum([
            rsi_trending_up,
            macd_hist_rising,
            volume_sustained,
            price_above_ema50_holding,
        ])

        # net_score_direction
        rsi_declining = bool(
            len(rsi_vals) >= 2 and
            all(rsi_vals[i] > rsi_vals[i + 1] for i in range(len(rsi_vals) - 1))
        )
        macd_declining = bool(
            len(macd_vals) >= 2 and
            all(macd_vals[i] > macd_vals[i + 1] for i in range(len(macd_vals) - 1))
        )

        if days_of_confluence >= 3:
            net_score_direction = "building"
        elif days_of_confluence == 0 or (rsi_declining and macd_declining):
            net_score_direction = "fading"
        elif days_of_confluence >= 2:
            net_score_direction = "building"
        else:
            net_score_direction = "mixed"

        # maturity_label
        if days_of_confluence >= 3:
            maturity_label = "strong"
        elif days_of_confluence == 2:
            maturity_label = "developing"
        elif days_of_confluence == 1:
            maturity_label = "weak"
        else:
            maturity_label = "none"

        return {
            "day_minus_2":                 d2,
            "day_minus_1":                 d1,
            "day_0":                       d0,
            "rsi_trending_up":             rsi_trending_up,
            "macd_hist_rising":            macd_hist_rising,
            "volume_sustained":            volume_sustained,
            "price_above_ema50_holding":   price_above_ema50_holding,
            "net_score_direction":         net_score_direction,
            "days_of_confluence":          days_of_confluence,
            "maturity_label":              maturity_label,
        }

    except Exception as e:
        logger.warning(f"[hist_ctx] _compute_context_from_df failed for {ticker}: {e}")
        return {}


def _fetch_and_compute(ticker: str, data_client=None) -> dict:
    """Fetch data then compute. Called inside a timeout executor."""
    df = None

    # Try Alpaca first
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment, DataFeed
        from datetime import datetime, timedelta, timezone
        from bot.data import get_data_client

        client = data_client or get_data_client()
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=_FETCH_DAYS),
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        if bars and ticker in bars:
            raw = bars[ticker].df.copy()
            if not raw.empty:
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                raw.columns = [c.capitalize() for c in raw.columns]
                df = raw
    except Exception as e:
        logger.debug(f"[hist_ctx] Alpaca fetch failed for {ticker}: {e}")

    # Fallback to yfinance
    if df is None or df.empty:
        try:
            import yfinance as yf
            raw = yf.Ticker(ticker).history(period="10d", interval="1d")
            if raw is not None and not raw.empty:
                raw.index = pd.to_datetime(raw.index).tz_localize(None)
                df = raw
        except Exception as e:
            logger.debug(f"[hist_ctx] yfinance fallback failed for {ticker}: {e}")

    return _compute_context_from_df(ticker, df)


def get_historical_context(ticker: str, data_client=None) -> dict:
    """
    Standalone per-ticker call. Fetches its own data with a 10-second timeout.
    Returns empty dict if unavailable. Never raises.
    """
    try:
        with ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_fetch_and_compute, ticker, data_client)
            try:
                return fut.result(timeout=10)
            except FutureTimeout:
                logger.warning(f"[hist_ctx] {ticker}: timed out after 10s — returning empty")
                return {}
    except Exception as e:
        logger.warning(f"[hist_ctx] {ticker}: get_historical_context failed: {e}")
        return {}


def get_historical_context_batch(tickers: list[str], data_client=None) -> dict[str, dict]:
    """
    Efficient batch version — fetches all tickers in ONE Alpaca API call then
    computes context per ticker locally. Used by run_full_scan to avoid one API
    call per ticker. Falls back to empty dicts on any failure.
    """
    if not tickers:
        return {}
    try:
        from bot.data import fetch_daily_bars_batch
        bars_map = fetch_daily_bars_batch(tickers, days=_FETCH_DAYS)
    except Exception as e:
        logger.warning(f"[hist_ctx] batch fetch failed: {e}")
        return {t: {} for t in tickers}

    results: dict[str, dict] = {}
    for ticker in tickers:
        df = bars_map.get(ticker)
        if df is not None and not df.empty:
            results[ticker] = _compute_context_from_df(ticker, df)
        else:
            results[ticker] = {}
    return results
