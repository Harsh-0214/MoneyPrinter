"""Alpaca market data provider — replaces yfinance for price/volume/OHLCV data."""

import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_data_client = None


def get_data_client():
    global _data_client
    if _data_client is None:
        from alpaca.data.historical import StockHistoricalDataClient
        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        _data_client = StockHistoricalDataClient(api_key, secret_key)
    return _data_client


def _bars_to_df(bar_list: list) -> Optional[pd.DataFrame]:
    """Convert a list of Alpaca Bar objects to a capitalised OHLCV DataFrame."""
    if not bar_list:
        return None
    df = pd.DataFrame([{
        "timestamp": b.timestamp,
        "Open":   float(b.open),
        "High":   float(b.high),
        "Low":    float(b.low),
        "Close":  float(b.close),
        "Volume": float(b.volume),
    } for b in bar_list]).set_index("timestamp")
    df.index = pd.to_datetime(df.index).tz_convert(None)
    return df


def fetch_daily_bars(ticker: str, days: int = 730) -> Optional[pd.DataFrame]:
    """
    Fetch daily OHLCV bars from Alpaca. Returns DataFrame with columns:
    Open, High, Low, Close, Volume — indexed by date.
    Returns None on failure.
    """
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment
        client = get_data_client()
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=days),
            adjustment=Adjustment.ALL,
        )
        bars = client.get_stock_bars(req)
        bar_list = (bars.data or {}).get(ticker) if bars and hasattr(bars, "data") else None
        if not bar_list:
            logger.warning(f"[data] {ticker}: no daily bars returned")
            return None
        df = _bars_to_df(bar_list)
        logger.info(f"[data] {ticker}: got {len(df)} daily bars")
        return df
    except Exception as e:
        logger.warning(f"[data] daily bars failed for {ticker}: {e}", exc_info=True)
        return None


def fetch_intraday_bars(ticker: str, days: int = 2) -> Optional[pd.DataFrame]:
    """Fetch 5-min intraday bars from Alpaca."""
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from alpaca.data.enums import Adjustment, DataFeed
        client = get_data_client()
        req = StockBarsRequest(
            symbol_or_symbols=ticker,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=datetime.now(timezone.utc) - timedelta(days=days),
            adjustment=Adjustment.ALL,
            feed=DataFeed.IEX,
        )
        bars = client.get_stock_bars(req)
        bar_list = (bars.data or {}).get(ticker) if bars and hasattr(bars, "data") else None
        if not bar_list:
            return None
        return _bars_to_df(bar_list)
    except Exception as e:
        logger.warning(f"[data] intraday bars failed for {ticker}: {e}")
        return None


def fetch_snapshot(ticker: str) -> Optional[dict]:
    """
    Fetch latest snapshot from Alpaca. Returns dict with:
    price, prev_close, last_volume, daily_open, daily_high, daily_low
    Returns None on failure.
    """
    try:
        from alpaca.data.requests import StockSnapshotRequest
        from alpaca.data.enums import DataFeed
        client = get_data_client()
        req = StockSnapshotRequest(symbol_or_symbols=[ticker], feed=DataFeed.IEX)
        snaps = client.get_stock_snapshot(req)
        if not snaps or ticker not in snaps:
            return None
        s = snaps[ticker]
        return {
            "price":      float(s.latest_trade.price) if s.latest_trade else None,
            "prev_close": float(s.previous_daily_bar.close) if s.previous_daily_bar else None,
            "last_volume":float(s.daily_bar.volume) if s.daily_bar else None,
            "daily_open": float(s.daily_bar.open)   if s.daily_bar else None,
            "daily_high": float(s.daily_bar.high)   if s.daily_bar else None,
            "daily_low":  float(s.daily_bar.low)    if s.daily_bar else None,
        }
    except Exception as e:
        logger.warning(f"[data] snapshot failed for {ticker}: {e}")
        return None


def fetch_snapshots_batch(tickers: list[str]) -> dict[str, dict]:
    """
    Fetch snapshots for multiple tickers in a single Alpaca API call.
    Returns dict of ticker -> snapshot dict (same format as fetch_snapshot).
    Missing/failed tickers are simply absent from the result.
    """
    if not tickers:
        return {}
    try:
        from alpaca.data.requests import StockSnapshotRequest
        from alpaca.data.enums import DataFeed
        client = get_data_client()
        req = StockSnapshotRequest(symbol_or_symbols=tickers, feed=DataFeed.IEX)
        snaps = client.get_stock_snapshot(req)
        result = {}
        for ticker, s in (snaps or {}).items():
            try:
                result[ticker] = {
                    "price":       float(s.latest_trade.price) if s.latest_trade else None,
                    "prev_close":  float(s.previous_daily_bar.close) if s.previous_daily_bar else None,
                    "last_volume": float(s.daily_bar.volume) if s.daily_bar else None,
                    "daily_open":  float(s.daily_bar.open)   if s.daily_bar else None,
                    "daily_high":  float(s.daily_bar.high)   if s.daily_bar else None,
                    "daily_low":   float(s.daily_bar.low)    if s.daily_bar else None,
                }
            except Exception:
                pass
        return result
    except Exception as e:
        logger.warning(f"[data] batch snapshot failed: {e}")
        return {}


def _fetch_daily_bars_yfinance(tickers: list[str], days: int = 365) -> dict[str, pd.DataFrame]:
    """yfinance fallback for fetch_daily_bars_batch — used when Alpaca credentials are absent."""
    try:
        import yfinance as yf
        from datetime import date as date_cls
        end_dt   = datetime.now(timezone.utc)
        start_dt = end_dt - timedelta(days=days)
        start_str = start_dt.strftime("%Y-%m-%d")
        end_str   = end_dt.strftime("%Y-%m-%d")
        batch_size = 50  # yfinance handles ~50 symbols comfortably per call
        result: dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), batch_size):
            chunk = tickers[i : i + batch_size]
            try:
                raw = yf.download(
                    " ".join(chunk),
                    start=start_str,
                    end=end_str,
                    auto_adjust=True,
                    progress=False,
                    threads=True,
                )
                if raw is None or raw.empty:
                    continue
                # Multi-ticker download returns MultiIndex columns
                if isinstance(raw.columns, pd.MultiIndex):
                    for tk in chunk:
                        try:
                            tk_upper = tk.upper()
                            df = raw.xs(tk_upper, level=1, axis=1).copy() if tk_upper in raw.columns.get_level_values(1) else None
                            if df is None or df.empty:
                                continue
                            df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                                                     "Close": "Close", "Volume": "Volume"})
                            df.index = pd.to_datetime(df.index).tz_localize(None)
                            result[tk] = df.dropna(subset=["Close"])
                        except Exception:
                            pass
                else:
                    # Single-ticker batch
                    if len(chunk) == 1:
                        df = raw.copy()
                        df.index = pd.to_datetime(df.index).tz_localize(None)
                        result[chunk[0]] = df.dropna(subset=["Close"])
            except Exception as e:
                logger.warning(f"[data] yfinance chunk {chunk[:3]}... failed: {e}")
        logger.info(f"[data] yfinance fallback: fetched {len(result)}/{len(tickers)} tickers")
        return result
    except Exception as e:
        logger.warning(f"[data] yfinance fallback failed: {e}")
        return {}


def fetch_daily_bars_batch(tickers: list[str], days: int = 365) -> dict[str, pd.DataFrame]:
    """
    Fetch daily bars for multiple tickers.
    Tries Alpaca first; falls back to yfinance when credentials are absent.
    Returns dict of ticker -> DataFrame (Open/High/Low/Close/Volume, date-indexed).
    Missing/failed tickers are absent from the result.
    """
    if not tickers:
        return {}
    # Check if Alpaca credentials are configured before attempting
    if not (os.environ.get("ALPACA_API_KEY") and os.environ.get("ALPACA_SECRET_KEY")):
        logger.info("[data] No Alpaca credentials — using yfinance fallback")
        return _fetch_daily_bars_yfinance(tickers, days)
    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from alpaca.data.enums import Adjustment
        client = get_data_client()
        req = StockBarsRequest(
            symbol_or_symbols=tickers,
            timeframe=TimeFrame.Day,
            start=datetime.now(timezone.utc) - timedelta(days=days),
            adjustment=Adjustment.ALL,
        )
        bars = client.get_stock_bars(req)
        bars_data = (bars.data or {}) if bars and hasattr(bars, "data") else {}
        result = {}
        for ticker in tickers:
            bar_list = bars_data.get(ticker)
            if bar_list:
                df = _bars_to_df(bar_list)
                if df is not None:
                    result[ticker] = df
        if not result:
            logger.warning("[data] Alpaca returned no data — trying yfinance fallback")
            return _fetch_daily_bars_yfinance(tickers, days)
        return result
    except Exception as e:
        logger.warning(f"[data] batch daily bars failed: {e} — trying yfinance fallback")
        return _fetch_daily_bars_yfinance(tickers, days)


def fetch_vix(days: int = 5) -> Optional[pd.DataFrame]:
    """Fetch VIX history — Alpaca doesn't carry ^VIX so falls back to yfinance."""
    try:
        import yfinance as yf
        df = yf.Ticker("^VIX").history(period=f"{days}d", interval="1d")
        if df is not None and not df.empty:
            return df
    except Exception as e:
        logger.warning(f"[data] VIX fetch failed: {e}")
    return None
