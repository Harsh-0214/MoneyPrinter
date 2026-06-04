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
        if not bars or ticker not in bars:
            return None
        df = bars[ticker].df.copy()
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_convert(None)
        df.columns = [c.capitalize() for c in df.columns]
        return df
    except Exception as e:
        logger.warning(f"[data] daily bars failed for {ticker}: {e}")
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
        if not bars or ticker not in bars:
            return None
        df = bars[ticker].df.copy()
        if df.empty:
            return None
        df.index = pd.to_datetime(df.index).tz_convert(None)
        df.columns = [c.capitalize() for c in df.columns]
        return df
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


def fetch_daily_bars_batch(tickers: list[str], days: int = 365) -> dict[str, pd.DataFrame]:
    """
    Fetch daily bars for multiple tickers in a single Alpaca API call.
    Returns dict of ticker -> DataFrame (Open/High/Low/Close/Volume, date-indexed).
    Missing/failed tickers are absent from the result.
    """
    if not tickers:
        return {}
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
        result = {}
        for ticker in tickers:
            try:
                if bars and ticker in bars:
                    df = bars[ticker].df.copy()
                    if not df.empty:
                        df.index = pd.to_datetime(df.index).tz_convert(None)
                        df.columns = [c.capitalize() for c in df.columns]
                        result[ticker] = df
            except Exception:
                pass
        return result
    except Exception as e:
        logger.warning(f"[data] batch daily bars failed: {e}")
        return {}


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
