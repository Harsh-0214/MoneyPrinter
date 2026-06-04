"""Alpaca order execution — all calls wrapped in retry logic."""

import logging
import os
import time
import uuid
from typing import Optional

logger = logging.getLogger(__name__)

MAX_RETRIES    = 3
BACKOFF_BASE   = 2   # seconds
CALL_TIMEOUT   = 12  # seconds per attempt before giving up


def _retry(fn, *args, **kwargs):
    """Call fn with hard 12s timeout per attempt, exponential backoff on failure."""
    import concurrent.futures
    for attempt in range(MAX_RETRIES):
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                future = ex.submit(fn, *args, **kwargs)
                return future.result(timeout=CALL_TIMEOUT)
        except concurrent.futures.TimeoutError:
            e = TimeoutError(f"Alpaca API call timed out after {CALL_TIMEOUT}s")
            if attempt == MAX_RETRIES - 1:
                raise e
            wait = BACKOFF_BASE ** attempt
            logger.warning(f"[trader] attempt {attempt+1} timed out — retrying in {wait}s")
            time.sleep(wait)
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                raise
            wait = BACKOFF_BASE ** attempt
            logger.warning(f"[trader] attempt {attempt+1} failed: {e} — retrying in {wait}s")
            time.sleep(wait)


def build_client() -> object:
    """Build and return an Alpaca TradingClient."""
    from alpaca.trading.client import TradingClient
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    paper      = "paper-api" in base_url
    return TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)


def build_data_client() -> object:
    """Build Alpaca StockHistoricalDataClient."""
    from alpaca.data.historical import StockHistoricalDataClient
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)


def get_account(client) -> dict:
    """Return account cash, portfolio_value, buying_power."""
    def _get():
        acc = client.get_account()
        return {
            "cash":            float(acc.cash),
            "portfolio_value": float(acc.portfolio_value),
            "buying_power":    float(acc.buying_power),
            "equity":          float(acc.equity),
        }
    try:
        return _retry(_get)
    except Exception as e:
        logger.error(f"[trader] get_account failed: {e}")
        return {"cash": 0, "portfolio_value": 0, "buying_power": 0, "equity": 0}


def get_positions(client) -> list[dict]:
    """Return all open positions."""
    def _get():
        positions = client.get_all_positions()
        result = []
        for p in positions:
            result.append({
                "symbol":           p.symbol,
                "qty":              float(p.qty),
                "avg_entry_price":  float(p.avg_entry_price),
                "current_price":    float(p.current_price) if p.current_price else None,
                "unrealized_pl":    float(p.unrealized_pl) if p.unrealized_pl else None,
                "unrealized_plpc":  float(p.unrealized_plpc) if p.unrealized_plpc else None,
                "side":             str(p.side),
            })
        return result
    try:
        return _retry(_get)
    except Exception as e:
        logger.error(f"[trader] get_positions failed: {e}")
        return []


def get_market_status(client) -> bool:
    """Return True if market is currently open."""
    try:
        clock = _retry(client.get_clock)
        return bool(clock.is_open)
    except Exception as e:
        logger.warning(f"[trader] get_market_status failed: {e}")
        return False


def get_latest_quote(data_client, ticker: str) -> dict:
    """Fetch latest bid/ask quote for a ticker."""
    if data_client is None:
        return {"ask": None, "bid": None}
    try:
        from alpaca.data.requests import StockLatestQuoteRequest
        req  = StockLatestQuoteRequest(symbol_or_symbols=[ticker])
        resp = _retry(data_client.get_stock_latest_quote, req)
        q = resp.get(ticker)
        if q:
            return {"ask": float(q.ask_price), "bid": float(q.bid_price)}
    except Exception as e:
        logger.warning(f"[trader] quote fetch failed for {ticker}: {e}")
    return {"ask": None, "bid": None}


def submit_order(
    client,
    ticker: str,
    side: str,
    qty: int,
    limit_price: float,
    dry_run: bool = False,
    stop_loss: Optional[float] = None,
    take_profit: Optional[float] = None,
    volume_ratio: Optional[float] = None,
) -> str:
    """
    Submit a limit order (or bracket order when stop_loss+take_profit provided for buys).
    Returns order ID string.
    side: 'buy' or 'sell'
    """
    if dry_run:
        # Simulated slippage
        slippage_pct = 0.001
        if volume_ratio is not None:
            if volume_ratio < 0.5:
                slippage_pct = 0.005
            elif volume_ratio < 1.0:
                slippage_pct = 0.002
        if side.lower() == "buy":
            simulated_fill = round(limit_price * (1 + slippage_pct), 2)
        else:
            simulated_fill = round(limit_price * (1 - slippage_pct), 2)
        fake_id = f"dry-{uuid.uuid4().hex[:8]}"
        logger.info(
            f"[trader] DRY_RUN: would submit {side} {qty} {ticker} @ ${limit_price:.2f} "
            f"(slippage {slippage_pct*100:.1f}% → fill ~${simulated_fill:.2f}) "
            f"stop={stop_loss} tp={take_profit} -> fake_id={fake_id}"
        )
        return fake_id

    from alpaca.trading.requests import LimitOrderRequest
    from alpaca.trading.enums   import OrderSide, TimeInForce

    alpaca_side = OrderSide.BUY if side.lower() == "buy" else OrderSide.SELL

    def _submit():
        if stop_loss and take_profit and side.lower() == "buy":
            from alpaca.trading.requests import TakeProfitRequest, StopLossRequest
            from alpaca.trading.enums import OrderClass
            req = LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
                limit_price=round(limit_price, 2),
                order_class=OrderClass.BRACKET,
                take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
                stop_loss=StopLossRequest(
                    stop_price=round(stop_loss, 2),
                    limit_price=round(stop_loss * 0.995, 2),
                ),
            )
        else:
            req = LimitOrderRequest(
                symbol        = ticker,
                qty           = qty,
                side          = alpaca_side,
                time_in_force = TimeInForce.DAY,
                limit_price   = round(limit_price, 2),
            )
        order = client.submit_order(req)
        return str(order.id)

    try:
        order_id = _retry(_submit)
        logger.info(f"[trader] Order submitted: {side} {qty} {ticker} @ {limit_price:.2f} id={order_id}")
        return order_id
    except Exception as e:
        logger.error(f"[trader] submit_order failed for {ticker}: {e}")
        raise


def close_position(client, ticker: str, dry_run: bool = False) -> None:
    """Market-order close the full position for a ticker."""
    if dry_run:
        logger.info(f"[trader] DRY_RUN: would close position {ticker}")
        return

    def _close():
        client.close_position(ticker)

    try:
        _retry(_close)
        logger.info(f"[trader] Position closed: {ticker}")
    except Exception as e:
        logger.error(f"[trader] close_position failed for {ticker}: {e}")
        raise


def check_order_filled(client, order_id: str, timeout: int = 60) -> dict:
    """Poll for order fill status. Returns status dict."""
    if order_id.startswith("dry-"):
        return {"status": "filled", "filled_avg_price": None, "order_id": order_id}

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            order = client.get_order_by_id(order_id)
            status = str(order.status)
            if status in ("filled", "partially_filled", "cancelled", "expired", "rejected"):
                return {
                    "status": status,
                    "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
                    "order_id": order_id,
                }
        except Exception as e:
            logger.warning(f"[trader] order status check failed: {e}")
        time.sleep(5)

    return {"status": "timeout", "filled_avg_price": None, "order_id": order_id}


def compute_limit_price(side: str, quote: dict, current_price: float) -> float:
    """Compute aggressive-but-safe limit price from quote."""
    if side == "buy":
        ask = quote.get("ask") or current_price
        return round(ask + 0.03, 2)
    else:
        bid = quote.get("bid") or current_price
        return round(bid - 0.03, 2)


