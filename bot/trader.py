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
HTTP_TIMEOUT   = (5, 12)  # (connect, read) seconds for the underlying requests session


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


def apply_http_timeout(client, timeout: tuple = HTTP_TIMEOUT):
    """
    alpaca-py (≤0.29) calls requests.Session.request() without a timeout, so a
    dead network hangs until the OS gives up (minutes). Wrap the client's
    session so every HTTP call gets a real connect/read timeout and fails in
    seconds instead — keeping scan cycles on their cadence.
    """
    session = getattr(client, "_session", None)
    if session is None:
        logger.warning("[trader] client has no _session — cannot set HTTP timeout")
        return client
    original_request = session.request

    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout)
        return original_request(method, url, **kwargs)

    session.request = request_with_timeout
    return client


def build_client() -> object:
    """Build and return an Alpaca TradingClient."""
    from alpaca.trading.client import TradingClient
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    base_url   = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
    paper      = "paper-api" in base_url
    client = TradingClient(api_key=api_key, secret_key=secret_key, paper=paper)
    return apply_http_timeout(client)


def build_data_client() -> object:
    """Build Alpaca StockHistoricalDataClient."""
    from alpaca.data.historical import StockHistoricalDataClient
    api_key    = os.environ["ALPACA_API_KEY"]
    secret_key = os.environ["ALPACA_SECRET_KEY"]
    client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
    return apply_http_timeout(client)


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


def get_positions(client, raise_on_error: bool = False) -> list[dict]:
    """
    Return all open positions.

    raise_on_error=False (default): returns [] on failure — acceptable for
    display/enrichment paths. raise_on_error=True: re-raises so callers that
    MUST distinguish "no positions" from "API failed" (reconciliation!) never
    mistake an outage for an empty portfolio.
    """
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
        if raise_on_error:
            raise
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
            # GTC so the stop/target legs survive across days — positions stay
            # protected even when no workflow is running.
            req = LimitOrderRequest(
                symbol=ticker,
                qty=qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.GTC,
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


def cancel_open_orders(client, ticker: str, dry_run: bool = False) -> int:
    """Cancel all open orders for a ticker (e.g. bracket/OCO legs before a close).
    Returns the number of orders cancelled."""
    if dry_run:
        logger.info(f"[trader] DRY_RUN: would cancel open orders for {ticker}")
        return 0

    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    cancelled = 0
    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
        orders = _retry(client.get_orders, req)
        for o in orders:
            try:
                _retry(client.cancel_order_by_id, str(o.id))
                cancelled += 1
            except Exception as e:
                logger.warning(f"[trader] cancel order {o.id} for {ticker} failed: {e}")
    except Exception as e:
        logger.warning(f"[trader] cancel_open_orders failed for {ticker}: {e}")
    if cancelled:
        logger.info(f"[trader] Cancelled {cancelled} open order(s) for {ticker}")
    return cancelled


def has_open_exit_order(client, ticker: str) -> bool:
    """True if the ticker already has an open sell order (stop/target leg) working."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus

    try:
        req = GetOrdersRequest(status=QueryOrderStatus.OPEN, symbols=[ticker])
        orders = _retry(client.get_orders, req)
        return any(str(o.side).lower().endswith("sell") for o in orders)
    except Exception as e:
        logger.warning(f"[trader] has_open_exit_order failed for {ticker}: {e}")
        return False


def submit_oco_exit(client, ticker: str, qty: int, take_profit: float,
                    stop_loss: float, dry_run: bool = False) -> Optional[str]:
    """
    Attach a GTC OCO exit (take-profit limit + stop-loss) to an existing long
    position so it stays protected when the bot isn't running.
    Returns order ID or None on failure.
    """
    if dry_run:
        logger.info(
            f"[trader] DRY_RUN: would submit OCO exit {qty} {ticker} "
            f"tp={take_profit:.2f} sl={stop_loss:.2f}"
        )
        return None

    from alpaca.trading.requests import (LimitOrderRequest, TakeProfitRequest,
                                         StopLossRequest)
    from alpaca.trading.enums import OrderSide, TimeInForce, OrderClass

    def _submit():
        req = LimitOrderRequest(
            symbol=ticker,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
            limit_price=round(take_profit, 2),
            order_class=OrderClass.OCO,
            take_profit=TakeProfitRequest(limit_price=round(take_profit, 2)),
            stop_loss=StopLossRequest(
                stop_price=round(stop_loss, 2),
                limit_price=round(stop_loss * 0.995, 2),
            ),
        )
        order = client.submit_order(req)
        return str(order.id)

    try:
        order_id = _retry(_submit)
        logger.info(
            f"[trader] OCO exit submitted: {qty} {ticker} "
            f"tp={take_profit:.2f} sl={stop_loss:.2f} id={order_id}"
        )
        return order_id
    except Exception as e:
        logger.error(f"[trader] submit_oco_exit failed for {ticker}: {e}")
        return None


def get_entry_fill_info(client, ticker: str) -> Optional[dict]:
    """
    Return {'filled_at': iso str, 'filled_avg_price': float, 'filled_qty': float}
    for the most recent filled BUY order on a ticker. Used to adopt positions
    that exist on Alpaca but are missing from the trades DB.
    """
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderSide

    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[ticker],
            side=OrderSide.BUY,
            limit=50,
        )
        orders = _retry(client.get_orders, req)
        fills = [o for o in orders if str(o.status).lower().endswith("filled") and o.filled_at]
        if not fills:
            return None
        latest = max(fills, key=lambda o: o.filled_at)
        return {
            "filled_at":        latest.filled_at.isoformat(),
            "filled_avg_price": float(latest.filled_avg_price) if latest.filled_avg_price else None,
            "filled_qty":       float(latest.filled_qty) if latest.filled_qty else None,
        }
    except Exception as e:
        logger.warning(f"[trader] get_entry_fill_info failed for {ticker}: {e}")
        return None


def get_last_sell_fill_price(client, ticker: str) -> Optional[float]:
    """Return fill price of the most recent filled SELL order, if any."""
    from alpaca.trading.requests import GetOrdersRequest
    from alpaca.trading.enums import QueryOrderStatus, OrderSide

    try:
        req = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            symbols=[ticker],
            side=OrderSide.SELL,
            limit=50,
        )
        orders = _retry(client.get_orders, req)
        fills = [o for o in orders if str(o.status).lower().endswith("filled") and o.filled_at]
        if not fills:
            return None
        latest = max(fills, key=lambda o: o.filled_at)
        return float(latest.filled_avg_price) if latest.filled_avg_price else None
    except Exception as e:
        logger.warning(f"[trader] get_last_sell_fill_price failed for {ticker}: {e}")
        return None


def close_position(client, ticker: str, dry_run: bool = False) -> None:
    """Market-order close the full position for a ticker.
    Cancels open orders first — Alpaca rejects closes while shares are held
    by working bracket/OCO legs."""
    if dry_run:
        logger.info(f"[trader] DRY_RUN: would close position {ticker}")
        return

    cancel_open_orders(client, ticker)

    def _close():
        client.close_position(ticker)

    try:
        _retry(_close)
        logger.info(f"[trader] Position closed: {ticker}")
    except Exception as e:
        logger.error(f"[trader] close_position failed for {ticker}: {e}")
        raise


def _order_snapshot(order, order_id: str) -> dict:
    return {
        "status": str(order.status).split(".")[-1].lower(),
        "filled_avg_price": float(order.filled_avg_price) if order.filled_avg_price else None,
        "filled_qty": float(order.filled_qty) if order.filled_qty else 0.0,
        "order_id": order_id,
    }


def check_order_filled(client, order_id: str, timeout: int = 60) -> dict:
    """
    Poll for order fill status. If still unfilled at timeout, CANCEL the order
    so it can never fill later as an untracked position, then report the final
    state (cancelling may race a fill — the post-cancel poll catches that).
    Returns dict with status / filled_avg_price / filled_qty / order_id.
    """
    if order_id.startswith("dry-"):
        return {"status": "filled", "filled_avg_price": None, "filled_qty": None,
                "order_id": order_id}

    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            order = client.get_order_by_id(order_id)
            snap = _order_snapshot(order, order_id)
            if snap["status"] in ("filled", "cancelled", "expired", "rejected"):
                return snap
        except Exception as e:
            logger.warning(f"[trader] order status check failed: {e}")
        time.sleep(5)

    # Timeout: cancel so the order can't fill later untracked
    logger.warning(f"[trader] order {order_id} unfilled after {timeout}s — cancelling")
    try:
        client.cancel_order_by_id(order_id)
    except Exception as e:
        logger.warning(f"[trader] cancel after timeout failed: {e}")

    # Final state — the cancel may have raced a (partial) fill
    for _ in range(6):
        try:
            order = client.get_order_by_id(order_id)
            snap = _order_snapshot(order, order_id)
            if snap["status"] in ("filled", "cancelled", "expired", "rejected"):
                return snap
        except Exception as e:
            logger.warning(f"[trader] post-cancel status check failed: {e}")
        time.sleep(5)

    return {"status": "timeout", "filled_avg_price": None, "filled_qty": 0.0,
            "order_id": order_id}


def compute_limit_price(side: str, quote: dict, current_price: float) -> float:
    """Compute aggressive-but-safe limit price from quote."""
    if side == "buy":
        ask = quote.get("ask") or current_price
        return round(ask + 0.03, 2)
    else:
        bid = quote.get("bid") or current_price
        return round(bid - 0.03, 2)


