"""SQLite trade logger — auto-creates all tables on first run."""

import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "trades.db"


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db() -> None:
    """Create all tables if they don't exist."""
    conn = _connect()
    cur = conn.cursor()

    cur.executescript("""
    CREATE TABLE IF NOT EXISTS rejections (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp        TEXT,
        session          TEXT,
        ticker           TEXT,
        net_score        INTEGER,
        confidence       REAL,
        action           TEXT,
        rejection_reason TEXT,
        bull_score       INTEGER,
        bear_score       INTEGER,
        strategy         TEXT
    );

    CREATE TABLE IF NOT EXISTS trades (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT,
        session             TEXT,
        ticker              TEXT,
        action              TEXT,
        strategy            TEXT,
        time_horizon        TEXT,
        quantity            INTEGER,
        entry_price         REAL,
        limit_price         REAL,
        stop_loss           REAL,
        take_profit         REAL,
        confidence          REAL,
        net_score           INTEGER,
        bull_score          INTEGER,
        bear_score          INTEGER,
        signals_triggered   TEXT,
        signals_against     TEXT,
        reasoning           TEXT,
        risk_reward         REAL,
        macro_bias          TEXT,
        vix_level           REAL,
        alpaca_order_id     TEXT,
        status              TEXT,
        exit_price          REAL,
        exit_timestamp      TEXT,
        pnl_dollar          REAL,
        pnl_pct             REAL
    );

    CREATE TABLE IF NOT EXISTS daily_summary (
        date                    TEXT PRIMARY KEY,
        starting_value          REAL,
        ending_value            REAL,
        cash                    REAL,
        total_trades            INTEGER,
        winning_trades          INTEGER,
        losing_trades           INTEGER,
        gross_pnl               REAL,
        win_rate                REAL,
        best_trade              TEXT,
        worst_trade             TEXT,
        macro_bias              TEXT,
        vix_level               REAL,
        kill_switch_triggered   INTEGER,
        notes                   TEXT
    );

    CREATE TABLE IF NOT EXISTS scan_log (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp           TEXT,
        session             TEXT,
        tickers_scanned     INTEGER,
        signals_generated   INTEGER,
        trades_executed     INTEGER,
        total_bull_signals  INTEGER,
        total_bear_signals  INTEGER
    );
    """)
    conn.commit()

    # Add columns if missing (migration-safe)
    for col, col_type in [
        ("highest_price_seen",  "REAL"),
        ("trailing_stop_price", "REAL"),
        ("ai_confirmed",        "INTEGER"),
        ("ai_reasoning",        "TEXT"),
        ("return_5d",           "REAL"),
        ("return_1m",           "REAL"),
        ("return_3m",           "REAL"),
        ("velocity_penalty",    "REAL"),
        ("fundamental_score",   "INTEGER"),
        ("hype_penalty",        "REAL"),
        ("breakout_quality",    "TEXT"),
        ("breakout_level",      "REAL"),
    ]:
        try:
            conn.execute(f"ALTER TABLE trades ADD COLUMN {col} {col_type}")
            conn.commit()
        except Exception:
            pass  # column already exists

    conn.close()
    logger.info(f"[logger] DB initialized at {DB_PATH}")


def log_trade(
    session: str,
    ticker: str,
    action: str,
    strategy: str,
    time_horizon: str,
    quantity: int,
    entry_price: float,
    limit_price: float,
    stop_loss: Optional[float],
    take_profit: Optional[float],
    confidence: float,
    net_score: int,
    bull_score: int,
    bear_score: int,
    signals_triggered: list,
    signals_against: list,
    reasoning: str,
    risk_reward: float,
    macro_bias: str,
    vix_level: float,
    alpaca_order_id: str,
    status: str = "open",
    ai_confirmed: Optional[bool] = None,
    ai_reasoning: Optional[str] = None,
    return_5d: Optional[float] = None,
    return_1m: Optional[float] = None,
    return_3m: Optional[float] = None,
    velocity_penalty: Optional[float] = None,
    fundamental_score: Optional[int] = None,
    hype_penalty: Optional[float] = None,
    breakout_quality: Optional[str] = None,
    breakout_level: Optional[float] = None,
) -> int:
    """Insert a trade record. Returns the new row ID."""
    init_db()
    conn = _connect()
    try:
        cur = conn.cursor()
        cur.execute("""
        INSERT INTO trades (
            timestamp, session, ticker, action, strategy, time_horizon,
            quantity, entry_price, limit_price, stop_loss, take_profit,
            confidence, net_score, bull_score, bear_score,
            signals_triggered, signals_against, reasoning, risk_reward,
            macro_bias, vix_level, alpaca_order_id, status,
            ai_confirmed, ai_reasoning,
            return_5d, return_1m, return_3m, velocity_penalty,
            fundamental_score, hype_penalty, breakout_quality, breakout_level
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            session,
            ticker,
            action,
            strategy,
            time_horizon,
            quantity,
            entry_price,
            limit_price,
            stop_loss,
            take_profit,
            confidence,
            net_score,
            bull_score,
            bear_score,
            json.dumps(signals_triggered),
            json.dumps(signals_against),
            reasoning,
            risk_reward,
            macro_bias,
            vix_level,
            alpaca_order_id,
            status,
            int(ai_confirmed) if ai_confirmed is not None else None,
            ai_reasoning,
            return_5d,
            return_1m,
            return_3m,
            velocity_penalty,
            fundamental_score,
            hype_penalty,
            breakout_quality,
            breakout_level,
        ))
        conn.commit()
        row_id = cur.lastrowid
        logger.info(f"[logger] Trade logged: id={row_id} {ticker} {action} qty={quantity}")
        return row_id
    finally:
        conn.close()


def update_trade_exit(
    trade_id: int,
    exit_price: float,
    status: str,
    pnl_dollar: float,
    pnl_pct: float,
) -> None:
    init_db()
    conn = _connect()
    try:
        conn.execute("""
        UPDATE trades SET
            exit_price      = ?,
            exit_timestamp  = ?,
            status          = ?,
            pnl_dollar      = ?,
            pnl_pct         = ?
        WHERE id = ?
        """, (exit_price, datetime.utcnow().isoformat(), status, pnl_dollar, pnl_pct, trade_id))
        conn.commit()
        logger.info(f"[logger] Trade {trade_id} updated: status={status} pnl=${pnl_dollar:.2f}")
    finally:
        conn.close()


def get_open_trades() -> list[dict]:
    """Return all trades with status='open'."""
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM trades WHERE status = 'open' ORDER BY timestamp DESC"
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_trades_today() -> list[dict]:
    init_db()
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM trades WHERE timestamp LIKE ? ORDER BY timestamp DESC",
            (f"{today}%",)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()



def log_daily_summary(
    date: str,
    starting_value: float,
    ending_value: float,
    cash: float,
    total_trades: int,
    winning_trades: int,
    losing_trades: int,
    gross_pnl: float,
    win_rate: float,
    best_trade: str,
    worst_trade: str,
    macro_bias: str,
    vix_level: float,
    kill_switch_triggered: bool,
    notes: str = "",
) -> None:
    init_db()
    conn = _connect()
    try:
        conn.execute("""
        INSERT OR REPLACE INTO daily_summary (
            date, starting_value, ending_value, cash,
            total_trades, winning_trades, losing_trades,
            gross_pnl, win_rate, best_trade, worst_trade,
            macro_bias, vix_level, kill_switch_triggered, notes
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            date, starting_value, ending_value, cash,
            total_trades, winning_trades, losing_trades,
            gross_pnl, win_rate, best_trade, worst_trade,
            macro_bias, vix_level, int(kill_switch_triggered), notes,
        ))
        conn.commit()
        logger.info(f"[logger] Daily summary logged for {date}")
    finally:
        conn.close()


def get_daily_summaries(days: int = 7) -> list[dict]:
    init_db()
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM daily_summary ORDER BY date DESC LIMIT ?", (days,)
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def log_scan(
    session: str,
    tickers_scanned: int,
    signals_generated: int,
    trades_executed: int,
    total_bull: int,
    total_bear: int,
) -> None:
    init_db()
    conn = _connect()
    try:
        conn.execute("""
        INSERT INTO scan_log (
            timestamp, session, tickers_scanned,
            signals_generated, trades_executed,
            total_bull_signals, total_bear_signals
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            session, tickers_scanned,
            signals_generated, trades_executed,
            total_bull, total_bear,
        ))
        conn.commit()
    finally:
        conn.close()


def update_trade_trailing(trade_id: int, highest_price_seen: float, trailing_stop_price: float) -> None:
    """Update trailing stop fields for an open trade."""
    init_db()
    conn = _connect()
    try:
        conn.execute("""
        UPDATE trades SET
            highest_price_seen  = ?,
            trailing_stop_price = ?
        WHERE id = ?
        """, (highest_price_seen, trailing_stop_price, trade_id))
        conn.commit()
    finally:
        conn.close()



def update_trade_quantity(trade_id: int, new_qty: int) -> None:
    """Update remaining quantity for an open trade (e.g. after a partial exit)."""
    init_db()
    conn = _connect()
    try:
        conn.execute("UPDATE trades SET quantity = ? WHERE id = ?", (new_qty, trade_id))
        conn.commit()
    finally:
        conn.close()


def update_trade_stop(trade_id: int, new_stop: float) -> None:
    """Update the stop_loss for an open trade (e.g. move to breakeven after partial exit)."""
    init_db()
    conn = _connect()
    try:
        conn.execute("UPDATE trades SET stop_loss = ? WHERE id = ?", (new_stop, trade_id))
        conn.commit()
    finally:
        conn.close()


def log_rejection(
    session: str,
    ticker: str,
    net_score: int,
    confidence: float,
    action: str,
    rejection_reason: str,
    bull_score: int = 0,
    bear_score: int = 0,
    strategy: str = "",
) -> None:
    """Log a rejected trade signal to the rejections table."""
    init_db()
    conn = _connect()
    try:
        conn.execute("""
        INSERT INTO rejections (
            timestamp, session, ticker, net_score, confidence,
            action, rejection_reason, bull_score, bear_score, strategy
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            session, ticker, net_score, confidence,
            action, rejection_reason, bull_score, bear_score, strategy,
        ))
        conn.commit()
    except Exception as e:
        logger.warning(f"[logger] log_rejection failed for {ticker}: {e}")
    finally:
        conn.close()
