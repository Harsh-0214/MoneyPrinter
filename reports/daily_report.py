"""Daily HTML report generator for the trading bot.

Reads from data/trades.db, optionally enriches with live Alpaca prices,
and writes a self-contained HTML report to reports/YYYY-MM-DD.html.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, date
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DB_PATH      = Path(__file__).parent.parent / "data" / "trades.db"
REPORTS_DIR  = Path(__file__).parent


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def _get_trades_for_date(date_str: str) -> list[dict]:
    if not DB_PATH.exists():
        return []
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM trades WHERE timestamp LIKE ? ORDER BY timestamp DESC",
            (f"{date_str}%",)
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"[report] trades query failed: {e}")
        return []


def _get_daily_summary(date_str: str) -> Optional[dict]:
    if not DB_PATH.exists():
        return None
    try:
        conn = _connect()
        row = conn.execute(
            "SELECT * FROM daily_summary WHERE date = ?", (date_str,)
        ).fetchone()
        conn.close()
        return dict(row) if row else None
    except Exception as e:
        logger.warning(f"[report] daily_summary query failed: {e}")
        return None


def _get_all_time_stats() -> dict:
    if not DB_PATH.exists():
        return {}
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT pnl_dollar, ticker, strategy FROM trades WHERE pnl_dollar IS NOT NULL"
        ).fetchall()
        conn.close()
        if not rows:
            return {}
        pnls = [float(r["pnl_dollar"]) for r in rows]
        winners = [p for p in pnls if p > 0]
        losers  = [p for p in pnls if p <= 0]
        return {
            "total":      len(pnls),
            "win_rate":   len(winners) / len(pnls) if pnls else 0,
            "avg_winner": sum(winners) / len(winners) if winners else 0,
            "avg_loser":  sum(losers)  / len(losers)  if losers  else 0,
        }
    except Exception as e:
        logger.warning(f"[report] all-time stats query failed: {e}")
        return {}


def _get_open_positions() -> list[dict]:
    if not DB_PATH.exists():
        return []
    try:
        conn = _connect()
        rows = conn.execute(
            "SELECT * FROM trades WHERE status IN ('open','dry_run') ORDER BY timestamp DESC"
        ).fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"[report] open positions query failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Alpaca live price enrichment (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch_live_prices(tickers: list[str]) -> dict:
    """Returns {ticker: current_price} dict. Empty on any failure."""
    if not tickers:
        return {}
    try:
        import yfinance as yf
        prices = {}
        for ticker in tickers:
            try:
                fi = yf.Ticker(ticker).fast_info
                p = getattr(fi, "last_price", None)
                if p and float(p) > 0:
                    prices[ticker] = float(p)
            except Exception:
                pass
        return prices
    except Exception as e:
        logger.debug(f"[report] live price fetch skipped: {e}")
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# HTML generation
# ─────────────────────────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, monospace;
    background: #0d1117; color: #c9d1d9; line-height: 1.5;
    font-size: 14px;
}
a { color: #58a6ff; }
.container { max-width: 1200px; margin: 0 auto; padding: 20px; }
h1 { font-size: 1.8rem; color: #e6edf3; margin-bottom: 4px; }
h2 { font-size: 1.2rem; color: #8b949e; margin: 24px 0 10px; border-bottom: 1px solid #21262d; padding-bottom: 6px; }
h3 { font-size: 1rem; color: #c9d1d9; }

.header-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 16px 0; }
.metric-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 14px 16px;
}
.metric-card .label { font-size: 11px; color: #8b949e; text-transform: uppercase; letter-spacing: .5px; }
.metric-card .value { font-size: 1.4rem; font-weight: 700; margin-top: 4px; }

.green  { color: #3fb950; }
.red    { color: #f85149; }
.blue   { color: #58a6ff; }
.yellow { color: #d29922; }
.muted  { color: #8b949e; }

.executive-summary {
    background: #161b22; border-left: 3px solid #58a6ff;
    border-radius: 4px; padding: 14px 18px; margin: 12px 0;
    font-size: 0.95rem; color: #e6edf3;
}

table { width: 100%; border-collapse: collapse; margin: 8px 0; font-size: 13px; }
th { background: #161b22; color: #8b949e; text-align: left; padding: 8px 10px; font-weight: 600; border-bottom: 2px solid #21262d; }
td { padding: 8px 10px; border-bottom: 1px solid #21262d; vertical-align: top; }
tr:hover td { background: #1c2128; }

.badge {
    display: inline-block; padding: 2px 8px; border-radius: 12px;
    font-size: 11px; font-weight: 700; letter-spacing: .5px;
}
.badge-buy   { background: #1a3a1e; color: #3fb950; border: 1px solid #238636; }
.badge-short { background: #3a1a1a; color: #f85149; border: 1px solid #da3633; }
.badge-open  { background: #1a2a3a; color: #58a6ff; border: 1px solid #1f6feb; }
.badge-closed { background: #21262d; color: #8b949e; border: 1px solid #30363d; }

.signal-list { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 4px; }
.signal-chip {
    background: #1f2937; color: #93c5fd; border: 1px solid #1d4ed8;
    border-radius: 10px; padding: 1px 7px; font-size: 11px;
}
.signal-chip.against {
    background: #2a1a1a; color: #fca5a5; border-color: #991b1b;
}

.reasoning { color: #8b949e; font-size: 12px; margin-top: 4px; font-style: italic; max-width: 600px; }

.trade-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0;
}
.trade-card .trade-header {
    display: flex; align-items: center; gap: 12px; margin-bottom: 8px; flex-wrap: wrap;
}
.trade-meta { display: grid; grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); gap: 8px; margin: 8px 0; }
.trade-meta-item .meta-label { font-size: 10px; color: #8b949e; text-transform: uppercase; }
.trade-meta-item .meta-value { font-size: 13px; color: #e6edf3; font-weight: 600; }

.stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; }
.stat-card {
    background: #161b22; border: 1px solid #21262d; border-radius: 8px; padding: 14px;
}
.stat-card .stat-label { font-size: 11px; color: #8b949e; text-transform: uppercase; }
.stat-card .stat-value { font-size: 1.2rem; font-weight: 700; margin-top: 2px; }

footer {
    margin-top: 40px; padding: 16px 0; border-top: 1px solid #21262d;
    color: #8b949e; font-size: 11px; text-align: center;
}

@media (max-width: 600px) {
    .header-grid { grid-template-columns: repeat(2, 1fr); }
    .trade-meta  { grid-template-columns: repeat(2, 1fr); }
}
"""


def _safe_float(val, default=0.0) -> float:
    try:
        return float(val) if val is not None else default
    except (TypeError, ValueError):
        return default


def _pnl_class(val: float) -> str:
    return "green" if val >= 0 else "red"


def _fmt_dollar(val: float) -> str:
    sign = "+" if val >= 0 else ""
    return f"{sign}${val:,.2f}"


def _signals_html(signals_json, css_class: str = "") -> str:
    try:
        sigs = json.loads(signals_json) if isinstance(signals_json, str) else (signals_json or [])
    except Exception:
        sigs = []
    if not sigs:
        return ""
    chips = "".join(
        f'<span class="signal-chip {css_class}">{s}</span>' for s in sigs
    )
    return f'<div class="signal-list">{chips}</div>'


def _build_executive_summary(date_str: str, trades: list, summary: Optional[dict],
                              live_prices: dict) -> str:
    closed  = [t for t in trades if t.get("pnl_dollar") is not None]
    open_t  = [t for t in trades if t.get("status") in ("open", "dry_run")]
    winners = [t for t in closed if _safe_float(t.get("pnl_dollar")) > 0]
    gross   = sum(_safe_float(t.get("pnl_dollar")) for t in closed)
    regime  = (summary or {}).get("macro_bias", "unknown")
    vix     = _safe_float((summary or {}).get("vix_level"), 0)

    parts = []
    if not trades:
        parts.append(f"No trades were executed on {date_str}.")
    else:
        direction = "positive" if gross >= 0 else "negative"
        parts.append(
            f"The bot executed {len(trades)} trade(s) on {date_str}, "
            f"closing {len(closed)} position(s) for a {direction} P&amp;L of {_fmt_dollar(gross)}."
        )
        if closed:
            wr = len(winners) / len(closed) * 100
            parts.append(
                f"Win rate for the day was {wr:.0f}% ({len(winners)} winner(s), "
                f"{len(closed) - len(winners)} loser(s))."
            )
        if open_t:
            parts.append(f"{len(open_t)} position(s) remain open heading into the next session.")

    regime_note = {
        "bull":    "SPY was in a confirmed bull regime, providing favorable conditions for long entries.",
        "caution": "SPY was in a caution zone (below EMA50); bull signals were discounted 20%.",
        "bear":    "SPY was in a bear regime; only high-conviction shorts above 80% confidence were considered.",
    }.get(regime, "")
    if regime_note:
        parts.append(regime_note)

    if vix >= 25:
        parts.append(f"VIX at {vix:.1f} elevated caution — position sizes were reduced via VIX multiplier.")

    return "<br>".join(parts)


def _build_trades_section(trades: list, live_prices: dict) -> str:
    if not trades:
        return "<p class='muted'>No trades today.</p>"

    cards = []
    for t in trades:
        ticker  = t.get("ticker", "?")
        action  = t.get("action", "?")
        entry   = _safe_float(t.get("entry_price"))
        stop    = _safe_float(t.get("stop_loss"))
        target  = _safe_float(t.get("take_profit"))
        conf    = _safe_float(t.get("confidence"))
        net_sc  = t.get("net_score", 0) or 0
        qty     = t.get("quantity", 0) or 0
        strat   = t.get("strategy", "?")
        status  = t.get("status", "open")
        reason  = t.get("reasoning", "")
        pnl_d   = t.get("pnl_dollar")
        exit_p  = t.get("exit_price")

        pos_val = round(entry * qty, 2)
        badge_action = "badge-buy" if action == "buy" else "badge-short"
        closed_badge = "badge-closed" if pnl_d is not None else "badge-open"
        closed_text  = "CLOSED" if pnl_d is not None else "OPEN"

        pnl_html = ""
        if pnl_d is not None:
            pnl_f = _safe_float(pnl_d)
            pnl_html = f'<span class="{_pnl_class(pnl_f)}" style="font-weight:700">{_fmt_dollar(pnl_f)}</span>'

        exit_html = f"${_safe_float(exit_p):.2f}" if exit_p is not None else "—"

        sigs_html    = _signals_html(t.get("signals_triggered"), "")
        against_html = _signals_html(t.get("signals_against"), "against")

        cards.append(f"""
<div class="trade-card">
  <div class="trade-header">
    <h3 style="font-size:1.2rem;color:#e6edf3">{ticker}</h3>
    <span class="badge {badge_action}">{action.upper()}</span>
    <span class="badge {closed_badge}">{closed_text}</span>
    <span class="muted" style="font-size:12px">Strategy: {strat}</span>
    {pnl_html}
  </div>
  <div class="trade-meta">
    <div class="trade-meta-item"><div class="meta-label">Entry</div><div class="meta-value">${entry:.2f}</div></div>
    <div class="trade-meta-item"><div class="meta-label">Qty</div><div class="meta-value">{qty}</div></div>
    <div class="trade-meta-item"><div class="meta-label">Position Value</div><div class="meta-value">${pos_val:,.2f}</div></div>
    <div class="trade-meta-item"><div class="meta-label">Stop Loss</div><div class="meta-value red">${stop:.2f}</div></div>
    <div class="trade-meta-item"><div class="meta-label">Take Profit</div><div class="meta-value green">${target:.2f}</div></div>
    <div class="trade-meta-item"><div class="meta-label">Confidence</div><div class="meta-value blue">{conf:.0%}</div></div>
    <div class="trade-meta-item"><div class="meta-label">Net Score</div><div class="meta-value">{net_sc}</div></div>
    <div class="trade-meta-item"><div class="meta-label">Exit Price</div><div class="meta-value">{exit_html}</div></div>
  </div>
  <div class="reasoning">{reason}</div>
  <div style="margin-top:8px">
    <span style="font-size:11px;color:#8b949e">SIGNALS FOR:</span> {sigs_html}
    <span style="font-size:11px;color:#8b949e;margin-top:4px;display:block">SIGNALS AGAINST:</span> {against_html}
  </div>
</div>""")

    return "\n".join(cards)


def _build_portfolio_section(open_positions: list, live_prices: dict) -> str:
    if not open_positions:
        return "<p class='muted'>No open positions.</p>"

    rows = []
    for pos in open_positions:
        ticker  = pos.get("ticker", "?")
        entry   = _safe_float(pos.get("entry_price"))
        qty     = pos.get("quantity", 0) or 0
        strat   = pos.get("strategy", "?")
        current = live_prices.get(ticker, entry)
        if entry > 0 and current:
            unrealized = (current - entry) / entry * 100
            if pos.get("action") in ("short", "sell"):
                unrealized = -unrealized
            unr_dollar = (current - entry) * qty
            if pos.get("action") in ("short", "sell"):
                unr_dollar = -unr_dollar
        else:
            unrealized = 0.0
            unr_dollar = 0.0

        c = _pnl_class(unrealized)
        rows.append(f"""<tr>
            <td><strong>{ticker}</strong></td>
            <td>{pos.get('action','')}</td>
            <td>{qty}</td>
            <td>${entry:.2f}</td>
            <td>${_safe_float(current):.2f}</td>
            <td class="{c}">{unrealized:+.2f}%</td>
            <td class="{c}">{_fmt_dollar(unr_dollar)}</td>
            <td>${_safe_float(pos.get('stop_loss')):.2f}</td>
            <td>${_safe_float(pos.get('take_profit')):.2f}</td>
            <td>{strat}</td>
        </tr>""")

    return f"""<table>
<thead><tr>
  <th>Ticker</th><th>Action</th><th>Qty</th><th>Entry</th><th>Current</th>
  <th>Unrlzd %</th><th>Unrlzd $</th><th>Stop</th><th>Target</th><th>Strategy</th>
</tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>"""


def _build_stats_section(trades_today: list, all_time: dict) -> str:
    closed  = [t for t in trades_today if t.get("pnl_dollar") is not None]
    winners = [t for t in closed if _safe_float(t.get("pnl_dollar")) > 0]
    losers  = [t for t in closed if _safe_float(t.get("pnl_dollar")) <= 0]
    wr_day  = len(winners) / len(closed) if closed else 0
    avg_win = sum(_safe_float(t.get("pnl_dollar")) for t in winners) / len(winners) if winners else 0
    avg_los = sum(_safe_float(t.get("pnl_dollar")) for t in losers)  / len(losers)  if losers  else 0
    best    = max(closed, key=lambda t: _safe_float(t.get("pnl_dollar")), default=None)
    worst   = min(closed, key=lambda t: _safe_float(t.get("pnl_dollar")), default=None)

    best_str  = f"{best['ticker']} {_fmt_dollar(_safe_float(best.get('pnl_dollar')))}" if best else "N/A"
    worst_str = f"{worst['ticker']} {_fmt_dollar(_safe_float(worst.get('pnl_dollar')))}" if worst else "N/A"

    at_wr  = all_time.get("win_rate", 0)
    at_win = all_time.get("avg_winner", 0)
    at_los = all_time.get("avg_loser", 0)
    at_tot = all_time.get("total", 0)

    return f"""<div class="stats-grid">
  <div class="stat-card">
    <div class="stat-label">Today Win Rate</div>
    <div class="stat-value blue">{wr_day:.0%}</div>
    <div class="muted">{len(winners)}W / {len(losers)}L</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Today Avg Winner</div>
    <div class="stat-value green">{_fmt_dollar(avg_win)}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Today Avg Loser</div>
    <div class="stat-value red">{_fmt_dollar(avg_los)}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Best Trade Today</div>
    <div class="stat-value green" style="font-size:.9rem">{best_str}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">Worst Trade Today</div>
    <div class="stat-value red" style="font-size:.9rem">{worst_str}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">All-Time Win Rate</div>
    <div class="stat-value blue">{at_wr:.0%}</div>
    <div class="muted">{at_tot} total trades</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">All-Time Avg Winner</div>
    <div class="stat-value green">{_fmt_dollar(at_win)}</div>
  </div>
  <div class="stat-card">
    <div class="stat-label">All-Time Avg Loser</div>
    <div class="stat-value red">{_fmt_dollar(at_los)}</div>
  </div>
</div>"""


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def generate_report(date_str: str = None, output_dir: str = None) -> str:
    """Generate HTML report. Returns path to generated file."""
    if date_str is None:
        date_str = datetime.utcnow().strftime("%Y-%m-%d")

    out_dir = Path(output_dir) if output_dir else REPORTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{date_str}.html"

    trades_today  = _get_trades_for_date(date_str)
    summary       = _get_daily_summary(date_str)
    open_positions = _get_open_positions()
    all_time_stats = _get_all_time_stats()

    # Enrich with live prices
    all_tickers = list({t.get("ticker") for t in (trades_today + open_positions) if t.get("ticker")})
    live_prices = _fetch_live_prices(all_tickers)

    closed   = [t for t in trades_today if t.get("pnl_dollar") is not None]
    gross_pnl = sum(_safe_float(t.get("pnl_dollar")) for t in closed)

    if summary:
        portfolio_value  = _safe_float(summary.get("ending_value"))
        starting_value   = _safe_float(summary.get("starting_value"))
        cash             = _safe_float(summary.get("cash"))
        vix              = _safe_float(summary.get("vix_level"))
        spy_regime       = summary.get("macro_bias", "unknown")
    else:
        portfolio_value  = 0.0
        starting_value   = 0.0
        cash             = 0.0
        vix              = 0.0
        spy_regime       = "unknown"

    daily_pnl_pct = (gross_pnl / starting_value * 100) if starting_value > 0 else 0.0
    pnl_cls       = _pnl_class(gross_pnl)

    regime_colors = {"bull": "green", "caution": "yellow", "bear": "red"}
    regime_color  = regime_colors.get(spy_regime, "muted")

    exec_summary_html = _build_executive_summary(date_str, trades_today, summary, live_prices)
    trades_html       = _build_trades_section(trades_today, live_prices)
    portfolio_html    = _build_portfolio_section(open_positions, live_prices)
    stats_html        = _build_stats_section(trades_today, all_time_stats)

    generated_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Trading Bot Report — {date_str}</title>
  <style>{_CSS}</style>
</head>
<body>
<div class="container">

  <!-- HEADER -->
  <div style="margin-bottom:8px">
    <h1>Trading Bot Daily Report</h1>
    <div class="muted">{date_str}</div>
  </div>

  <div class="header-grid">
    <div class="metric-card">
      <div class="label">Portfolio Value</div>
      <div class="value blue">${portfolio_value:,.2f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Starting Value</div>
      <div class="value muted">${starting_value:,.2f}</div>
    </div>
    <div class="metric-card">
      <div class="label">Daily P&amp;L</div>
      <div class="value {pnl_cls}">{_fmt_dollar(gross_pnl)}</div>
    </div>
    <div class="metric-card">
      <div class="label">Daily P&amp;L %</div>
      <div class="value {pnl_cls}">{daily_pnl_pct:+.2f}%</div>
    </div>
    <div class="metric-card">
      <div class="label">Cash</div>
      <div class="value muted">${cash:,.2f}</div>
    </div>
    <div class="metric-card">
      <div class="label">VIX</div>
      <div class="value {'red' if vix >= 25 else 'green'}">{vix:.1f}</div>
    </div>
    <div class="metric-card">
      <div class="label">SPY Regime</div>
      <div class="value {regime_color}">{spy_regime.upper()}</div>
    </div>
    <div class="metric-card">
      <div class="label">Trades Today</div>
      <div class="value">{len(trades_today)}</div>
    </div>
  </div>

  <!-- EXECUTIVE SUMMARY -->
  <h2>Executive Summary</h2>
  <div class="executive-summary">{exec_summary_html}</div>

  <!-- TRADES TODAY -->
  <h2>Trades Today ({len(trades_today)})</h2>
  {trades_html}

  <!-- PORTFOLIO SNAPSHOT -->
  <h2>Portfolio Snapshot — Open Positions ({len(open_positions)})</h2>
  {portfolio_html}

  <!-- STATISTICS -->
  <h2>Statistics</h2>
  {stats_html}

  <!-- FOOTER -->
  <footer>
    <p>Generated at {generated_at} | This report is for informational purposes only.</p>
    <p style="margin-top:4px">⚠ Not financial advice. Past performance does not guarantee future results.</p>
  </footer>

</div>
</body>
</html>"""

    out_path.write_text(html, encoding="utf-8")
    logger.info(f"[report] HTML report written to {out_path}")
    return str(out_path)
