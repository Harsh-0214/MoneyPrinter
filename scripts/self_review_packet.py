#!/usr/bin/env python3
"""Build the daily self-review packet.

Collects everything an automated reviewer needs to judge how the bot is
actually performing — open holdings, closed-trade statistics sliced several
ways, daily summaries, rejection reasons, scan throughput, AI-filter accuracy
and recent CI failures — and writes it to a single markdown file.

The output is designed to be pasted straight into an LLM context, so it is
size-bounded (see MAX_BYTES) and contains numbers rather than raw dumps.

Usage:
    python scripts/self_review_packet.py                # last 30 days
    python scripts/self_review_packet.py --days 90
    python scripts/self_review_packet.py --no-live      # skip Alpaca calls
    python scripts/self_review_packet.py --out /tmp/p.md
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Hard ceiling on the packet so a runaway history can't blow the review budget.
MAX_BYTES = 220_000

DEFAULT_OUT = Path("reports/self_review/latest.md")


# ── helpers ───────────────────────────────────────────────────────────────────

def _f(v, default: float = 0.0) -> float:
    try:
        if v is None:
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def _i(v, default: int = 0) -> int:
    try:
        if v is None:
            return default
        return int(v)
    except (TypeError, ValueError):
        return default


def _day(ts) -> str:
    return (ts or "")[:10]


def _parse_ts(ts) -> datetime | None:
    """Parse a stored timestamp to an aware UTC datetime.

    Rows migrated from the pre-Turso SQLite file can carry naive timestamps,
    so anything without a tzinfo is assumed to be UTC.
    """
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _pct(n: int, d: int) -> str:
    return f"{(n / d * 100):.1f}%" if d else "—"


def _table(headers: list[str], rows: list[list], limit: int | None = None) -> str:
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        return "_(none)_\n"
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join("" if c is None else str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


def _stats(trades: list[dict]) -> dict:
    """Win/loss aggregates for a list of closed trades."""
    pnls = [_f(t.get("pnl_dollar")) for t in trades]
    pcts = [_f(t.get("pnl_pct")) for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = abs(sum(losses))
    return {
        "n": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": (len(wins) / len(trades) * 100) if trades else 0.0,
        "pnl": sum(pnls),
        "avg_pct": statistics.mean(pcts) if pcts else 0.0,
        "med_pct": statistics.median(pcts) if pcts else 0.0,
        "avg_win": statistics.mean(wins) if wins else 0.0,
        "avg_loss": statistics.mean(losses) if losses else 0.0,
        "expectancy": (sum(pnls) / len(trades)) if trades else 0.0,
        "profit_factor": (gross_win / gross_loss) if gross_loss else float("inf") if gross_win else 0.0,
    }


def _stat_row(label, s: dict) -> list:
    pf = s["profit_factor"]
    pf_s = "∞" if pf == float("inf") else f"{pf:.2f}"
    return [label, s["n"], f'{s["win_rate"]:.0f}%', f'${s["pnl"]:,.0f}',
            f'{s["avg_pct"]:+.2f}%', f'{s["med_pct"]:+.2f}%',
            f'${s["expectancy"]:+,.0f}', pf_s]


_STAT_HEADERS = ["group", "n", "win%", "net P&L", "avg %", "med %", "expectancy", "PF"]


def _grouped_stats(trades: list[dict], key_fn, min_n: int = 1) -> str:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in trades:
        groups[str(key_fn(t) or "unknown")].append(t)
    rows = [_stat_row(k, _stats(v)) for k, v in groups.items() if len(v) >= min_n]
    rows.sort(key=lambda r: -r[1])
    return _table(_STAT_HEADERS, rows)


def _bucket(v: float, edges: list[float]) -> str:
    for i, e in enumerate(edges):
        if v < e:
            return f"<{e:g}" if i == 0 else f"{edges[i-1]:g}–{e:g}"
    return f">={edges[-1]:g}"


def _hold_days(t: dict) -> float | None:
    a, b = _parse_ts(t.get("timestamp")), _parse_ts(t.get("exit_timestamp"))
    return (b - a).total_seconds() / 86400 if a and b else None


# ── collectors ────────────────────────────────────────────────────────────────

def collect_db(days: int) -> dict:
    from bot.logger import _connect, init_db

    init_db()
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    conn = _connect()
    try:
        def q(sql, params=()):
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

        return {
            "open": q("SELECT * FROM trades WHERE status='open' ORDER BY timestamp"),
            # Fall back to the entry timestamp so a closed trade with a missing
            # exit_timestamp (possible on repaired rows) still gets reviewed.
            "closed": q("SELECT * FROM trades WHERE status!='open' AND "
                        "COALESCE(exit_timestamp, timestamp) >= ? "
                        "ORDER BY COALESCE(exit_timestamp, timestamp) DESC", (cutoff,)),
            "all_closed_n": q("SELECT COUNT(*) AS n FROM trades WHERE status!='open'")[0]["n"],
            "summaries": q("SELECT * FROM daily_summary WHERE date >= ? ORDER BY date DESC", (cutoff,)),
            "rejections": q("SELECT rejection_reason, COUNT(*) AS n, AVG(net_score) AS avg_net, "
                            "AVG(confidence) AS avg_conf FROM rejections WHERE timestamp >= ? "
                            "GROUP BY rejection_reason ORDER BY n DESC", (cutoff,)),
            "rejected_tickers": q("SELECT ticker, COUNT(*) AS n FROM rejections WHERE timestamp >= ? "
                                  "GROUP BY ticker ORDER BY n DESC LIMIT 25", (cutoff,)),
            "scans": q("SELECT * FROM scan_log WHERE timestamp >= ? ORDER BY timestamp DESC", (cutoff,)),
        }
    finally:
        conn.close()


def collect_live_positions() -> dict:
    """Live Alpaca account + positions. Returns {} when creds/API unavailable."""
    try:
        from bot.trader import build_client, get_positions
        client = build_client()
        out: dict = {"positions": {p["symbol"]: p for p in get_positions(client)}}
        try:
            acct = client.get_account()
            out["account"] = {
                "equity": _f(getattr(acct, "equity", None)),
                "cash": _f(getattr(acct, "cash", None)),
                "buying_power": _f(getattr(acct, "buying_power", None)),
                "last_equity": _f(getattr(acct, "last_equity", None)),
            }
        except Exception:
            pass
        return out
    except Exception as e:  # offline / no creds / API down
        return {"error": f"{type(e).__name__}: {e}"}


def collect_ci(limit: int = 30) -> list[dict]:
    """Recent GitHub Actions runs via the gh CLI (no-op when gh is absent)."""
    try:
        raw = subprocess.run(
            ["gh", "run", "list", "--limit", str(limit),
             "--json", "name,status,conclusion,createdAt,displayTitle,url"],
            capture_output=True, text=True, timeout=60,
        )
        if raw.returncode != 0:
            return []
        return json.loads(raw.stdout or "[]")
    except Exception:
        return []


def collect_code_changes(days: int, limit: int = 40) -> list[str]:
    """Non-bot commits in the window, so the reviewer can attribute performance
    shifts to code changes (including its own previous proposals)."""
    try:
        raw = subprocess.run(
            ["git", "log", f"--since={days} days ago", "--date=short",
             "--pretty=format:%ad|%h|%s", "--no-merges"],
            capture_output=True, text=True, timeout=30,
        )
        if raw.returncode != 0:
            return []
        return [ln for ln in raw.stdout.splitlines()
                if "[skip ci]" not in ln and not ln.split("|")[-1].startswith("bot:")][:limit]
    except Exception:
        return []


def collect_local_logs(max_lines: int = 60) -> list[str]:
    """Tail of ERROR/WARNING lines from any committed log files."""
    lines: list[str] = []
    for p in sorted(Path("logs").glob("*.log")) if Path("logs").is_dir() else []:
        try:
            for ln in p.read_text(errors="replace").splitlines():
                if "ERROR" in ln or "CRITICAL" in ln or "Traceback" in ln:
                    lines.append(f"{p.name}: {ln.strip()}")
        except Exception:
            continue
    return lines[-max_lines:]


# ── rendering ─────────────────────────────────────────────────────────────────

def render(db: dict, live: dict, ci: list[dict], log_lines: list[str],
           commits: list[str], days: int) -> str:
    now = datetime.now(timezone.utc)
    open_tr, closed = db["open"], db["closed"]
    L: list[str] = []
    add = L.append

    add(f"# Bot self-review packet — {now:%Y-%m-%d %H:%M UTC}")
    add(f"\nWindow: last **{days}** days · open positions: **{len(open_tr)}** · "
        f"closed trades in window: **{len(closed)}** (lifetime {db['all_closed_n']})\n")

    # ── account ──
    acct = live.get("account")
    if acct:
        chg = acct["equity"] - acct["last_equity"]
        add("## Account\n")
        add(f"- Equity: **${acct['equity']:,.2f}** ({chg:+,.2f} vs prior close)")
        add(f"- Cash: ${acct['cash']:,.2f} · Buying power: ${acct['buying_power']:,.2f}\n")
    elif live.get("error"):
        add(f"## Account\n\n_Live broker data unavailable: {live['error']}_\n")

    # ── holdings ──
    add("## Open holdings\n")
    lp = live.get("positions", {})
    rows = []
    for t in open_tr:
        tk = t["ticker"]
        entry = _f(t.get("entry_price"))
        cur = _f((lp.get(tk) or {}).get("current_price"), entry)
        pnl = ((cur - entry) / entry * 100) if entry else 0.0
        if (t.get("action") or "") in ("short", "sell"):
            pnl = -pnl
        stop, tgt = _f(t.get("stop_loss")), _f(t.get("take_profit"))
        opened = _parse_ts(t.get("timestamp"))
        held = (now - opened).days if opened else None
        rows.append([
            tk, t.get("action"), t.get("strategy"), t.get("time_horizon"),
            _i(t.get("quantity")), f"${entry:.2f}", f"${cur:.2f}", f"{pnl:+.2f}%",
            f"${stop:.2f}", f"${tgt:.2f}",
            f"{((cur - stop) / cur * 100):+.1f}%" if cur and stop else "—",
            f"{((tgt - cur) / cur * 100):+.1f}%" if cur and tgt else "—",
            held, f"{_f(t.get('confidence')):.2f}", _i(t.get("net_score")),
            "yes" if _i(t.get("ai_confirmed")) else "no",
        ])
    add(_table(["ticker", "side", "strategy", "horizon", "qty", "entry", "last", "P&L%",
                "stop", "target", "→stop", "→target", "days", "conf", "net", "ai_ok"], rows))

    if open_tr:
        add("\n### Holding concentration\n")
        by_strat = Counter(t.get("strategy") or "?" for t in open_tr)
        by_side = Counter(t.get("action") or "?" for t in open_tr)
        exposure = sum(_f(t.get("entry_price")) * _i(t.get("quantity")) for t in open_tr)
        add(f"- Cost-basis exposure: **${exposure:,.0f}**"
            + (f" ({exposure / acct['equity'] * 100:.0f}% of equity)" if acct and acct["equity"] else ""))
        add(f"- By strategy: {dict(by_strat)}")
        add(f"- By side: {dict(by_side)}\n")

    # ── closed-trade performance ──
    add(f"## Closed trades — last {days} days\n")
    if not closed:
        add("_No closed trades in window._\n")
    else:
        add("### Overall\n")
        add(_table(_STAT_HEADERS, [_stat_row("all", _stats(closed))]))

        add("\n### By strategy\n")
        add(_grouped_stats(closed, lambda t: t.get("strategy")))

        add("\n### By exit status\n")
        add(_grouped_stats(closed, lambda t: t.get("status")))

        add("\n### By time horizon\n")
        add(_grouped_stats(closed, lambda t: t.get("time_horizon")))

        add("\n### By side\n")
        add(_grouped_stats(closed, lambda t: t.get("action")))

        add("\n### By macro bias at entry\n")
        add(_grouped_stats(closed, lambda t: t.get("macro_bias")))

        add("\n### By VIX bucket at entry\n")
        add(_grouped_stats(closed, lambda t: _bucket(_f(t.get("vix_level")), [15, 20, 25, 30])))

        add("\n### By entry confidence\n")
        add(_grouped_stats(closed, lambda t: _bucket(_f(t.get("confidence")), [0.6, 0.7, 0.8, 0.9])))

        add("\n### By entry net score\n")
        add(_grouped_stats(closed, lambda t: _bucket(abs(_f(t.get("net_score"))), [40, 60, 75, 90])))

        add("\n### By hold duration\n")
        add(_grouped_stats(closed, lambda t: _bucket(_hold_days(t) or 0, [1, 3, 7, 21])))

        add("\n### AI filter vs outcome\n")
        add(_grouped_stats(closed, lambda t: "ai_confirmed" if _i(t.get("ai_confirmed")) else "not_confirmed"))

        add("\n### Repeat tickers (n >= 2)\n")
        add(_grouped_stats(closed, lambda t: t.get("ticker"), min_n=2))

        # worst / best with reasoning so the reviewer can see *why* it entered
        def detail_rows(sample):
            return [[
                t.get("ticker"), t.get("action"), t.get("strategy"), t.get("status"),
                f'{_f(t.get("pnl_pct")):+.2f}%', f'${_f(t.get("pnl_dollar")):+,.0f}',
                _day(t.get("timestamp")), _day(t.get("exit_timestamp")),
                f'{_f(t.get("confidence")):.2f}', _i(t.get("net_score")),
                (t.get("signals_triggered") or "")[:110].replace("|", "/").replace("\n", " "),
                (t.get("reasoning") or t.get("ai_reasoning") or "")[:220].replace("|", "/").replace("\n", " "),
            ] for t in sample]

        hdr = ["ticker", "side", "strategy", "exit", "P&L%", "P&L$", "in", "out",
               "conf", "net", "signals", "reasoning"]
        by_pnl = sorted(closed, key=lambda t: _f(t.get("pnl_dollar")))
        add("\n### 15 worst losers\n")
        add(_table(hdr, detail_rows(by_pnl[:15])))
        add("\n### 10 best winners\n")
        add(_table(hdr, detail_rows(list(reversed(by_pnl))[:10])))

    # ── daily summaries ──
    add("\n## Daily summaries\n")
    add(_table(
        ["date", "start", "end", "day P&L", "trades", "W/L", "win%", "macro", "VIX", "kill", "notes"],
        [[s.get("date"), f'${_f(s.get("starting_value")):,.0f}', f'${_f(s.get("ending_value")):,.0f}',
          f'${_f(s.get("ending_value")) - _f(s.get("starting_value")):+,.0f}',
          _i(s.get("total_trades")), f'{_i(s.get("winning_trades"))}/{_i(s.get("losing_trades"))}',
          f'{_f(s.get("win_rate")):.0f}%', s.get("macro_bias"), f'{_f(s.get("vix_level")):.1f}',
          "YES" if _i(s.get("kill_switch_triggered")) else "",
          (s.get("notes") or "")[:120].replace("|", "/").replace("\n", " ")]
         for s in db["summaries"]], limit=45))

    # ── funnel ──
    add("\n## Scan funnel\n")
    scans = db["scans"]
    by_session: dict[str, list[dict]] = defaultdict(list)
    for s in scans:
        by_session[s.get("session") or "?"].append(s)
    add(_table(["session", "runs", "tickers scanned", "signals", "trades", "signal rate", "fill rate"],
               [[k, len(v), sum(_i(x.get("tickers_scanned")) for x in v),
                 sum(_i(x.get("signals_generated")) for x in v),
                 sum(_i(x.get("trades_executed")) for x in v),
                 _pct(sum(_i(x.get("signals_generated")) for x in v),
                      sum(_i(x.get("tickers_scanned")) for x in v)),
                 _pct(sum(_i(x.get("trades_executed")) for x in v),
                      sum(_i(x.get("signals_generated")) for x in v))]
                for k, v in sorted(by_session.items())]))

    add("\n### Rejection reasons\n")
    add(_table(["reason", "count", "avg net", "avg conf"],
               [[r.get("rejection_reason"), _i(r.get("n")), f'{_f(r.get("avg_net")):.0f}',
                 f'{_f(r.get("avg_conf")):.2f}'] for r in db["rejections"]], limit=40))

    add("\n### Most-rejected tickers\n")
    add(_table(["ticker", "rejections"],
               [[r.get("ticker"), _i(r.get("n"))] for r in db["rejected_tickers"]]))

    # ── CI health ──
    add("\n## Recent workflow runs\n")
    if ci:
        bad = [r for r in ci if r.get("conclusion") not in ("success", None, "")]
        add(f"Failures/cancellations in last {len(ci)} runs: **{len(bad)}**\n")
        add(_table(["workflow", "conclusion", "when", "title", "url"],
                   [[r.get("name"), r.get("conclusion"), (r.get("createdAt") or "")[:16],
                     (r.get("displayTitle") or "")[:60], r.get("url")] for r in bad], limit=20))
    else:
        add("_gh CLI unavailable — no CI data._\n")

    # ── code changes in the window ──
    add("\n## Code changes in this window\n")
    add("_Performance shifts should be read against these — including changes "
        "landed by previous self-improve runs._\n")
    add(_table(["date", "commit", "subject"],
               [ln.split("|", 2) for ln in commits if ln.count("|") >= 2]))

    if log_lines:
        add("\n## Recent error lines\n\n```\n" + "\n".join(log_lines) + "\n```\n")

    text = "\n".join(L)
    if len(text.encode()) > MAX_BYTES:
        text = text.encode()[:MAX_BYTES].decode(errors="ignore") + "\n\n_[truncated]_\n"
    return text


# ── entry point ───────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Build the daily self-review packet")
    ap.add_argument("--days", type=int, default=30, help="lookback window (default 30)")
    ap.add_argument("--no-live", action="store_true", help="skip Alpaca live enrichment")
    ap.add_argument("--no-ci", action="store_true", help="skip gh CLI workflow history")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="output markdown path")
    args = ap.parse_args()

    db = collect_db(args.days)
    live = {} if args.no_live else collect_live_positions()
    ci = [] if args.no_ci else collect_ci()
    text = render(db, live, ci, collect_local_logs(),
                  collect_code_changes(args.days), args.days)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text)
    dated = out.parent / f"{datetime.now(timezone.utc):%Y-%m-%d}.md"
    dated.write_text(text)

    print(f"Wrote {out} ({len(text.encode()):,} bytes) and {dated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
