#!/usr/bin/env python3
"""
One-off repair for the 2026-06-11 reconciliation corruption.

What happened: an Alpaca get_positions() failure was swallowed (returned []),
so reconcile_positions treated the outage as an empty portfolio and marked
every open adopted position (trades 20-26) closed_external with ~+$10.5k of
phantom realized P&L. The code fix lives in bot/trader.py / bot/portfolio.py;
this script repairs the data:

  1. Re-opens trades 20-26 if they are still marked closed_external from that
     event (clears exit_price / exit_timestamp / pnl_dollar / pnl_pct).
  2. Caps the runaway adopted brackets on MRVL (#24) and ARM (#26) — their
     ATR(14) came out at ~12% of price (3x-6x every other position's), giving
     stops 35-37% below entry. Re-derives stop at 10% below entry
     (ADOPT_MAX_STOP_PCT) and target at 2x risk, matching the new clamp in
     bot/portfolio.reconcile_positions.

Idempotent: safe to run repeatedly; already-repaired rows are left alone.

Usage:  python3 scripts/repair_trades.py [path/to/trades.db]
"""

import sqlite3
import sys
from pathlib import Path

PHANTOM_CLOSE_IDS = (20, 21, 22, 23, 24, 25, 26)
ADOPT_MAX_STOP_PCT = 0.10
ADOPT_RISK_REWARD = 2.0
BRACKET_REPAIR_IDS = (24, 26)  # MRVL, ARM


def repair(db_path: Path) -> None:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    # ── 1. Re-open phantom closes ─────────────────────────────────────────
    placeholders = ",".join("?" * len(PHANTOM_CLOSE_IDS))
    rows = conn.execute(
        f"SELECT id, ticker, status, pnl_dollar FROM trades "
        f"WHERE id IN ({placeholders}) AND session = 'reconcile' "
        f"AND status IN ('closed', 'closed_external')",
        PHANTOM_CLOSE_IDS,
    ).fetchall()

    if rows:
        phantom_pnl = sum(r["pnl_dollar"] or 0 for r in rows)
        for r in rows:
            print(f"  re-opening trade #{r['id']} {r['ticker']} "
                  f"(was {r['status']}, phantom pnl ${r['pnl_dollar'] or 0:+,.2f})")
        conn.execute(
            f"UPDATE trades SET status = 'open', exit_price = NULL, "
            f"exit_timestamp = NULL, pnl_dollar = NULL, pnl_pct = NULL "
            f"WHERE id IN ({placeholders}) AND session = 'reconcile' "
            f"AND status IN ('closed', 'closed_external')",
            PHANTOM_CLOSE_IDS,
        )
        print(f"  removed ${phantom_pnl:+,.2f} of phantom realized P&L")
    else:
        print("  trades 20-26: no phantom closes found (already open) — nothing to do")

    # ── 2. Cap runaway adopted brackets ───────────────────────────────────
    for trade_id in BRACKET_REPAIR_IDS:
        row = conn.execute(
            "SELECT id, ticker, entry_price, stop_loss, take_profit FROM trades "
            "WHERE id = ? AND session = 'reconcile' AND status = 'open'",
            (trade_id,),
        ).fetchone()
        if not row:
            continue
        entry = float(row["entry_price"])
        max_stop = round(entry * (1 - ADOPT_MAX_STOP_PCT), 2)
        if float(row["stop_loss"]) >= max_stop:
            print(f"  trade #{trade_id} {row['ticker']}: bracket already sane — skipping")
            continue
        new_stop = max_stop
        new_target = round(entry + ADOPT_RISK_REWARD * (entry - new_stop), 2)
        conn.execute(
            "UPDATE trades SET stop_loss = ?, take_profit = ? WHERE id = ?",
            (new_stop, new_target, trade_id),
        )
        print(f"  trade #{trade_id} {row['ticker']}: bracket "
              f"{row['stop_loss']:.2f}/{row['take_profit']:.2f} → "
              f"{new_stop:.2f}/{new_target:.2f} "
              f"(stop −{ADOPT_MAX_STOP_PCT * 100:.0f}%, target +{ADOPT_MAX_STOP_PCT * ADOPT_RISK_REWARD * 100:.0f}%)")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/trades.db")
    if not path.exists():
        sys.exit(f"DB not found: {path}")
    print(f"Repairing {path} ...")
    repair(path)
    print("Done.")
