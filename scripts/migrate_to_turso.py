#!/usr/bin/env python3
"""One-off migration: copy the local SQLite trades.db into Turso (libSQL).

Run once, after TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are set. The schema is
created by the app's own init_db() so it is identical to normal operation, then
every row from each table is copied across.

Safety: refuses to run if the Turso `trades` table already holds rows, so it can
never clobber data the bot has already written.
"""
import os
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

LOCAL_DB = REPO_ROOT / "data" / "trades.db"
TABLES = ["trades", "daily_summary", "scan_log", "rejections"]


def main() -> None:
    if not os.getenv("TURSO_DATABASE_URL"):
        sys.exit("TURSO_DATABASE_URL not set — aborting")
    if not LOCAL_DB.exists():
        sys.exit(f"local db not found at {LOCAL_DB}")

    # Create the schema on Turso using the app's own init_db (env vars route
    # _connect() to Turso), guaranteeing an identical table layout.
    from bot.logger import init_db, _connect
    init_db()

    dst = _connect()
    existing = dst.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
    if existing > 0:
        sys.exit(
            f"Turso already has {existing} trade rows — aborting so live data "
            f"is not overwritten. Drop the rows first if a re-migration is intended."
        )

    src = sqlite3.connect(str(LOCAL_DB))
    src.row_factory = sqlite3.Row

    grand_total = 0
    for table in TABLES:
        try:
            rows = src.execute(f"SELECT * FROM {table}").fetchall()
        except sqlite3.OperationalError:
            print(f"  {table}: not present locally — skipped")
            continue
        for r in rows:
            cols = r.keys()
            collist = ",".join(cols)
            placeholders = ",".join(["?"] * len(cols))
            dst.execute(
                f"INSERT INTO {table} ({collist}) VALUES ({placeholders})",
                tuple(r[c] for c in cols),
            )
        dst.commit()
        grand_total += len(rows)
        print(f"  {table}: {len(rows)} rows")

    src.close()
    dst.close()
    print(f"Migration complete — {grand_total} rows copied to Turso.")


if __name__ == "__main__":
    main()
