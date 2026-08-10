# MoneyPrinter — repo guide

Autonomous paper-trading bot. GitHub Actions runs the sessions on a schedule,
state lives in Turso (libSQL) via `bot/logger.py`, and Alpaca paper is the broker.

## Layout

| Path | What it is |
| --- | --- |
| `main.py` | Session router (`--session discovery/premarket/continuous/eod_summary/backtest/holdings`) plus portfolio-level risk gates |
| `bot/scorer.py` | Bull/bear scoring — the core signal engine |
| `bot/indicators.py` | Technical indicator computation |
| `bot/strategies.py` | Regime router + per-strategy entry logic |
| `bot/ai_filter.py` | Claude second-opinion pass on scored tickers |
| `bot/risk.py` | Position sizing, kill switch |
| `bot/trader.py` | Alpaca order placement |
| `bot/logger.py` | DB schema + all reads/writes (`trades`, `rejections`, `daily_summary`, `scan_log`) |
| `bot/discovery.py` | Universe discovery → `discovered_tickers.json` |
| `scripts/self_review_packet.py` | Builds the daily performance packet |
| `.github/workflows/` | Session schedules + the self-improve loop |

## Checks

Run before committing anything:

```bash
./scripts/selfcheck.sh
```

There is no unit-test suite. `python backtest.py` / `--session backtest` is the
only way to evaluate a signal change, and it needs network access.

## Invariants — do not change these without an explicit human request

1. **Paper trading only.** `ALPACA_BASE_URL` stays `https://paper-api.alpaca.markets`.
   Never add a live-trading endpoint, key, or code path.
2. **Risk limits only tighten, never loosen, in an automated change.**
   `MAX_DAILY_LOSS_PCT`, `MAX_PORTFOLIO_EXPOSURE_PCT`, `MAX_TOTAL_EXPOSURE_PCT`,
   `MAX_NEW_ENTRIES_PER_SESSION`, `MAX_TRADES_PER_SESSION`, `MAX_STOP_WIDTH_PCT`,
   `MAX_POSITIONS_PER_SECTOR` (all in `main.py`) and the kill switch in
   `bot/risk.py` are human-owned. Loosening any of them requires a human.
3. **Never remove or bypass the kill switch, the sector cap, or the stop-loss
   path.**
4. **No secrets in code.** Everything comes from env vars / repo secrets.
5. **Don't rewrite history or push to `main` directly.** Automated work goes on
   a branch and through a pull request.
6. **Schema changes are additive.** `bot/logger.py:init_db()` uses a
   migration-safe `ALTER TABLE` loop — add columns there, never drop or rename.
7. **Don't touch `data/live_feed.json`, `discovered_tickers.json`, or
   `reports/`** — those are bot-written state, committed by the session
   workflows.

## Style

Match the surrounding code: type hints on new functions, `logger` (not `print`)
inside `bot/`, `rich` console output in `main.py`, box-drawing section banners.
