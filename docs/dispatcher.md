# External dispatcher setup

GitHub's `schedule:` cron is best-effort: runs routinely start 5–40 minutes
late and are occasionally dropped entirely. The workflows compensate today by
firing ~2 hours early and burning runner minutes waiting in-job. An external
dispatcher closes the remaining gap (dropped runs) and makes start times
exact: a scheduler you control calls the `workflow_dispatch` API at the
precise ET times, and the GitHub crons demote themselves to fallbacks.

## How the two triggers coexist

Every workflow gate understands the repository variable
`EXTERNAL_DISPATCHER`:

- **Unset / `false` (default):** behavior is exactly what it is today —
  scheduled runs wait in-job until their target time and proceed.
- **`true`:** the dispatched run owns the slot. The scheduled (fallback) run
  waits at its target time watching for a `workflow_dispatch` run to appear;
  if one shows up it bows out, and only if none arrives within 5 minutes of
  target does it proceed itself.

Failure matrix with `EXTERNAL_DISPATCHER=true`:

| GitHub cron | Dispatcher | Result |
|-------------|-----------|--------|
| fires | fires | dispatched run starts on time; scheduled run bows out |
| dropped | fires | dispatched run starts on time |
| fires | down | scheduled run proceeds ~5 min after target |
| dropped | down | no run (same exposure as today with cron alone) |

Concurrency groups (`discovery`, `premarket`, `trading-day`, `eod`) serialize
any residual overlap, and `main.py` exits on market close, so even a stray
duplicate run is wasteful rather than harmful.

## Step 1 — create a fine-grained PAT

GitHub → Settings → Developer settings → Fine-grained personal access tokens:

- **Repository access:** only `Harsh-0214/MoneyPrinter`
- **Permissions:** Repository → **Actions: Read and write** (nothing else)
- Set an expiry you'll remember to rotate (e.g. 1 year)

This token only lives in the external scheduler's config.

## Step 2 — create the scheduler jobs

Recommended: [cron-job.org](https://cron-job.org) (free, minute precision,
timezone-aware so DST is handled). Google Cloud Scheduler or AWS EventBridge
work identically if you prefer.

Create one job per slot, schedule timezone **America/New_York**, Mon–Fri:

| Time (ET) | Workflow | URL (POST) |
|-----------|----------|------------|
| 08:30 | Discovery | `https://api.github.com/repos/Harsh-0214/MoneyPrinter/actions/workflows/discovery.yml/dispatches` |
| 09:00 | Pre-Market | `https://api.github.com/repos/Harsh-0214/MoneyPrinter/actions/workflows/premarket.yml/dispatches` |
| 09:30 | Trading Day (morning) | `https://api.github.com/repos/Harsh-0214/MoneyPrinter/actions/workflows/trading_day.yml/dispatches` |
| 13:07 | Trading Day (afternoon) | `https://api.github.com/repos/Harsh-0214/MoneyPrinter/actions/workflows/trading_day.yml/dispatches` |
| 16:15 | EOD Summary | `https://api.github.com/repos/Harsh-0214/MoneyPrinter/actions/workflows/eod_summary.yml/dispatches` |

Each job's request settings:

- Method: `POST`
- Headers:
  - `Authorization: Bearer <your PAT>`
  - `Accept: application/vnd.github+json`
  - `X-GitHub-Api-Version: 2022-11-28`
- Body: `{"ref":"main"}`

A successful dispatch returns HTTP **204** with an empty body — configure the
job to treat 204 as success and alert (cron-job.org emails you) on anything
else, so an expired PAT doesn't fail silently.

## Step 3 — test

```bash
GITHUB_PAT=github_pat_xxx scripts/dispatch_workflow.sh test_ai_filter.yml
```

Then check the Actions tab: the run should appear within seconds with event
`workflow_dispatch`. Repeat for one of the real workflows outside market
hours if you want (dispatched runs start immediately, and `main.py` exits
cleanly when the market is closed).

## Step 4 — flip the switch

Repo → Settings → Secrets and variables → Actions → **Variables** → New:
`EXTERNAL_DISPATCHER` = `true`.

The next scheduled runs become fallbacks automatically. To revert to
cron-only at any time, set it to `false` or delete it — no code change
needed.

## Notes

- The trading_day morning fallback proceeds at **09:35 ET** when no dispatch
  arrives (vs 09:30 sharp from the dispatcher); the afternoon fallback runs
  immediately unless a dispatched run took the slot in the previous 20 min.
- `backtest.yml` and `test_ai_filter.yml` are manual-only and need no
  dispatcher jobs.
- The dispatcher times are ET (exchange time); the GitHub fallback crons
  remain UTC-fixed, which is fine since they only act when the dispatcher is
  silent.
