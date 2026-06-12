#!/usr/bin/env bash
# Trigger a MoneyPrinter workflow via the GitHub API — the same call the
# external dispatcher (cron-job.org / Cloud Scheduler) makes on schedule.
#
# Usage:
#   GITHUB_PAT=github_pat_xxx scripts/dispatch_workflow.sh trading_day.yml
#   GITHUB_PAT=github_pat_xxx scripts/dispatch_workflow.sh discovery.yml main
set -euo pipefail

WF="${1:?usage: dispatch_workflow.sh <workflow-file.yml> [ref]}"
REF="${2:-main}"
: "${GITHUB_PAT:?set GITHUB_PAT to a fine-grained PAT with Actions read/write on Harsh-0214/MoneyPrinter}"

curl -fsS -X POST \
  -H "Authorization: Bearer ${GITHUB_PAT}" \
  -H "Accept: application/vnd.github+json" \
  -H "X-GitHub-Api-Version: 2022-11-28" \
  "https://api.github.com/repos/Harsh-0214/MoneyPrinter/actions/workflows/${WF}/dispatches" \
  -d "{\"ref\":\"${REF}\"}"

echo "Dispatched ${WF} on ${REF}."
