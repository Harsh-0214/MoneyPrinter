#!/usr/bin/env bash
# Fast sanity gate for automated (and manual) changes.
#
#   ./scripts/selfcheck.sh
#
# Byte-compiles every Python file and imports every bot module, so syntax
# errors, bad imports and import-time crashes are caught before anything is
# committed. Network and broker calls are not exercised.
set -uo pipefail

fail=0
step() { printf '\n\033[1m== %s\033[0m\n' "$1"; }
check() { if [ "$1" -ne 0 ]; then echo "FAILED: $2"; fail=1; else echo "ok: $2"; fi; }

step "Byte-compile"
python -m compileall -q bot main.py backtest.py generate_pdf.py scripts >/dev/null
check $? "compileall"

step "Import bot modules"
python - <<'PY'
import importlib, sys
mods = ["bot.data", "bot.indicators", "bot.scorer", "bot.strategies", "bot.risk",
        "bot.trader", "bot.portfolio", "bot.logger", "bot.news", "bot.discovery",
        "bot.ai_filter", "bot.live_feed", "bot.historical_context"]
bad = []
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as e:
        bad.append(f"{m}: {type(e).__name__}: {e}")
if bad:
    print("\n".join(bad)); sys.exit(1)
print(f"imported {len(mods)} modules")
PY
check $? "imports"

step "CLI wiring"
python main.py --help >/dev/null 2>&1
check $? "main.py --help"

step "Self-review packet builds offline"
python scripts/self_review_packet.py --days 7 --no-live --no-ci \
  --out /tmp/selfcheck_packet.md >/dev/null 2>&1
check $? "self_review_packet.py"

step "Workflow YAML parses"
python - <<'PY'
import glob, sys
try:
    import yaml
except ImportError:
    print("pyyaml not installed - skipped"); sys.exit(0)
bad = []
for f in glob.glob(".github/workflows/*.yml"):
    try:
        yaml.safe_load(open(f))
    except Exception as e:
        bad.append(f"{f}: {e}")
if bad:
    print("\n".join(bad)); sys.exit(1)
print("all workflow files parse")
PY
check $? "workflow yaml"

echo
if [ "$fail" -ne 0 ]; then echo "SELFCHECK FAILED"; exit 1; fi
echo "SELFCHECK PASSED"
