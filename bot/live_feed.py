"""Writes bot decisions to data/live_feed.json for the Vercel dashboard."""

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

FEED_PATH   = Path(__file__).parent.parent / "data" / "live_feed.json"
MAX_ENTRIES = 200


def write_live_feed(decisions: list[dict], session: str) -> None:
    """
    Append this run's decisions to data/live_feed.json.

    Each decision dict should contain the scored/AI-filtered signal fields.
    Keeps only the last MAX_ENTRIES entries. Safe to call with an empty list.
    """
    FEED_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Load existing feed
    existing: dict = {}
    if FEED_PATH.exists():
        try:
            with open(FEED_PATH) as f:
                existing = json.load(f)
        except Exception as e:
            logger.warning(f"[live_feed] Could not load existing feed: {e} — starting fresh")

    entries: list = existing.get("entries", [])

    now_iso = datetime.now(timezone.utc).isoformat()

    # Count today's runs for meta
    today_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    today_runs   = existing.get("meta", {}).get("total_runs_today", 0)
    last_meta_date = existing.get("meta", {}).get("last_updated", "")[:10]
    if last_meta_date != today_prefix:
        today_runs = 0  # reset at midnight
    today_runs += 1

    # Build new entry rows
    for d in decisions:
        entries.append({
            "timestamp":   now_iso,
            "session":     session,
            "ticker":      d.get("ticker", ""),
            "action":      d.get("action", "hold"),
            "net_score":   d.get("net_score", 0),
            "confidence":  round(float(d.get("confidence") or 0), 4),
            "strategy":    d.get("strategy", ""),
            "bull_score":  d.get("bull_score", 0),
            "bear_score":  d.get("bear_score", 0),
            "ai_confirmed": d.get("ai_confirmed"),
            "ai_reasoning": d.get("ai_reasoning", ""),
            "entry_price":  d.get("entry_price"),
            "stop_loss":    d.get("stop_loss"),
            "take_profit":  d.get("take_profit"),
            "reasoning":    d.get("reasoning", ""),
        })

    # Trim to last MAX_ENTRIES (oldest first, so trim from front)
    if len(entries) > MAX_ENTRIES:
        entries = entries[-MAX_ENTRIES:]

    feed = {
        "meta": {
            "last_updated":    now_iso,
            "last_session":    session,
            "total_runs_today": today_runs,
        },
        "entries": entries,
    }

    with open(FEED_PATH, "w") as f:
        json.dump(feed, f, indent=2, default=str)

    logger.info(f"[live_feed] Wrote {len(decisions)} decision(s) — {len(entries)} total entries")
