"""Claude Sonnet second-opinion analyst for every scored ticker.

claude_analyze_ticker(ticker, indicators, scorer_result) → dict with
  decision: "buy" | "short" | "hold"
  confidence: 0.0-1.0
  reasoning: str

apply_ai_opinion(scorer_result, indicators) is the public entry point.
It runs on ALL tickers (not just buys), so Claude can:
  - Confirm a buy/short the scorer found
  - Downgrade a buy/short to hold
  - Upgrade a borderline hold to buy/short

Falls back silently if the API key is missing or the call fails —
the scorer result is returned unchanged so the bot operates normally.

run_ai_filter_batch(scored_list) parallelises calls across a list of
(scorer_result, indicators) pairs using ThreadPoolExecutor.
"""

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

logger = logging.getLogger(__name__)

_AUTO_EXECUTE_NET   = 85
_AUTO_EXECUTE_CONF  = 0.85
_CLAUDE_MIN_NET     = 60
_CLAUDE_MIN_CONF    = 0.65

_SYSTEM_PROMPT = """\
You are a senior portfolio manager and quantitative analyst providing a \
second opinion before any trade is executed.

Your job is NOT to rubber-stamp technical signals. A stock being at an \
all-time low, deeply oversold, or in a dip is NOT by itself a reason to buy. \
A stock being at an all-time high or overbought is NOT by itself a reason to \
short. Technical indicators tell you WHAT has happened — you must reason about \
WHY it happened and WHETHER that justifies a trade.

For every BUY candidate you must ask:
  1. Is there a plausible reason this stock should recover or continue higher?
     (sector tailwinds, earnings beat, product cycle, institutional accumulation)
  2. Or is the dip/low caused by structural problems, deteriorating fundamentals,
     or a sector in secular decline? If so, reject the buy.
  3. Is the entry timed well — is momentum actually turning, or is this a
     falling knife?

For every SHORT candidate you must ask:
  1. Is there a genuine reason this stock should fall from here?
     (valuation extended beyond fundamentals, negative catalyst, sector headwinds)
  2. Or is it strong for a real reason (earnings growth, market leadership)?
     If so, reject the short.
  3. Is the setup confirmed — is there actual distribution or just high RSI?

For HOLDS, ask whether the technical setup is truly ambiguous or whether
one direction is clearly better.

Be conservative. A two-sentence reasoning that cannot explain the business \
logic behind the trade should result in a 'hold'. The reasoning field in your \
response must justify the decision in terms of both technicals AND business \
logic, not just indicator values.

You have access to tactical real-time context showing what JUST happened on the chart.
Prioritize fresh_triggers_fired — if NONE triggered this cycle, the setup is STALE and you
should be very reluctant to enter. A setup unchanged for 3+ cycles should be passed.
Focus on momentum and recency — a fresh signal is worth 3x a stale one.\
"""

# Net score threshold — skip Claude on clean holds (|net| < this AND no open position)
_SKIP_CLAUDE_NET_THRESHOLD = 40

# Max parallel Claude calls per batch (avoids rate-limit bursts)
_MAX_WORKERS = 6

# Set True when Anthropic billing/credits are exhausted — disables Claude for the session
_credits_exhausted: bool = False


def _apply_defaults(scorer_result: dict) -> dict:
    """Copy scorer fields into the expected AI output format (no Claude call)."""
    scorer_result.setdefault("ai_confirmed", True)
    scorer_result.setdefault("ai_reasoning", "scorer-auto: thresholds met, no Claude call")
    scorer_result.setdefault("ai_entry_price", scorer_result.get("entry_price"))
    scorer_result.setdefault("ai_stop_loss", scorer_result.get("stop_loss"))
    scorer_result.setdefault("ai_take_profit", scorer_result.get("take_profit"))
    scorer_result.setdefault("ai_risk_reward", scorer_result.get("risk_reward"))
    scorer_result.setdefault("ai_entry_condition", "scorer-auto")
    return scorer_result


def _fallback(scorer_action: str) -> dict:
    """Pass-through fallback that preserves the scorer's original decision."""
    return {
        "decision":        scorer_action,
        "confidence":      1.0,
        "reasoning":       "AI filter unavailable — scorer decision kept",
        "entry_price":     None,
        "stop_loss":       None,
        "take_profit":     None,
        "risk_reward":     None,
        "entry_condition": "",
    }


def _format_headlines(news: Optional[dict]) -> list[str]:
    """Return formatted headline lines for the prompt."""
    if not news:
        return ["  (no news available)"]
    top = news.get("top_headlines", [])
    if not top:
        return ["  (no headlines retrieved)"]
    lines = []
    for i, h in enumerate(top[:5], 1):
        text = h.get("text", "").strip()
        pol  = h.get("polarity", 0.0)
        sentiment_tag = "positive" if pol > 0.1 else "negative" if pol < -0.1 else "neutral"
        lines.append(f"  {i}. [{sentiment_tag:8s} {pol:+.2f}] {text[:220]}")
    return lines


def _build_tactical_context(ind: dict, score: dict) -> list[str]:
    """Build tactical context block from entry triggers and indicators."""
    try:
        triggers = ind.get("entry_triggers") or {}
        fresh_names = triggers.get("fresh_trigger_names", [])

        # Volume trend
        vol_ratio = ind.get("volume_ratio")
        if vol_ratio and vol_ratio > 1.3:
            vol_trend = "increasing on this move"
        elif vol_ratio and vol_ratio < 0.7:
            vol_trend = "fading (below average)"
        else:
            vol_trend = "normal"

        # MACD direction
        macd_hist      = ind.get("macd_hist") or 0
        macd_hist_prev = ind.get("macd_hist_prev1") or 0
        if macd_hist > macd_hist_prev:
            macd_dir = "rising"
        elif macd_hist < macd_hist_prev:
            macd_dir = "falling"
        else:
            macd_dir = "flat"

        # Setup type
        triggered_sigs = score.get("signals_triggered") or []
        if ("price_just_broke_r1_fresh" in triggered_sigs
                or triggers.get("price_just_broke_52wk_high")):
            setup_type = "breakout"
        elif triggers.get("rsi_just_crossed_30_up") or triggers.get("stochrsi_just_crossed_bullish"):
            setup_type = "bounce_off_support"
        elif triggers.get("rsi_just_crossed_70_down"):
            setup_type = "mean_reversion_oversold"
        else:
            setup_type = score.get("strategy", "unknown")

        display_names = fresh_names if fresh_names else ["NONE — setup is stale"]
        return [
            "",
            "TACTICAL CONTEXT (what JUST happened):",
            f"Fresh triggers fired this cycle: {display_names}",
            f"Volume trend: {vol_trend}",
            f"MACD direction: {macd_dir}",
            f"Setup type: {setup_type}",
        ]
    except Exception:
        return ["", "TACTICAL CONTEXT: unavailable"]


def _build_prompt(ticker: str, ind: dict, score: dict, news: Optional[dict] = None) -> str:
    def _f(val, fmt=".2f", fallback="n/a"):
        try:
            return format(float(val), fmt) if val is not None else fallback
        except (TypeError, ValueError):
            return fallback

    scorer_action = score.get("action", "hold").upper()
    net           = score.get("net_score", 0)
    confidence    = score.get("confidence", 0.0)
    position      = score.get("_position")  # live position dict or None

    # Build position context block
    if position:
        qty        = position.get("qty", 0)
        avg_entry  = position.get("avg_entry_price", 0)
        curr_price = position.get("current_price") or ind.get("current_price", 0)
        unreal_pct = position.get("unrealized_plpc")
        side       = position.get("side", "long")
        if unreal_pct is not None:
            pnl_str = f"{float(unreal_pct)*100:+.1f}%"
        else:
            pnl_str = "unknown"
        position_block = [
            "--- CURRENT POSITION (YOU ALREADY OWN THIS STOCK) ---",
            f"Side:             {side}",
            f"Shares held:      {qty}",
            f"Avg entry price:  ${_f(avg_entry)}",
            f"Current price:    ${_f(curr_price)}",
            f"Unrealized P&L:   {pnl_str}",
            "",
            "Since you already hold this position, your decisions are:",
            "  'buy'  = add more shares (only if conviction is high and risk is justified)",
            "  'sell' = exit the position now (cut the loss or take profit early)",
            "  'hold' = keep the position as-is, no new order",
            "",
        ]
        valid_decisions = '"buy" (add), "sell" (exit), or "hold" (keep)'
    else:
        position_block = []
        valid_decisions = '"buy", "short", or "hold"'

    lines = [
        f"TICKER: {ticker}",
        f"Scorer decision: {scorer_action}  (net score={net}, confidence={confidence:.0%})",
        f"Strategy hint:   {score.get('strategy', '?')}",
        f"Current price:   ${_f(score.get('entry_price'))}",
        f"Stop loss:       ${_f(score.get('stop_loss'))}",
        f"Take profit:     ${_f(score.get('take_profit'))}",
        f"Risk/reward:     {_f(score.get('risk_reward'))}",
        *position_block,
        "",
        "--- INDICATORS ---",
        f"RSI (14):        {_f(ind.get('rsi'))}",
        f"MACD histogram:  {_f(ind.get('macd_hist'))}",
        f"EMA alignment:   {'full bull' if (ind.get('ema9') or 0) > (ind.get('ema21') or 0) > (ind.get('ema50') or 0) > (ind.get('ema200') or 0) else 'partial bull' if (ind.get('ema9') or 0) > (ind.get('ema21') or 0) > (ind.get('ema50') or 0) else 'mixed/bear'}",
        f"ADX:             {_f(ind.get('adx'))}  (+DI {_f(ind.get('adx_di_plus'))} / -DI {_f(ind.get('adx_di_minus'))})",
        f"Volume ratio:    {_f(ind.get('volume_ratio'))}x avg",
        f"ATR:             {_f(ind.get('atr'))}",
        f"BB %B:           {_f(ind.get('bb_pctb'))}",
        f"VWAP diff:       price is {'above' if (ind.get('current_price') or 0) > (ind.get('vwap') or 0) else 'below'} VWAP",
        f"Intraday move:   {_f(ind.get('intraday_move_pct'))}%",
        f"Gap today:       {_f(ind.get('gap_pct'))}%",
        "",
        "--- MACRO ---",
        f"VIX:             {_f(score.get('vix') or ind.get('vix'))}",
        f"SPY regime:      {score.get('macro_bias') or ind.get('spy_regime', 'unknown')}",
        "",
        "--- NEWS ---",
        f"Sentiment polarity: {_f((news or {}).get('avg_polarity') or ind.get('news_polarity') or ind.get('news_sentiment_polarity'))}  (-1=very negative, +1=very positive)",
        f"Articles analyzed:  {(news or {}).get('headline_count', (news or {}).get('article_count', 0))}",
        f"Bull keyword boost: {(news or {}).get('bull_keyword_boost', 0)}  (earnings beat, raised guidance, FDA approved, major contract, etc.)",
        f"Bear keyword boost: {(news or {}).get('bear_keyword_boost', 0)}  (SEC probe, class action, CEO resigned, revenue miss, guidance cut, etc.)",
        f"SEC 8-K filing:     {(news or {}).get('sec_8k_flag', False)}  (material event filed with SEC in last 24h)",
        f"Earnings risk:      {(news or {}).get('earnings_risk', {}).get('risk_level', 'none')}  (block=within 3 days, warn=within 7 days)",
        "",
        "RECENT HEADLINES (read these carefully — they are the actual news):",
        *_format_headlines(news),
        "",
        "--- RULE-BASED SIGNALS ---",
        f"For:     {json.dumps(score.get('signals_triggered', []))}",
        f"Against: {json.dumps(score.get('signals_against', []))}",
        "",
        "--- SCORER REASONING ---",
        score.get("reasoning", ""),
        "",
        "--- YOUR ANALYSIS ---",
        "Before deciding, reason through the following:",
        f"  1. Read the headlines above. Do they explain WHY the stock is moving?",
        f"     Does the news support or contradict the technical setup?",
        f"     A negative headline with a buy signal = likely hold or reject.",
        f"     A positive catalyst with a buy signal = stronger case.",
        f"  2. Is the {'dip a buying opportunity or a falling knife?' if score.get('action') != 'short' else 'high a short opportunity or justified strength?'}",
        f"     Is there a business/macro reason for the move to reverse?",
        f"  3. Would a rational investor with access to these headlines and these",
        f"     indicators make this trade today, or wait for more clarity?",
        "",
        *_build_tactical_context(ind, score),
        "",
        "Based on your analysis, give your final independent decision.",
        "Your 'reasoning' must explain the business/macro logic — not just restate",
        "the indicator values. If you cannot justify the trade logically, say 'hold'.",
        "",
        "ALWAYS include specific price guidance in every response, even for holds:",
        f"  - entry_price: the specific price level where entry makes sense RIGHT NOW",
        f"    (for holds: what price would make you change to buy/short — be specific)",
        f"  - stop_loss: where you'd cut the loss (for longs: below support; for shorts: above resistance)",
        f"  - take_profit: realistic price target based on next resistance/support",
        f"  - risk_reward: take_profit distance / stop_loss distance (aim for >= 2.0)",
        f"  - entry_condition: ONE sentence describing what needs to happen before entry",
        f"    (e.g. 'Wait for RSI to cool below 55 and price to hold $142 support' or",
        f"     'Enter now — momentum confirmed, catalyst present, risk defined')",
        "",
        'Respond with ONLY a valid JSON object — no markdown, no explanation outside it:',
        f'{{"decision": {valid_decisions}, "confidence": 0.0-1.0, '
        f'"reasoning": "2-3 sentences: technicals + business logic + why now or why wait", '
        f'"entry_price": <number>, "stop_loss": <number>, "take_profit": <number>, '
        f'"risk_reward": <number>, "entry_condition": "one sentence on exact trigger"}}',
    ]
    return "\n".join(lines)


def claude_analyze_ticker(ticker: str, indicators: dict, scorer_result: dict,
                          news: Optional[dict] = None) -> dict:
    """
    Ask Claude for an independent buy/short/hold decision on one ticker.
    Never raises — returns a fallback that preserves the scorer decision.
    """
    global _credits_exhausted  # noqa: PLW0603
    scorer_action = scorer_result.get("action", "hold")

    if _credits_exhausted:
        logger.debug(f"[AI] {ticker}: credits exhausted — skipping Claude")
        return _fallback(scorer_action)

    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.debug("[AI] ANTHROPIC_API_KEY not set — skipping")
            return _fallback(scorer_action)

        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_prompt(ticker, indicators, scorer_result, news=news)

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=768,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=20,
        )

        raw = message.content[0].text.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result     = json.loads(raw)
        decision   = str(result.get("decision", scorer_action)).lower()
        confidence = float(result.get("confidence", 1.0))
        reasoning  = str(result.get("reasoning", ""))

        has_position = bool(scorer_result.get("_position"))
        valid = {"buy", "sell", "hold"} if has_position else {"buy", "short", "hold"}
        if decision not in valid:
            logger.warning(f"[AI] {ticker}: unexpected decision '{decision}' — keeping scorer decision")
            decision = scorer_action

        def _safe_float(v):
            try: return round(float(v), 2) if v is not None else None
            except: return None

        return {
            "decision":        decision,
            "confidence":      confidence,
            "reasoning":       reasoning,
            "entry_price":     _safe_float(result.get("entry_price")),
            "stop_loss":       _safe_float(result.get("stop_loss")),
            "take_profit":     _safe_float(result.get("take_profit")),
            "risk_reward":     _safe_float(result.get("risk_reward")),
            "entry_condition": str(result.get("entry_condition", "")),
        }

    except Exception as e:
        err_str = str(e).lower()
        # Detect exhausted credits / billing errors — disable Claude for rest of session
        if any(kw in err_str for kw in ("credit", "billing", "quota", "insufficient", "overloaded", "529")):
            _credits_exhausted = True
            logger.warning(f"[AI] Credits/billing error — disabling Claude for this session: {e}")
        else:
            logger.warning(f"[AI] {ticker}: error — {e} — keeping scorer decision")
        return _fallback(scorer_action)


def apply_ai_opinion(scorer_result: dict, indicators: dict,
                     news: Optional[dict] = None) -> dict:
    """
    3-tier AI filter:
      Tier 3 — Auto-hold: no Claude for clean holds below threshold
      Tier 1 — Auto-execute: no Claude for very high confidence buys/shorts
      Tier 2 — Claude reviews borderline cases

    Always returns a valid scorer_result dict, never raises.
    """
    try:
        ticker        = scorer_result.get("ticker", "?")
        scorer_action = scorer_result.get("action", "hold")
        net           = scorer_result.get("net_score", 0)
        conf          = scorer_result.get("confidence", 0.0)
        has_position  = bool(scorer_result.get("_position"))

        # Tier 3: Auto-hold (no Claude) — clean holds with low conviction and no open position
        if (scorer_action == "hold"
                and abs(net) < _CLAUDE_MIN_NET
                and not has_position):
            logger.info(f"[AI] {ticker} [SKIP] net={net} — below threshold, no Claude call")
            scorer_result.setdefault("ai_confirmed", None)
            scorer_result.setdefault("ai_reasoning", "")
            scorer_result.setdefault("ai_entry_price", None)
            scorer_result.setdefault("ai_stop_loss", None)
            scorer_result.setdefault("ai_take_profit", None)
            scorer_result.setdefault("ai_risk_reward", None)
            scorer_result.setdefault("ai_entry_condition", "")
            return scorer_result

        # Tier 1: Auto-execute (no Claude) — very high conviction buys/shorts only
        if (scorer_action in ("buy", "short")
                and net > _AUTO_EXECUTE_NET
                and conf > _AUTO_EXECUTE_CONF
                and not has_position):
            logger.info(f"[AI] {ticker} [AUTO] net={net} conf={conf:.2f} — executing without Claude")
            return _apply_defaults(scorer_result)

        # Tier 2: Claude reviews
        news_data = news or scorer_result.get("_news", {})
        ai = claude_analyze_ticker(ticker, indicators, scorer_result, news=news_data)
        ai_decision  = ai.get("decision", scorer_action)
        ai_conf      = ai.get("confidence", 1.0)
        ai_reasoning = ai.get("reasoning", "")

        scorer_result["ai_confirmed"]       = (ai_decision == scorer_action)
        scorer_result["ai_reasoning"]       = ai_reasoning
        scorer_result["ai_entry_price"]     = ai.get("entry_price")
        scorer_result["ai_stop_loss"]       = ai.get("stop_loss")
        scorer_result["ai_take_profit"]     = ai.get("take_profit")
        scorer_result["ai_risk_reward"]     = ai.get("risk_reward")
        scorer_result["ai_entry_condition"] = ai.get("entry_condition", "")

        if has_position:
            if ai_decision == "sell":
                logger.info(f"[AI] {ticker}: SELL (exit position) recommended by Claude — {ai_reasoning}")
                scorer_result["action"] = "sell"
            elif ai_decision == "buy":
                logger.info(f"[AI] {ticker}: ADD MORE recommended by Claude (conf={ai_conf:.2f}) — {ai_reasoning}")
                scorer_result["action"] = "buy"
                scorer_result["confidence"] = ai_conf
            else:
                logger.info(f"[AI] {ticker}: HOLD position — {ai_reasoning}")
                scorer_result["action"] = "hold"
        elif ai_decision == scorer_action:
            logger.info(
                f"[AI] {ticker}: {scorer_action.upper()} CONFIRMED by Claude "
                f"(conf={ai_conf:.2f}) — {ai_reasoning}"
            )
        elif ai_decision == "hold" and scorer_action != "hold":
            logger.info(
                f"[AI] {ticker}: {scorer_action.upper()} DOWNGRADED to HOLD by Claude — {ai_reasoning}"
            )
            scorer_result["action"] = "hold"
        elif ai_decision in ("buy", "short") and scorer_action == "hold":
            logger.info(
                f"[AI] {ticker}: HOLD UPGRADED to {ai_decision.upper()} by Claude "
                f"(conf={ai_conf:.2f}) — {ai_reasoning}"
            )
            scorer_result["action"] = ai_decision
            scorer_result["confidence"] = ai_conf
        else:
            logger.info(
                f"[AI] {ticker}: Claude said {ai_decision.upper()} vs scorer {scorer_action.upper()} "
                f"— defaulting to HOLD"
            )
            scorer_result["action"] = "hold"

        return scorer_result

    except Exception as e:
        logger.warning(f"[AI] apply_ai_opinion error for {scorer_result.get('ticker','?')}: {e}")
        return scorer_result


def run_ai_filter_batch(scored_pairs: list[tuple[dict, dict]]) -> list[dict]:
    """
    Run apply_ai_opinion concurrently across a list of (scorer_result, indicators) tuples.
    Returns the list of updated scorer_result dicts in the same order.
    """
    if not scored_pairs:
        return []

    results = [None] * len(scored_pairs)

    # Pre-filter: resolve Tier 1 (auto-execute) and Tier 3 (auto-hold) without spawning threads
    needs_claude = {}
    for i, pair in enumerate(scored_pairs):
        s = pair[0]
        action       = s.get("action", "hold")
        net          = s.get("net_score", 0)
        conf         = s.get("confidence", 0.0)
        has_position = bool(s.get("_position"))
        # Tier 3: auto-hold
        if (action == "hold"
                and abs(net) < _CLAUDE_MIN_NET
                and not has_position):
            s.setdefault("ai_confirmed", None)
            s.setdefault("ai_reasoning", "")
            s.setdefault("ai_entry_price", None)
            s.setdefault("ai_stop_loss", None)
            s.setdefault("ai_take_profit", None)
            s.setdefault("ai_risk_reward", None)
            s.setdefault("ai_entry_condition", "")
            results[i] = s
        # Tier 1: auto-execute
        elif (action in ("buy", "short")
                and net > _AUTO_EXECUTE_NET
                and conf > _AUTO_EXECUTE_CONF
                and not has_position):
            results[i] = _apply_defaults(s)
        else:
            needs_claude[i] = pair

    if needs_claude:
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            future_to_idx = {
                ex.submit(apply_ai_opinion, pair[0], pair[1],
                          pair[0].get("_news", {})): i
                for i, pair in needs_claude.items()
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.warning(f"[AI] batch future error at index {idx}: {e}")
                    results[idx] = needs_claude[idx][0]

    logger.info(f"[AI] batch: {len(needs_claude)} Claude calls, {len(scored_pairs)-len(needs_claude)} clean-hold skips")
    return results


# Keep old name as alias so execute_signals import doesn't break during transition
def apply_ai_confirmation(scorer_result: dict, indicators: dict) -> dict:
    return apply_ai_opinion(scorer_result, indicators)
