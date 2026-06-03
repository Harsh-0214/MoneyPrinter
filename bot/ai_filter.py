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

_SYSTEM_PROMPT = (
    "You are a professional quantitative trading analyst reviewing every stock "
    "a rules-based algorithm has just scored. You receive the full indicator "
    "picture and the algorithm's preliminary decision. Your job is to give an "
    "independent second opinion: confirm the decision, override it, or change it. "
    "Be conservative — only recommend 'buy' or 'short' when the evidence is clear. "
    "If in doubt, say 'hold'."
)

# Tickers with net_score below this in absolute value are noise — skip Claude.
_MIN_NET_FOR_AI = 20

# Max parallel Claude calls per batch (avoids rate-limit bursts)
_MAX_WORKERS = 6


def _fallback(scorer_action: str) -> dict:
    """Pass-through fallback that preserves the scorer's original decision."""
    return {
        "decision":   scorer_action,
        "confidence": 1.0,
        "reasoning":  "AI filter unavailable — scorer decision kept",
    }


def _build_prompt(ticker: str, ind: dict, score: dict, news: Optional[dict] = None) -> str:
    def _f(val, fmt=".2f", fallback="n/a"):
        try:
            return format(float(val), fmt) if val is not None else fallback
        except (TypeError, ValueError):
            return fallback

    scorer_action = score.get("action", "hold").upper()
    net           = score.get("net_score", 0)
    confidence    = score.get("confidence", 0.0)

    lines = [
        f"TICKER: {ticker}",
        f"Scorer decision: {scorer_action}  (net score={net}, confidence={confidence:.0%})",
        f"Strategy hint:   {score.get('strategy', '?')}",
        f"Current price:   ${_f(score.get('entry_price'))}",
        f"Stop loss:       ${_f(score.get('stop_loss'))}",
        f"Take profit:     ${_f(score.get('take_profit'))}",
        f"Risk/reward:     {_f(score.get('risk_reward'))}",
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
        f"Articles analyzed:  {(news or {}).get('article_count', 'n/a')}",
        f"Bull keyword boost: {(news or {}).get('bull_keyword_boost', 0)}",
        f"Bear keyword boost: {(news or {}).get('bear_keyword_boost', 0)}",
        f"SEC 8-K filing:     {(news or {}).get('sec_8k_flag', False)}",
        f"Earnings risk:      {(news or {}).get('earnings_risk', {}).get('risk_level', 'none')}",
        f"Recent headlines:   {'; '.join(((news or {}).get('headlines', []))[:3]) or 'none available'}",
        "",
        "--- RULE-BASED SIGNALS ---",
        f"For:     {json.dumps(score.get('signals_triggered', []))}",
        f"Against: {json.dumps(score.get('signals_against', []))}",
        "",
        "--- SCORER REASONING ---",
        score.get("reasoning", ""),
        "",
        "Based on ALL of the above, what is your independent recommendation?",
        "",
        'Respond with ONLY a valid JSON object — no markdown, no explanation outside it:',
        '{"decision": "buy" or "short" or "hold", "confidence": 0.0-1.0, "reasoning": "max two sentences"}',
    ]
    return "\n".join(lines)


def claude_analyze_ticker(ticker: str, indicators: dict, scorer_result: dict,
                          news: Optional[dict] = None) -> dict:
    """
    Ask Claude for an independent buy/short/hold decision on one ticker.
    Never raises — returns a fallback that preserves the scorer decision.
    """
    scorer_action = scorer_result.get("action", "hold")
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
            max_tokens=256,
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

        if decision not in ("buy", "short", "hold"):
            logger.warning(f"[AI] {ticker}: unexpected decision '{decision}' — keeping scorer decision")
            decision = scorer_action

        return {"decision": decision, "confidence": confidence, "reasoning": reasoning}

    except Exception as e:
        logger.warning(f"[AI] {ticker}: error — {e} — keeping scorer decision")
        return _fallback(scorer_action)


def apply_ai_opinion(scorer_result: dict, indicators: dict,
                     news: Optional[dict] = None) -> dict:
    """
    Run Claude's second opinion on any ticker, regardless of scorer action.
    Skips tickers where |net_score| < _MIN_NET_FOR_AI (pure noise).

    Possible outcomes:
      scorer buy  + Claude buy   → confirmed buy
      scorer buy  + Claude hold  → downgraded to hold
      scorer buy  + Claude short → downgraded to hold (don't reverse blindly)
      scorer hold + Claude buy   → upgraded to buy
      scorer hold + Claude hold  → hold unchanged
      scorer short + Claude short → confirmed short
      scorer short + Claude hold  → downgraded to hold
    Always returns a valid scorer_result dict, never raises.
    """
    try:
        ticker       = scorer_result.get("ticker", "?")
        scorer_action = scorer_result.get("action", "hold")
        net_score    = abs(scorer_result.get("net_score", 0))

        if net_score < _MIN_NET_FOR_AI:
            logger.debug(f"[AI] {ticker}: net={net_score} < {_MIN_NET_FOR_AI} — skipping AI")
            scorer_result["ai_confirmed"] = None
            scorer_result["ai_reasoning"] = "skipped — net score below AI threshold"
            return scorer_result

        news = news or scorer_result.get("_news", {})
        ai = claude_analyze_ticker(ticker, indicators, scorer_result, news=news)
        ai_decision  = ai.get("decision", scorer_action)
        ai_conf      = ai.get("confidence", 1.0)
        ai_reasoning = ai.get("reasoning", "")

        scorer_result["ai_confirmed"] = (ai_decision == scorer_action)
        scorer_result["ai_reasoning"] = ai_reasoning

        if ai_decision == scorer_action:
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
            # Claude's confidence becomes the signal confidence when it overrides
            scorer_result["confidence"] = ai_conf
        else:
            # e.g. scorer=buy, claude=short → don't blindly reverse, just hold
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

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        future_to_idx = {
            ex.submit(apply_ai_opinion, pair[0], pair[1],
                      pair[0].get("_news", {})): i
            for i, pair in enumerate(scored_pairs)
        }
        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"[AI] batch future error at index {idx}: {e}")
                results[idx] = scored_pairs[idx][0]  # return unchanged on error

    return results


# Keep old name as alias so execute_signals import doesn't break during transition
def apply_ai_confirmation(scorer_result: dict, indicators: dict) -> dict:
    return apply_ai_opinion(scorer_result, indicators)
