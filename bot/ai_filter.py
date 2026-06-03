"""Claude Sonnet final confirmation gate for trade signals.

apply_ai_confirmation(scorer_result, indicators) is the public entry point.
If the scorer says "hold", it passes through immediately.
If Claude is unavailable for any reason, the trade proceeds unchanged.
"""

import json
import logging
import os

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a professional quantitative trading analyst. "
    "You are the final confirmation before a trade is placed. "
    "Analyze the provided indicators and signal data and decide if this trade should proceed. "
    "Be conservative — if in doubt, reject it."
)

_FALLBACK = {
    "decision":   "confirm",
    "confidence": 1.0,
    "reasoning":  "AI filter unavailable — passing through",
}


def _build_prompt(ticker: str, ind: dict, score: dict) -> str:
    def _f(val, fmt=".2f", fallback="n/a"):
        try:
            return format(float(val), fmt) if val is not None else fallback
        except (TypeError, ValueError):
            return fallback

    lines = [
        f"TRADE SIGNAL — {ticker}",
        "",
        f"Action:          {score.get('action', '?').upper()}",
        f"Strategy:        {score.get('strategy', '?')}",
        f"Current price:   ${_f(score.get('entry_price'))}",
        f"Net score:       {score.get('net_score', 0)}",
        f"Confidence:      {_f(score.get('confidence'), '.2%')}",
        f"Stop loss:       ${_f(score.get('stop_loss'))}",
        f"Take profit:     ${_f(score.get('take_profit'))}",
        f"Risk/reward:     {_f(score.get('risk_reward'))}",
        "",
        "--- INDICATORS ---",
        f"RSI (14):        {_f(ind.get('rsi'))}",
        f"MACD histogram:  {_f(ind.get('macd_hist'))}",
        f"EMA9:            {_f(ind.get('ema9'))}",
        f"EMA21:           {_f(ind.get('ema21'))}",
        f"EMA50:           {_f(ind.get('ema50'))}",
        f"EMA200:          {_f(ind.get('ema200'))}",
        f"EMA alignment:   {'bull' if (ind.get('ema9') or 0) > (ind.get('ema21') or 0) > (ind.get('ema50') or 0) else 'mixed/bear'}",
        f"Volume ratio:    {_f(ind.get('volume_ratio'))}x",
        f"ATR:             {_f(ind.get('atr'))}",
        f"ADX:             {_f(ind.get('adx'))}",
        f"BB %B:           {_f(ind.get('bb_pctb'))}",
        f"Intraday move:   {_f(ind.get('intraday_move_pct'))}%",
        "",
        "--- MACRO ---",
        f"VIX:             {_f(ind.get('vix') or score.get('vix_level'))}",
        f"SPY regime:      {ind.get('spy_regime') or score.get('macro_bias', 'unknown')}",
        "",
        "--- NEWS ---",
        f"News sentiment:  {_f(ind.get('news_polarity') or ind.get('news_sentiment_polarity'))}",
        "",
        "--- SIGNALS ---",
        f"For:    {json.dumps(score.get('signals_triggered', []))}",
        f"Against:{json.dumps(score.get('signals_against', []))}",
        "",
        "--- REASONING ---",
        score.get("reasoning", ""),
        "",
        'Respond with ONLY a valid JSON object, nothing else: '
        '{"decision": "confirm" or "reject", "confidence": 0.0-1.0, "reasoning": "max two sentences"}',
    ]
    return "\n".join(lines)


def claude_confirm_trade(ticker: str, indicators: dict, scorer_result: dict) -> dict:
    """
    Call Claude Sonnet for a final trade confirmation.
    Returns a dict with keys: decision, confidence, reasoning.
    Never raises — on any failure returns _FALLBACK.
    """
    try:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            logger.debug("[AI FILTER] ANTHROPIC_API_KEY not set — skipping")
            return _FALLBACK

        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        prompt = _build_prompt(ticker, indicators, scorer_result)

        message = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=256,
            system=_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
            timeout=20,
        )

        raw = message.content[0].text.strip()

        # Strip any accidental markdown fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        result = json.loads(raw)

        decision   = str(result.get("decision", "confirm")).lower()
        confidence = float(result.get("confidence", 1.0))
        reasoning  = str(result.get("reasoning", ""))

        if decision not in ("confirm", "reject"):
            logger.warning(f"[AI FILTER] {ticker}: unexpected decision '{decision}' — treating as confirm")
            decision = "confirm"

        return {"decision": decision, "confidence": confidence, "reasoning": reasoning}

    except Exception as e:
        logger.warning(f"[AI FILTER] {ticker}: error — {e} — passing through")
        return _FALLBACK


def apply_ai_confirmation(scorer_result: dict, indicators: dict) -> dict:
    """
    Run the AI confirmation gate and mutate scorer_result in place.

    - action == "hold"  →  returns immediately, no Claude call
    - Claude "confirm"  →  attaches ai_confirmed=True, ai_reasoning
    - Claude "reject"   →  changes action to "hold", attaches ai_confirmed=False, ai_reasoning
    Always returns a valid scorer_result dict, never raises.
    """
    try:
        action = scorer_result.get("action", "hold")
        if action == "hold":
            return scorer_result

        ticker = scorer_result.get("ticker", "?")
        result = claude_confirm_trade(ticker, indicators, scorer_result)

        decision  = result.get("decision", "confirm")
        reasoning = result.get("reasoning", "")

        scorer_result["ai_confirmed"] = (decision == "confirm")
        scorer_result["ai_reasoning"] = reasoning

        if decision == "reject":
            logger.info(
                f"[AI FILTER] {ticker}: TRADE REJECTED by Claude — {reasoning}"
            )
            scorer_result["action"] = "hold"
        else:
            logger.info(
                f"[AI FILTER] {ticker}: trade CONFIRMED by Claude "
                f"(conf={result.get('confidence', 1.0):.2f}) — {reasoning}"
            )

        return scorer_result

    except Exception as e:
        logger.warning(f"[AI FILTER] apply_ai_confirmation error — {e} — returning unchanged")
        return scorer_result
