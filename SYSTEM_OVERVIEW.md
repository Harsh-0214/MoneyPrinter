# MoneyPrinter Trading Bot — System Overview

## 1. What This Bot Does

MoneyPrinter is a fully autonomous, rules-based algorithmic trading bot designed to identify and execute high-conviction intraday and swing trades on U.S. equities listed on the Alpaca paper-trading (or live) API. The bot runs entirely within GitHub Actions — no server, no manual intervention required. Each day it wakes up at scheduled times, scans a curated watchlist of liquid large-cap and momentum tickers, applies a deterministic multi-signal scoring engine, and submits limit orders when conviction thresholds are met.

The core philosophy is **systematic over intuitive**: every buy or short decision is produced by the same reproducible set of rules applied to real-time market data. More than 30 individual technical, volume, momentum, news, and macro signals contribute to a composite score. No machine-learning model or external AI API is consulted during live trading — the logic is pure Python, fully auditable, and version-controlled. This design prioritises reliability and explainability over theoretical return maximisation.

Position sizing, stop losses, take profits, and kill-switch logic are all embedded directly in the risk engine. The bot will refuse to open new positions when market-wide fear is extreme (VIX ≥ 35), when it has already lost 3% of portfolio value in a single day, or when the broader SPY trend is bearish. Every executed trade is persisted to an SQLite database and summarised in a self-contained daily HTML report.

---

## 2. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Actions                               │
│   .github/workflows/                                                │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│   │discovery │ │premarket │ │market_   │ │midday.yml│  ...         │
│   │  .yml    │ │  .yml    │ │open.yml  │ │          │             │
│   └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘             │
│        └────────────┴────────────┴─────────────┘                   │
│                              │                                      │
│                       python main.py                                │
│                         --session X                                 │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                ┌──────────────▼──────────────┐
                │         main.py             │
                │  get_macro_context()        │
                │  run_full_scan()            │
                │  execute_signals()          │
                │  session_*()                │
                └──────┬──────────────────────┘
                       │
          ┌────────────┼──────────────────┐
          │            │                  │
   ┌──────▼─────┐ ┌────▼──────┐  ┌───────▼──────┐
   │indicators  │ │  news.py  │  │  scorer.py   │
   │   .py      │ │ TextBlob  │  │  30+ signals │
   │ yfinance   │ │ NewsAPI   │  │  bull/bear   │
   │ ta library │ │ RSS feeds │  │  net_score   │
   │ 15-min VWAP│ │ SEC 8-K   │  │  confidence  │
   │ parallel   │ │ keyword   │  │  strategy    │
   │ fetch 8x   │ │ amplifier │  │  pick        │
   └──────┬─────┘ └────┬──────┘  └───────┬──────┘
          └────────────┘                  │
                                   ┌──────▼──────┐
                                   │  risk.py    │
                                   │ position    │
                                   │ sizing      │
                                   │ kill switch │
                                   │ trailing    │
                                   │ stop        │
                                   └──────┬──────┘
                                          │
                                   ┌──────▼──────┐
                                   │  trader.py  │
                                   │ Alpaca API  │
                                   │ limit orders│
                                   │ fill check  │
                                   └──────┬──────┘
                                          │
                          ┌───────────────┼───────────────┐
                          │               │               │
                   ┌──────▼──────┐ ┌──────▼──────┐ ┌─────▼──────────┐
                   │  logger.py  │ │ portfolio.py│ │reports/        │
                   │ SQLite DB   │ │ stop checks │ │daily_report.py │
                   │ data/       │ │ target hit  │ │YYYY-MM-DD.html │
                   │ trades.db   │ │ time exits  │ │                │
                   └─────────────┘ └─────────────┘ └────────────────┘
```

---

## 3. Daily Schedule

| Session | Cron (EDT) | GitHub Workflow | What It Does | Decisions Made |
|---|---|---|---|---|
| `discovery` | 8:30 AM | `discovery.yml` | Screens ~100 large-cap tickers for abnormal volume, gap moves, and proximity to 52-week highs. Saves up to 10 movers to `discovered_tickers.json`. | Which tickers to add to the day's scan universe beyond the static watchlist |
| `premarket` | 9:00 AM | `premarket.yml` | Fetches overnight news, scores sentiment, logs gap-ups/gap-downs (>2%). No trades placed. | Flags catalysts; logs scan activity |
| `market_open` | 9:35 AM | `market_open.yml` | Full scan of all tickers. Scores every ticker with all 30+ signals. Executes up to 3 trades at ≥65 net score, ≥65% confidence. | Primary entry session — highest trade count day |
| `midday` | 12:00 PM | `midday.yml` | Checks open positions for stops/targets/time exits. New entries only if net_score > 80 AND confidence ≥ 85% (1 trade max). | Stop/target exits; rare high-conviction midday additions |
| `market_close` | 3:30 PM | `market_close.yml` | Closes all scalp positions. Re-scores for overnight hold candidates. Closes positions where signal has flipped. | Which positions to hold overnight vs close |
| `eod_summary` | 4:15 PM | `eod_summary.yml` | Computes daily P&L, win rate, best/worst trade. Logs daily summary to DB. Generates HTML report. Commits `data/trades.db` + `reports/*.html` to git. | No trades — reporting and persistence only |
| `continuous` | 9:30 AM–4:00 PM | `trading_day.yml` | All-day loop scanning every 5 minutes. Replaces the above discrete sessions when running as a long-lived job. Includes trailing stop checks every cycle. | All entry, exit, and trailing stop decisions throughout the session |

---

## 4. Complete Indicator Reference

### TREND Indicators

**EMA (9, 21, 50, 200) — Exponential Moving Averages**
- Full bull alignment (EMA9 > EMA21 > EMA50 > EMA200): **+25 bull × ADX multiplier**
- Partial bull alignment (EMA9 > EMA21 > EMA50): **+18 bull × ADX multiplier**
- Full bear alignment (EMA9 < EMA21 < EMA50 < EMA200): **+25 bear × ADX multiplier**
- Partial bear alignment: **+18 bear × ADX multiplier**
- ADX multiplier: ADX > 30 → 1.3×; ADX < 20 → 0.5×; otherwise 1.0×

**MACD (12, 26, 9) — Moving Average Convergence Divergence**
- Histogram positive and rising 2+ consecutive bars: **+15 bull**
- Histogram negative and falling 2+ consecutive bars: **+15 bear**
- Bullish crossover (MACD line crosses above signal, last 3 bars): **+12 bull**
- Bearish crossover: **+12 bear**
- Histogram fading (positive but declining): **−5 bull** (signals_against)

**ADX (14) — Average Directional Index**
- ADX > 30: 1.3× multiplier on EMA score (strong confirmed trend)
- ADX < 20: 0.5× multiplier on EMA score (weak/no trend)
- DI+ dominant (DI+ > DI− when ADX > 25): **+8 bull**
- DI− dominant: **+8 bear**
- Short blocked when ADX > 30 and DI+ > DI− (strong uptrend)

**Parabolic SAR**
- Bullish SAR (price above SAR): **+8 bull**
- Bearish SAR (price below SAR): **+8 bear**

**VWAP — Volume Weighted Average Price**
- Price > 0.3% above VWAP: **+8 bull** (`price_above_vwap`)
- Price > 0.3% below VWAP: **+8 bear** (`price_below_vwap`)

---

### MOMENTUM Indicators

**RSI (14) — Relative Strength Index**
| RSI Range | Signal | Points |
|---|---|---|
| < 20 | Extremely oversold | +25 bull |
| 20–30 | Oversold | +15 bull |
| 30–40 | Weak | +5 bear |
| 40–50 | Bearish momentum | +8 bear |
| 50–60 | Healthy bull momentum | +8 bull |
| 60–70 | Strong | +5 bull |
| 70–80 | Overbought | +15 bear (signals_against) |
| > 80 | Extremely overbought | +25 bear (signals_against) |

**Stochastic RSI (14, 14, 3, 3)**
- K < 20 and K turning up (K > D): **+10 bull**
- K > 80 and K turning down (K < D): **+10 bear**
- Bullish cross below 30 (K crosses above D, both < 30): **+12 bull** (`stochrsi_bull_cross_below30`)
- Bearish cross above 70: **+12 bear**

**CCI (20) — Commodity Channel Index**
- CCI < −200: **+15 bull** (extremely oversold)
- CCI < −100: **+8 bull** (oversold)
- CCI > 100: **+8 bear** (signals_against)
- CCI > 200: **+15 bear** (signals_against)

**Williams %R (14)**
- Williams %R < −80 (oversold): **+8 bull**
- Williams %R > −20 (overbought): **+8 bear**

**Rate of Change (10)**
- ROC > 3%: **+6 bull**
- ROC < −3%: **+6 bear**

---

### VOLATILITY Indicators

**Bollinger Bands (20, 2)**
- Price at or below lower band (not in squeeze): **+12 bull**
- %B < 0.1 (deeply oversold): **+18 bull**
- %B > 0.9 (deeply overbought): **+18 bear**
- BB squeeze detected (bandwidth in bottom 20% of 20-bar range): logs `bb_squeeze_detected`; watch mode — score held pending breakout confirmation
- BB bandwidth expanding with positive net score: **+8 bull**
- BB bandwidth expanding with negative net score: **+8 bear**

**ATR (14) — Average True Range**
- ATR% > 4%: `high_vol_flag = True` → position size reduced 40% in risk engine
- Not a direct score signal; used for stop/target calculation in all strategies

**Keltner Channel (20, 2)**
- Price breaks above upper Keltner after BB squeeze: **+15 bull** (`kc_breakout_bull`)
- Price breaks below lower Keltner: **+15 bear**

---

### VOLUME Indicators

**OBV — On-Balance Volume**
- OBV slope positive (10-bar regression): **+8 bull**
- OBV slope negative: **+8 bear**
- Bull divergence (OBV at 20-bar high but price not): **+10 bull** (`obv_bull_divergence`)
- Bear divergence: **+10 bear**

**Volume Ratio (today vs 20-day average)**
- Surge ≥ 2× with price up: **+20 bull** (`volume_surge_bull`)
- Surge ≥ 2× with price down: **+20 bear**
- Confirm ≥ 1.5× with price up: **+12 bull** (`volume_confirm_bull`)
- Confirm ≥ 1.5× with price down: **+12 bear**
- Low < 0.7×: all accumulated bull/bear scores multiplied by **0.8×** (low-conviction penalty)

**MFI (14) — Money Flow Index**
- MFI < 20 (oversold): **+10 bull**
- MFI > 80 (overbought): **+10 bear**

---

### SUPPORT / RESISTANCE

**Pivot Points (classic, prior day OHLC)**
- Near S1 (within 0.3%): **+10 bull**
- Near S2: **+15 bull**
- Near R1: **+8 bear** (signals_against)
- Near R2: **+12 bear**
- Broke above R1 with volume ≥ 1.5×: **+20 bull** (overrides R1 resistance)
- Broke below S1 with volume ≥ 1.5×: **+15 bear** → triggers `breakdown` strategy

**52-Week Range**
- Within 2% of 52-week high without volume: **+10 bear** (resistance)
- Within 2% of 52-week high WITH volume ≥ 1.5×: **+20 bull** (breakout)
- Within 3% of 52-week low: **+10 bull** (support)

**EMA200 Short Filter**
- Price above EMA200: bear score reduced by 20 (do not fight long-term uptrend with shorts)

---

### NEWS Signals

**TextBlob Sentiment (NewsAPI + RSS feeds)**
- avg_polarity > 0.5: **+22 bull** (`news_very_positive`)
- avg_polarity > 0.3: **+15 bull** (`news_positive`)
- avg_polarity < −0.5: **+22 bear**
- avg_polarity < −0.3: **+15 bear**

**SEC 8-K Filing** (EDGAR atom feed)
- Recent 8-K detected: **+20 bull or bear** (amplifies whichever is dominant)

**Earnings Proximity** (yfinance calendar)
- Within 3 days: **BLOCK** — trade not placed (binary binary outcome risk)
- Within 7 days: **WARN** — confidence reduced by 0.20, `earnings_proximity` added to signals_against

**Keyword Amplifier** (on combined headline text)

Massive Bull (+25 each): `jensen huang`, `elon musk`, `trillion dollar`, `record revenue`, `beats expectations`, `raised guidance`, `major contract`, `fda approved`, `partnership with nvidia`, `ai breakthrough`, `blowout quarter`, `record earnings`

Massive Bear (+25 each): `sec investigation`, `class action`, `ceo resigned`, `revenue miss`, `guidance cut`, `data breach`, `recall`, `bankruptcy`, `delisted`, `doj probe`, `going concern`, `fraud`

Moderate Bull (+12 each): `upgrade`, `buy rating`, `price target raised`, `strong demand`, `market share gain`, `beat estimates`, `raised outlook`

Moderate Bear (+12 each): `downgrade`, `sell rating`, `price target cut`, `disappointing`, `headwinds`, `miss`, `below expectations`

---

### Intraday 15-Minute Signals

These are computed from 5-day, 15-minute bar history fetched separately from daily data.

- Price above intraday VWAP: **+5 bull**
- Price below intraday VWAP: **+5 bear**
- Intraday RSI(14) < 35: **+8 bull** (`intraday_rsi_oversold`)
- Intraday RSI(14) > 65: **+8 bear** (`intraday_rsi_overbought`)
- Intraday MACD histogram is available in the returned indicator dict but not scored directly (available for future use)

---

## 5. Scoring System

### How Scores Accumulate

Every ticker is scored by iterating through all signal groups in order. Two running totals are maintained:

- **`bull`** — raw point total from bullish signals
- **`bear`** — raw point total from bearish signals

At the end:
```
net_score  = bull - bear
confidence = clamp(net_score / 100.0, -1.0, 1.0)
```

Confidence represents directional conviction: 0.65 = 65% confidence long, −0.70 = 70% confidence short.

### Macro Modifiers Applied After Accumulation
- SPY caution regime: `bull *= 0.80` (−20%)
- SPY bear regime: `bull *= 0.60` (−40%)
- SPY bull regime: `bear *= 0.50` (−50% — shorts rarely work in bull market)

### Trade Decision Thresholds
| Condition | Action |
|---|---|
| net_score ≥ 65 AND confidence ≥ 0.65 | **BUY** |
| net_score ≤ −70 AND abs(confidence) ≥ 0.70 AND VIX < 25 | **SHORT** |
| Otherwise | **HOLD** |

Confidence is additionally penalised by −0.05 when strategy resolves to `mixed`, and by −0.20 when earnings are within 7 days.

### Worked Example — NVDA Buy Signal

| Signal | Component | Points |
|---|---|---|
| EMA9 > EMA21 > EMA50 > EMA200 | Full bull alignment | +25 |
| ADX = 34 > 30 | Strong trend multiplier | × 1.3 → **+32.5** |
| DI+ = 28 > DI− = 14 | DI+ dominant | +8 |
| MACD histogram positive, rising 2 bars | Momentum confirming | +15 |
| RSI = 56 (50–60) | Healthy momentum | +8 |
| Volume ratio = 2.1× with price up | Volume surge bull | +20 |
| OBV rising | Accumulation | +8 |
| Price 0.6% above VWAP | Above VWAP | +8 |
| "beats expectations" in headline | Keyword bull boost | +25 |
| **SPY bull regime** | Bear signals × 0.5 | — |
| **bear = 5 × 0.5 = 2.5, bull = 124.5** | | |
| **net_score = 122** | **confidence = 1.00 (clamped)** | |
| **ACTION: BUY** | net ≥ 65, conf ≥ 0.65 | |

---

## 6. Strategy Definitions

### `trend_follow`
**Trigger:** EMA9 > EMA21 > EMA50 (or full stack), ADX > 18, MACD histogram positive, volume confirming (≥ 1.5×).
**Time horizon:** Swing (target 5–20 days).
**Stop:** 2.5 × ATR below entry. **Target:** 2.5 × 2.5 ATR above entry (R:R = 2.5).
**Example:** NVDA at $880, ATR = $18. Stop = $835, Target = $993. Hold 10–15 days.

### `mean_reversion`
**Trigger:** RSI < 38 or > 68, OR Bollinger %B < 0.15 or > 0.85, AND NOT in full EMA bull alignment (don't buy oversold stocks in downtrends).
**Time horizon:** Scalp (1–3 days).
**Stop:** 2.5 × ATR. **Target:** R:R = 2.0.
**Example:** COIN RSI = 22, %B = 0.06. Buy the bounce expecting 48-hour mean reversion.

### `breakout`
**Trigger:** Price within 2% above R1 or 52-week high AND volume ratio > 1.3×.
**Time horizon:** Swing.
**Stop:** 2.0 × ATR (tight — breakouts that fail usually fail fast). **Target:** R:R = 3.0.
**Example:** AMD breaks 52-week high at $185 on 2.3× volume. Entry $187, Stop $181, Target $199.

### `breakdown`
**Trigger:** Price breaks below S1 with volume ≥ 1.5× (`broke_below_s1_with_volume` signal).
**Time horizon:** Swing.
**Stop:** 2.0 × ATR above entry. **Target:** R:R = 2.5.
**Example:** SMCI breaks S1 = $38 on 1.8× volume. Short entry $37.50.

### `squeeze_breakout`
**Trigger:** Bollinger Band squeeze detected (bandwidth in bottom 20% of 20-bar range) AND price breaks above upper Keltner Channel.
**Time horizon:** Swing.
**Stop:** 2.5 × ATR. **Target:** R:R = 2.5.
**Example:** PLTR in 10-day tight range, volatility compressed. KC breakout above $22.50 fires signal.

### `news_momentum`
**Trigger:** News sentiment positive/very_positive AND either EMA alignment or volume confirmation.
**Time horizon:** Scalp (hours to 1 day).
**Stop:** 2.0 × ATR. **Target:** R:R = 2.0.
**Example:** NVDA headline "major contract" (+25 keyword boost), positive polarity (+15). Quick scalp trade on the news catalyst.

### `mixed`
**Trigger:** No dominant strategy pattern. Confidence penalised by −0.05. Nearly always results in `hold` action after the penalty.
**Note:** The system deliberately avoids `mixed` trades — if you can't identify *why* you're buying, don't.

---

## 7. Risk Management Rules

### Position Sizing Formula
```
dollar_risk  = portfolio_value × 0.02 × confidence × vix_multiplier × vol_adj
shares       = floor(dollar_risk / (ATR × 1.5))
max_shares   = floor(portfolio_value × 0.10 / price)
final_shares = min(shares, max_shares)
```

Where `vol_adj = 0.60` if `atr_pct > 4%` (high volatility), else `1.0`.

**Example:** Portfolio $100,000, confidence 0.80, ATR $3.00, VIX = 18 (mult = 0.85).
- dollar_risk = $100,000 × 0.02 × 0.80 × 0.85 × 1.0 = $1,360
- shares = floor($1,360 / ($3.00 × 1.5)) = floor(302.2) = **302 shares**

### VIX Position Multipliers
| VIX Level | Multiplier | Effect |
|---|---|---|
| < 15 | 1.00 | Full size |
| 15–20 | 0.85 | −15% |
| 20–25 | 0.70 | −30% |
| 25–35 | 0.50 | −50% |
| ≥ 35 | 0.00 | No new longs; kill switch on shorts |

### Kill Switch
Activated when daily realised P&L < −3% of starting portfolio value. Once triggered, no new orders are placed for the remainder of the session.

Example: Portfolio starts at $100,000. After three losing trades totalling −$3,100 (−3.1%), kill switch activates.

### Trailing Stop Loss
- Activated only after position gains ≥ 8% from entry (`TRAILING_ACTIVATE_PCT = 0.08`)
- Once activated, stop trails 5% below the highest price seen (`TRAILING_TRAIL_PCT = 0.05`)
- Trail only moves up, never down (locks in profits as price rises)
- Updated every scan cycle in `session_continuous()`
- Persisted to DB via `highest_price_seen` and `trailing_stop_price` columns

Example: Entry $100, trail activates at $108. If price reaches $120, trail = $120 × 0.95 = **$114**. If price drops to $114, position is closed.

### Stop Losses (Strategy-Specific ATR Multipliers)
| Strategy | SL Multiplier | TP R:R |
|---|---|---|
| position | 3.0 × ATR | 4.0 |
| trend_follow | 2.5 × ATR | 2.5 |
| breakout | 2.0 × ATR | 3.0 |
| breakdown | 2.0 × ATR | 2.5 |
| squeeze_breakout | 2.5 × ATR | 2.5 |
| mean_reversion | 2.5 × ATR | 2.0 |
| news_momentum | 2.0 × ATR | 2.0 |

### Time Exits
Positions that remain open beyond their maximum hold window are automatically closed by `check_time_exits()`:
- scalp → 5 days
- swing → 20 days
- position → 45 days

### Correlation Guard
No more than 2 positions are held simultaneously within any correlation group:
- `AI_CHIPS`: NVDA, AMD, MRVL, SMCI, AVGO
- `BIG_TECH`: AAPL, MSFT, GOOGL, META, AMZN
- `CRYPTO_ADJACENT`: COIN, MSTR, SOFI
- `ENERGY`: XOM, CVX

---

## 8. Macro Filter Logic

### SPY Regime Classification
SPY's price vs EMA50 and EMA200 determines the regime applied in each session:

| Condition | Regime | Bull Discount | Bear Discount | Buy Suppressed? |
|---|---|---|---|---|
| SPY > EMA50 AND > EMA200 | **bull** | — | −50% | No |
| SPY < EMA50, > EMA200 | **caution** | −20% | — | No (but all buys need extra conviction) |
| SPY < EMA50 AND < EMA200 | **bear** | −40% | — | Yes, in `run_full_scan()` (bearish_market flag) |

In **bear** regime, the `run_full_scan()` function additionally suppresses all `buy` actions and requires confidence ≥ 0.80 for shorts.

In a **bull** regime, the scorer hard-blocks all short entries (shorts rarely work with the trend pointing up).

### VIX Gates (Beyond Position Sizing)
- VIX > 35: entire session aborts for `market_open`; `session_continuous()` skips scans
- VIX > 35 and net > 0: `_no_signal()` returns `hold` — no longs in extreme fear
- VIX > 25 and abs(confidence) < 0.80: `_no_signal()` — insufficient conviction for elevated volatility

---

## 9. Trade Execution Flow

1. **Signal generated** — `score_ticker()` returns action=`buy` with net_score ≥ 65
2. **Correlation guard** — checks if 2+ tickers from same group already held; skips if so
3. **Duplicate guard** — `_has_open_position()` checks DB + live Alpaca positions
4. **Position sizing** — `calculate_position()` returns shares count (0 if kill switch active)
5. **Quote fetch** — `get_latest_quote()` retrieves current bid/ask from Alpaca data API
6. **Limit price** — `compute_limit_price()` sets buy limit slightly above ask (or below bid for sells)
7. **Order submission** — `submit_order()` calls Alpaca broker API via `_retry()` with 12s timeout
   - In DRY_RUN mode: order_id = `"DRY_RUN"`, no real capital at risk
8. **Fill check** — `check_order_filled()` polls for up to 60 seconds
9. **DB logging** — `log_trade()` inserts full trade record including all signal metadata
10. **Rich console output** — colour-coded trade confirmation printed to terminal/CI log
11. **Intra-session** — `session_continuous()` checks stops/targets/time-exits/trailing-stops every 5 minutes
12. **EOD report** — `generate_report()` builds HTML report from DB, writes to `reports/YYYY-MM-DD.html`

---

## 10. What the Bot Does NOT Do

- **No options trading** — equity only; no calls, puts, spreads, or synthetics
- **No leverage** — position sizes are calculated to risk 2% of portfolio; no margin amplification
- **No first-5-minute trading** — `market_open` session starts at 9:35 AM, not 9:30 AM; avoids open-print volatility
- **No earnings holds** — positions are blocked entirely when earnings are within 3 days; 7-day warn reduces confidence
- **No penny stocks or OTC** — universe is curated to liquid large-caps with market cap ≥ $10B
- **No AI/LLM model calls** — all scoring is deterministic rules-based Python; no ChatGPT, Claude, or similar
- **No HFT** — minimum 5-minute scan interval; this is not a high-frequency system
- **No cross-day shorts in bull markets** — shorts are hard-blocked when SPY regime = `bull`
- **No portfolio construction optimisation** — each trade is sized independently; no Modern Portfolio Theory correlation weighting (the correlation guard is a simple cap, not an optimiser)
- **No dividends, splits, or corporate action handling** — yfinance `auto_adjust=True` handles splits but dividends are not tracked as income events

---

## 11. File Structure Reference

```
MoneyPrinter/
├── main.py                    # Entry point; session router; execute_signals(); run_full_scan()
├── watchlist.json             # Static ticker universe + company names
├── discovered_tickers.json    # Dynamically populated by discovery session
├── requirements.txt           # Python dependencies
├── SYSTEM_OVERVIEW.md         # This document
│
├── bot/
│   ├── __init__.py
│   ├── indicators.py          # get_indicators(), get_indicators_batch(), get_intraday_indicators()
│   │                          #   compute_indicators_from_df(), compute_vwap(), compute_pivot_points()
│   ├── news.py                # get_news_sentiment(), get_news_batch()
│   │                          #   keyword_amplifier(), _check_earnings_proximity()
│   │                          #   _fetch_newsapi(), _fetch_rss(), _check_sec_8k()
│   ├── scorer.py              # score_ticker(), _no_signal(), _pick_strategy_hint()
│   ├── strategies.py          # classify_strategy(), STRATEGY_CONFIGS, _classify()
│   ├── risk.py                # calculate_position(), update_trailing_stop()
│   │                          #   get_vix_multiplier(), init_daily_state(), record_trade_pnl()
│   │                          #   is_kill_switch_active()
│   ├── logger.py              # init_db(), log_trade(), update_trade_exit()
│   │                          #   update_trade_trailing(), get_open_trades(), get_trades_today()
│   │                          #   log_daily_summary(), log_scan()
│   ├── trader.py              # build_client(), build_data_client(), submit_order()
│   │                          #   get_account(), get_positions(), get_latest_quote()
│   │                          #   check_order_filled(), compute_limit_price()
│   ├── portfolio.py           # check_stops(), check_targets(), check_time_exits()
│   │                          #   get_open_positions(), close_position_and_log()
│   └── discovery.py           # run_discovery(), scan_rising_movers(), get_discovered_tickers()
│                              #   UNIVERSE list (~100 large-caps)
│
├── reports/
│   ├── __init__.py
│   ├── daily_report.py        # generate_report() → reports/YYYY-MM-DD.html
│   └── YYYY-MM-DD.html        # Generated daily HTML reports (committed by eod_summary workflow)
│
├── data/
│   └── trades.db              # SQLite database (committed by eod_summary workflow)
│
├── logs/
│   └── bot_YYYYMMDD.log       # Daily log file
│
├── dashboard/
│   └── view_trades.py         # Rich CLI dashboard for reviewing trade history
│
└── .github/workflows/
    ├── discovery.yml          # 8:30 AM EDT
    ├── premarket.yml          # 9:00 AM EDT
    ├── market_open.yml        # 9:35 AM EDT
    ├── midday.yml             # 12:00 PM EDT
    ├── market_close.yml       # 3:30 PM EDT
    ├── eod_summary.yml        # 4:15 PM EDT — also commits reports/*.html
    └── trading_day.yml        # 9:30 AM–4:00 PM EDT (continuous loop)
```

---

## 12. Performance Tracking

### SQLite Database — `data/trades.db`

**`trades` table** — one row per order placed:

| Column | Type | Description |
|---|---|---|
| id | INTEGER PK | Auto-increment row ID |
| timestamp | TEXT | UTC ISO timestamp of entry |
| session | TEXT | Which session placed the trade |
| ticker | TEXT | Equity symbol |
| action | TEXT | `buy` / `short` |
| strategy | TEXT | One of 7 strategy names |
| time_horizon | TEXT | `scalp` / `swing` / `position` |
| quantity | INTEGER | Shares traded |
| entry_price | REAL | Real-time price at entry (or actual fill) |
| limit_price | REAL | Limit price submitted to exchange |
| stop_loss | REAL | Computed stop loss level |
| take_profit | REAL | Computed take profit level |
| confidence | REAL | 0.0–1.0 directional confidence |
| net_score | INTEGER | bull − bear final score |
| bull_score | INTEGER | Accumulated bull points |
| bear_score | INTEGER | Accumulated bear points |
| signals_triggered | TEXT | JSON array of signal names |
| signals_against | TEXT | JSON array of counter-signals |
| reasoning | TEXT | Human-readable explanation |
| risk_reward | REAL | R:R ratio for this trade |
| macro_bias | TEXT | SPY regime at entry time |
| vix_level | REAL | VIX at entry time |
| alpaca_order_id | TEXT | Alpaca order UUID (or `DRY_RUN`) |
| status | TEXT | `open` / `dry_run` / `closed` / `stopped` / `target_hit` / `time_exit` / `trailing_stop` |
| exit_price | REAL | Fill price at exit |
| exit_timestamp | TEXT | UTC ISO timestamp of exit |
| pnl_dollar | REAL | Realised P&L in dollars |
| pnl_pct | REAL | Realised P&L as percentage |
| highest_price_seen | REAL | For trailing stop: highest price since entry |
| trailing_stop_price | REAL | Current trailing stop level |

**`daily_summary` table** — one row per trading day:
date, starting_value, ending_value, cash, total_trades, winning_trades, losing_trades, gross_pnl, win_rate, best_trade, worst_trade, macro_bias, vix_level, kill_switch_triggered, notes

**`scan_log` table** — one row per scan cycle:
id, timestamp, session, tickers_scanned, signals_generated, trades_executed, total_bull_signals, total_bear_signals

### `--session holdings`
Running `python main.py --session holdings` prints a Rich table of all open DB positions, enriched with live Alpaca prices if available. Shows ticker, action, quantity, entry price, current price, unrealised P&L %, stop, target, strategy, and entry timestamp.

### Daily HTML Reports
Generated by `reports/daily_report.py` at EOD. Self-contained single HTML file with inline CSS, dark mode, mobile-responsive layout. Includes: header metrics, executive summary paragraph, individual trade cards with full signal breakdowns, open portfolio snapshot with unrealised P&L, today/all-time statistics, and footer disclaimer. Reports are committed to the repo by `eod_summary.yml`.

---

## 13. Known Limitations and Future Improvements

### Current Limitations

1. **Single-process, single-day state** — The kill switch, trailing stop `highest_price_seen`, and daily P&L accumulator are in-memory variables. If the GitHub Actions runner restarts mid-session, state resets. The trailing stop state is persisted to SQLite, but in-memory risk state is not.

2. **yfinance dependency** — All market data flows through yfinance, which is an unofficial Yahoo Finance scraper. It can break without notice when Yahoo changes its API. A production system should use a paid data vendor (Polygon.io, IEX Cloud) with a stable contract.

3. **No real-time tick data** — The bot uses 5-minute intraday bars and 15-minute indicators, not tick-by-tick data. Stop losses are checked by polling, not by exchange-native stop orders. A fast intraday move (earnings gap, circuit breaker) could blow through stops before the next cycle.

4. **No partial fills or order management** — Once submitted, orders are treated as fill-or-cancel. There is no logic for partial fills, order amendments, or dynamic repricing based on market conditions.

5. **Static ATR-based stops** — Stop losses do not account for key support/resistance levels or round numbers. A more sophisticated system would combine ATR with S/R levels.

6. **NewsAPI quota** — The free tier of NewsAPI limits requests. Under heavy scanning with many tickers, the bot may exhaust its daily quota, causing news signals to fall back to 0.0 polarity (neutral).

7. **No overnight gap protection** — Positions held overnight are exposed to gap-down risk. The bot does not use options collars or GTC stop orders that would fire on a gap open.

8. **Backtest does not include slippage or commission** — The walk-forward backtest assumes perfect fills at the entry bar open price with zero commission. Real results will be worse due to slippage, spread, and per-trade fees.

### Planned Improvements

- **Real-time stop orders via Alpaca** — Submit exchange-native stop orders instead of polling-based exits to eliminate the gap-risk window between scans
- **Position-level P&L from Alpaca** — Replace DB-estimated unrealised P&L with live values from `get_positions()` for more accurate portfolio snapshot
- **Options-based tail-risk hedging** — Buy protective puts on high-conviction swing trades near earnings season
- **Sector rotation overlay** — Track relative strength of 11 GICS sectors; tilt exposure toward outperforming sectors
- **Machine-learning signal weighting** — Use historical signal → outcome data in the DB to learn optimal weights for each of the 30+ signals, replacing the hand-tuned point values
- **Multi-broker support** — Abstract the trader layer to support Interactive Brokers and other APIs beyond Alpaca
- **Telegram/Slack notifications** — Push real-time trade alerts and EOD summaries to a messaging channel rather than relying solely on CI logs and HTML reports
