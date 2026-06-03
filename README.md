# MoneyPrinter — Autonomous Algorithmic Trading Bot

A fully autonomous algorithmic trading bot that scores stocks with 30+ technical indicators, validates every trade with Claude AI, and executes orders on Alpaca Paper Trading — all running on a free GitHub Actions cron schedule with no server required.

---

## How It Works

### 1. Rules-Based Scoring Engine
Every ticker is scored on a **bull/bear point system** across five signal categories:

| Category | Indicators |
|---|---|
| Trend | EMA alignment (9/21/50/200), ADX, MACD, Parabolic SAR, VWAP |
| Momentum | RSI, Stochastic RSI, CCI, Williams %R |
| Volatility | Bollinger Bands (%B), Keltner Channels, ATR, BB squeeze |
| Volume | OBV, Volume Ratio, MFI |
| News | yfinance headlines, TextBlob sentiment, keyword boosts |

**Net Score** = Bull Score − Bear Score  
**Confidence** = Net Score ÷ 100

**Action thresholds:**
- Net ≥ 65 AND confidence ≥ 0.65 → `buy`
- Net ≥ 70 AND confidence ≥ 0.70 → `short`
- Otherwise → `hold`

### 2. Macro Filter
Before scoring any ticker the bot checks:
- **VIX** — scales position sizes down as fear rises, halts new longs above VIX 35
- **SPY regime** — discounts bull signals 20–50% when SPY is in caution/bear territory

### 3. Fundamental Quality Filter
Every ticker is scored on fundamentals via yfinance (cached per session):
- Revenue growth, EPS beat history, institutional ownership, short interest
- Adds/subtracts bull/bear points and classifies breakout quality

### 4. Multi-Timeframe Velocity System
Returns over 1d / 5d / 1m / 3m are computed and compared to thresholds:
- Large recent gains trigger a **velocity penalty** that reduces confidence
- Penalty is halved for `fundamental` breakouts, doubled for `hype` breakouts
- Hard cap of 0.45 penalty so no stock is completely zeroed out

### 5. Hype Detection
News headlines are scanned for:
- **Penalties**: Jim Cramer mentions, retail FOMO language, Reddit/WSB references, short-squeeze narratives
- **Boosts**: Earnings beats, raised guidance, insider buying, analyst upgrades with price targets

### 6. Claude AI Confirmation
Every scored ticker (buy, hold, or short) is sent to **Claude claude-sonnet-4-6** with:
- All 30+ indicators and their values
- Bull/bear signals triggered
- Multi-timeframe returns and velocity data
- News headlines and sentiment
- Current held position context (if stock is already owned)

Claude returns a structured decision with `entry_price`, `stop_loss`, `take_profit`, `risk_reward`, and `entry_condition` for every ticker — including holds.

### 7. Position Awareness
At the start of each scan the bot fetches live Alpaca positions. For stocks already held:
- Claude's valid decisions are `hold`, `add`, or `sell`
- A SELL signal closes the full position via Alpaca
- The open position (qty, avg entry, unrealized P&L) is shown in the dashboard

### 8. Risk Management
- Base risk per trade: **2% of portfolio**, scaled by confidence × VIX multiplier
- Maximum position size: **10% of portfolio** per stock
- Stop loss: **1.5× ATR** below entry
- Take profit: **3.75× ATR** above entry (≈2.5× risk/reward)
- **Kill switch**: If daily P&L falls below −3% of starting value, all new orders halt
- Hard block on stocks with intraday move > 15%; raised threshold if move > 10%

---

## Strategies

| Strategy | Trigger | Time Horizon |
|---|---|---|
| `trend_follow` | Full EMA alignment + ADX > 22 + MACD rising + volume confirmation | Swing — 2–10 days |
| `breakout` | Price breaks R1 resistance or 52-week high with volume | Swing — 2–10 days |
| `squeeze_breakout` | Bollinger Band squeeze resolved + Keltner Channel breakout | Swing — 2–10 days |
| `breakdown` | Price breaks S1 support or 52-week low with volume (short side) | Swing — 2–10 days |
| `mean_reversion` | RSI/BB/CCI deeply oversold — bounce back to mean | Scalp — same day to 2 days |
| `news_momentum` | Positive news catalyst + EMA alignment or volume surge | Scalp — same day to 2 days |

---

## GitHub Actions Schedule

Runs automatically Monday–Friday, no server needed:

| Workflow | UTC | EDT | Action |
|---|---|---|---|
| `discovery.yml` | `30 12` | 8:30 AM | Pre-market scan, identify setups |
| `premarket.yml` | `0 13` | 9:00 AM | News and gap analysis |
| `trading_day.yml` | `30 13` | 9:30 AM | Full continuous session — scores, Claude, execute (runs all day) |
| `eod_summary.yml` | `15 20` | 4:15 PM | End-of-day P&L summary |
| `test_ai_filter.yml` | Manual | — | Test specific tickers via workflow_dispatch |

---

## Live Dashboard

A real-time web dashboard shows every decision — buy, hold, short, full indicator breakdown, Claude's reasoning, and news headlines.

### Deploy to Vercel (free, ~2 minutes)

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click **Add New → Project** and import this repo
3. Under **Root Directory**, set it to `vercel-dashboard`
4. Click **Deploy**

Vercel auto-redeploys every time the bot commits new data to `main`.

### How it works

- After each scan cycle the bot writes every decision to `data/live_feed.json`
- GitHub Actions commits and pushes that file to `main`
- The dashboard fetches the raw JSON from GitHub every 30 seconds
- No server, no database — just a static HTML file reading a JSON file

### What's in the analysis panel (click any row)

- **Technical Indicators** — RSI, ADX, BB%B, Volume Ratio, MFI, StochRSI, MACD, CCI, Williams %R, ATR, VWAP, full EMA stack, 52-week range, distance from EMA200, VIX, SPY regime
- **Rule-Based Signals** — all bull/bear signals triggered, scorer reasoning
- **Claude's Analysis** — entry price, stop loss, take profit, risk/reward, entry condition, full Claude reasoning, news headlines with sentiment scores

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/Harsh-0214/MoneyPrinter.git
cd MoneyPrinter
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add GitHub Secrets
Go to **Settings → Secrets and variables → Actions** and add:

| Secret | Where to get it |
|---|---|
| `ALPACA_API_KEY` | [alpaca.markets](https://alpaca.markets) → Paper Trading → API Keys |
| `ALPACA_SECRET_KEY` | Same as above |
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) |
| `NEWS_API_KEY` | [newsapi.org](https://newsapi.org) (optional — yfinance is primary) |

### 4. Enable write permissions for Actions
Go to **Settings → Actions → General → Workflow permissions**  
Select: **Read and write permissions**

### 5. Running locally
```bash
# Test a specific session
python main.py --session discovery
python main.py --session premarket
python main.py --session continuous
python main.py --session eod_summary

# Test Claude AI filter on specific tickers
python main.py --session test_ai --tickers NVDA,AAPL,MSFT
```

---

## Project Structure

```
MoneyPrinter/
├── .github/workflows/
│   ├── discovery.yml          # 8:30 AM EDT
│   ├── premarket.yml          # 9:00 AM EDT
│   ├── trading_day.yml        # 9:30 AM EDT (continuous)
│   ├── eod_summary.yml        # 4:15 PM EDT
│   └── test_ai_filter.yml     # Manual trigger
├── bot/
│   ├── indicators.py          # 30+ technical indicator calculations (2yr history)
│   ├── news.py                # News fetching (yfinance primary) + sentiment + hype detection
│   ├── scorer.py              # Rules engine, velocity system, fundamental quality filter
│   ├── ai_filter.py           # Claude AI confirmation + price guidance
│   ├── risk.py                # Position sizing, VIX scaling, kill switch
│   ├── trader.py              # Alpaca order execution
│   ├── live_feed.py           # Writes data/live_feed.json for dashboard
│   └── logger.py              # SQLite trade logger (data/trades.db)
├── vercel-dashboard/
│   └── index.html             # Full-featured responsive web dashboard
├── data/
│   ├── trades.db              # SQLite DB (auto-committed by Actions)
│   └── live_feed.json         # Live decision feed (auto-committed by Actions)
├── watchlist.json             # Tickers to scan + company name mappings
├── main.py                    # Session router + execution logic
└── requirements.txt
```

---

## Customizing the Watchlist

Edit `watchlist.json` to add or remove tickers:

```json
{
  "trade": {
    "tech": ["AAPL", "MSFT", "NVDA", "YOUR_TICKER"],
    "finance": ["JPM", "GS"]
  },
  "company_names": {
    "YOUR_TICKER": "Your Company Inc"
  }
}
```

The `company_names` mapping improves news search accuracy.

---

## Cost

| Service | Cost |
|---|---|
| Alpaca Paper Trading | Free |
| Anthropic API (Claude) | ~$0.01–0.05 per full scan (claude-sonnet-4-6) |
| yfinance | Free |
| NewsAPI | Free (100 req/day, optional) |
| GitHub Actions | Free (public repos) |
| Vercel Dashboard | Free |
| **Total** | **~$0–$1/month** |

---

## Disclaimer

**This bot executes paper trades only.** It is not connected to any real brokerage account by default (`ALPACA_BASE_URL=https://paper-api.alpaca.markets`).

This software is provided for educational and research purposes only. It is **not financial advice**. Past performance of any algorithm does not guarantee future results. Never trade with money you cannot afford to lose.

**The authors accept no liability for any trading losses.**
