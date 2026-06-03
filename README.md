# MoneyPrinter — Autonomous Algorithmic Trading Bot

A fully autonomous, rules-based algorithmic trading bot that:
- Analyzes 20+ technical indicators and news sentiment per ticker
- Executes trades on **Alpaca Paper Trading** (zero-cost, no real money risk)
- Runs entirely via **GitHub Actions** on a cron schedule — no server required
- Uses **zero external AI APIs** — all decision logic is pure Python

---

## How It Works

### Scoring Engine
Every ticker is scored on a **bull/bear point system** across five signal categories:

| Category | Indicators |
|---|---|
| Trend | EMA alignment, ADX, MACD, Parabolic SAR, VWAP |
| Momentum | RSI, Stochastic RSI, CCI, Williams %R, Rate of Change |
| Volatility | Bollinger Bands, Keltner Channels, ATR |
| Volume | OBV, Volume Ratio, MFI |
| News | TextBlob sentiment, SEC 8-K detection |

**Net Score** = Bull Score − Bear Score  
**Action thresholds:**
- Net > 30 → `buy`
- Net < −30 → `short`/`sell`
- Otherwise → `hold`

**No trade is executed unless:** `|net_score| ≥ 30` AND `confidence ≥ 0.60`

### Macro Filter
Before scoring any ticker, the bot computes:
- **VIX level** — scales position sizes down as fear rises, halts new longs at VIX > 35
- **SPY regime** — discounts bull signals 20–40% when SPY is in caution/bear zone

### Strategies
| Strategy | Trigger | Time Horizon |
|---|---|---|
| `trend_follow` | EMA aligned + ADX>25 + MACD rising + volume | swing (3–10d) |
| `mean_reversion` | RSI oversold + BB squeeze + Stoch RSI turning | scalp (1–5d) |
| `breakout` | Price breaks R1/52wk high with volume >1.5x | swing (5–20d) |
| `breakdown` | Price breaks S1/52wk low with volume | swing |
| `squeeze_breakout` | BB squeeze resolved + KC breakout | swing |
| `news_momentum` | Sentiment >0.4 + trend confirmation | scalp (1–3d) |

### Risk Management
- Base risk per trade: **2% of portfolio**, scaled by confidence × VIX multiplier
- Maximum position size: **10% of portfolio** in any single stock
- Stop loss: **1.5× ATR** below/above entry (strategy-specific)
- Take profit: **2.5× the stop distance** (configurable by strategy)
- **Kill switch**: If daily P&L falls below −3% of starting value, all new orders halt

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/moneyprinter.git
cd moneyprinter
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"
```

### 3. Configure environment
```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Get API Keys

**Alpaca (Paper Trading — Free)**
1. Go to [alpaca.markets](https://alpaca.markets) → Create account
2. Switch to **Paper Trading** environment
3. Go to **API Keys** → Generate new key
4. Copy `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`

**NewsAPI (Free tier: 100 requests/day)**
1. Go to [newsapi.org](https://newsapi.org) → Get API Key
2. Copy key to `NEWS_API_KEY`

---

## Running Locally (Test Mode)

```bash
# Dry run — full logic, no real orders
DRY_RUN=true python main.py --session premarket
DRY_RUN=true python main.py --session market_open
DRY_RUN=true python main.py --session midday
DRY_RUN=true python main.py --session market_close
DRY_RUN=true python main.py --session eod_summary

# View the trade dashboard
python dashboard/view_trades.py
```

---

## GitHub Actions Setup

All 5 workflows run automatically on their cron schedules:

| Workflow | Cron (UTC) | EDT Local | Action |
|---|---|---|---|
| `premarket.yml` | `0 13 * * 1-5` | 9:00 AM | News scan, gap detection |
| `market_open.yml` | `5 14 * * 1-5` | 9:35 AM | Full score + trade execution |
| `midday.yml` | `0 16 * * 1-5` | 12:00 PM | Stop/target checks |
| `market_close.yml` | `30 19 * * 1-5` | 3:30 PM | Close scalps, overnight decisions |
| `eod_summary.yml` | `15 20 * * 1-5` | 4:15 PM | P&L report |

### Add secrets to GitHub
Go to your repo → **Settings → Secrets and variables → Actions**

Add these **Repository Secrets:**
- `ALPACA_API_KEY`
- `ALPACA_SECRET_KEY`
- `NEWS_API_KEY`

To control live vs dry run, add a **Repository Variable:**
- `DRY_RUN` = `true` (safe default) or `false` (live paper trading)

### Enable write permissions for Actions
Go to **Settings → Actions → General → Workflow permissions**  
Select: **Read and write permissions** (so the bot can commit `trades.db`)

---

## EDT vs EST (Daylight Saving Adjustment)

The cron times above use **EDT (UTC−4)**, valid **March–November**.

During **November–March (EST = UTC−5)**, add 1 hour to all UTC times:

| Workflow | EDT cron | EST cron |
|---|---|---|
| premarket | `0 13 * * 1-5` | `0 14 * * 1-5` |
| market_open | `5 14 * * 1-5` | `5 15 * * 1-5` |
| midday | `0 16 * * 1-5` | `0 17 * * 1-5` |
| market_close | `30 19 * * 1-5` | `30 20 * * 1-5` |
| eod_summary | `15 20 * * 1-5` | `15 21 * * 1-5` |

---

## Customizing the Watchlist

Edit `watchlist.json` to add or remove tickers:

```json
{
  "trade": {
    "tech": ["AAPL", "MSFT", "NVDA", "YOUR_TICKER"],
    ...
  },
  "company_names": {
    "YOUR_TICKER": "Your Company Inc"
  }
}
```

The `company_names` mapping is used for better news search accuracy.

---

## Trade Dashboard

```bash
python dashboard/view_trades.py
```

Displays:
1. Portfolio value, cash, daily P&L
2. Open positions table (color-coded: green = profit, red = loss, yellow = near stop)
3. Today's closed trades with WIN/LOSS labels
4. All-time stats: win rate, profit factor, avg winner/loser
5. 7-day P&L bar chart
6. Last scan summary

---

## Project Structure

```
MoneyPrinter/
├── .github/workflows/       # 5 GitHub Actions (one per session)
├── bot/
│   ├── indicators.py        # All technical indicator calculations
│   ├── news.py              # News fetching + sentiment scoring
│   ├── scorer.py            # Rules-based decision engine
│   ├── strategies.py        # Strategy classifier
│   ├── risk.py              # Position sizing + kill switch
│   ├── trader.py            # Alpaca order execution
│   ├── portfolio.py         # Portfolio/position state
│   └── logger.py            # SQLite trade logger
├── dashboard/
│   └── view_trades.py       # Rich CLI dashboard
├── data/
│   └── trades.db            # SQLite DB (auto-committed by Actions)
├── watchlist.json
├── main.py                  # Session router + all session logic
└── requirements.txt
```

---

## Migrating to Robinhood (robin_stocks)

The bot is designed so only `bot/trader.py` interacts with the broker API.
To switch to Robinhood:

1. `pip install robin_stocks`
2. Replace `bot/trader.py` with a Robinhood-backed implementation exposing the same functions:
   - `get_account()` → `robin_stocks.robinhood.account.load_portfolio_profile()`
   - `submit_order()` → `robin_stocks.robinhood.orders.order_buy_limit()`
   - `close_position()` → `robin_stocks.robinhood.orders.order_sell_market()`
   - etc.

Everything else — indicators, scorer, strategies, risk, logger, dashboard — remains unchanged.

---

## Cost

| Service | Cost |
|---|---|
| Alpaca Paper Trading | Free |
| NewsAPI | Free (100 req/day) |
| yfinance | Free |
| GitHub Actions | Free (public repos) / ~2000 min/month free (private) |
| Total | **$0/month** |

---

## Live Dashboard

A real-time web dashboard shows every decision the bot makes — buys, holds, shorts, Claude's reasoning — updated live from the repo.

### Deploy to Vercel (free, ~2 minutes)

1. Go to [vercel.com](https://vercel.com) and sign in with GitHub
2. Click **Add New → Project** and import `Harsh-0214/MoneyPrinter`
3. Under **Root Directory**, click **Edit** and set it to `vercel-dashboard`
4. Leave everything else as default and click **Deploy**

That's it. Vercel auto-redeploys every time the bot commits new data to `main`.

### How it works

- The bot writes every decision (buy, hold, short, AI verdict, reasoning) to `data/live_feed.json` after each scan cycle
- GitHub Actions commits and pushes that file to `main` at the end of every session
- The dashboard fetches `raw.githubusercontent.com/.../data/live_feed.json` every 30 seconds with a cache-bust param
- No server, no database — just a static HTML file reading a JSON file from GitHub

### What you see

| Column | Description |
|---|---|
| Time | UTC timestamp of the scan cycle |
| Ticker | Stock symbol |
| Action | **BUY** (green) / **SHORT** (red) / **HOLD** (gray) |
| Net Score | Bull minus bear points from the rules engine |
| Conf % | Confidence (net score / 100) |
| Strategy | Detected strategy pattern |
| AI Verdict | ✓ Claude confirmed · ✗ Claude rejected · — not evaluated |
| AI Reasoning | Claude's one-sentence justification (hover for full text) |

Summary cards show: total decisions today, buys, holds, shorts, Claude overrides, and rejections.

---

## Disclaimer

**This bot executes paper trades only.** It is not connected to any real brokerage account by default (`ALPACA_BASE_URL=https://paper-api.alpaca.markets`).

This software is provided for educational and research purposes only. It is **not financial advice**. Past performance of any algorithm does not guarantee future results. Never trade with money you cannot afford to lose.

**The authors accept no liability for any trading losses.**
