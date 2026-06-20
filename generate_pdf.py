"""Generate a simple overview PDF for MoneyPrinter."""

from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT

OUTPUT = "MoneyPrinter_Overview.pdf"

doc = SimpleDocTemplate(
    OUTPUT,
    pagesize=letter,
    rightMargin=0.8 * inch,
    leftMargin=0.8 * inch,
    topMargin=0.8 * inch,
    bottomMargin=0.8 * inch,
)

styles = getSampleStyleSheet()

title_style = ParagraphStyle(
    "Title",
    parent=styles["Normal"],
    fontSize=22,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#1a1a2e"),
    spaceAfter=4,
    alignment=TA_CENTER,
)
subtitle_style = ParagraphStyle(
    "Subtitle",
    parent=styles["Normal"],
    fontSize=11,
    fontName="Helvetica",
    textColor=colors.HexColor("#555555"),
    spaceAfter=16,
    alignment=TA_CENTER,
)
h2_style = ParagraphStyle(
    "H2",
    parent=styles["Normal"],
    fontSize=13,
    fontName="Helvetica-Bold",
    textColor=colors.HexColor("#16213e"),
    spaceBefore=14,
    spaceAfter=4,
)
body_style = ParagraphStyle(
    "Body",
    parent=styles["Normal"],
    fontSize=9.5,
    fontName="Helvetica",
    textColor=colors.HexColor("#222222"),
    leading=14,
    spaceAfter=6,
)
bullet_style = ParagraphStyle(
    "Bullet",
    parent=body_style,
    leftIndent=14,
    bulletIndent=4,
    spaceAfter=3,
)
code_style = ParagraphStyle(
    "Code",
    parent=styles["Normal"],
    fontSize=8.5,
    fontName="Courier",
    textColor=colors.HexColor("#2d2d2d"),
    backColor=colors.HexColor("#f4f4f4"),
    leftIndent=10,
    rightIndent=10,
    borderPad=4,
    leading=12,
    spaceAfter=6,
)

def h2(text):
    return Paragraph(text, h2_style)

def body(text):
    return Paragraph(text, body_style)

def bullet(text):
    return Paragraph(f"• {text}", bullet_style)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#cccccc"), spaceAfter=6)

def table(data, col_widths=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16213e")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f8f8")]),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


story = []

# ── Title ─────────────────────────────────────────────────────────────────────
story.append(Spacer(1, 0.1 * inch))
story.append(Paragraph("MoneyPrinter", title_style))
story.append(Paragraph("Autonomous Algorithmic Trading Bot — Simple Overview", subtitle_style))
story.append(hr())

# ── What Is It? ───────────────────────────────────────────────────────────────
story.append(h2("What Is It?"))
story.append(body(
    "MoneyPrinter is a fully automated stock trading bot that runs on GitHub Actions "
    "at no server cost. It scans a universe of ~150 large/mid-cap stocks every trading day, "
    "scores them with 30+ technical indicators, optionally validates high-confidence picks "
    "with Claude AI, and places bracket orders on Alpaca Paper Trading. "
    "Everything is free by default — no live money, no server."
))

# ── How a Trade Happens ───────────────────────────────────────────────────────
story.append(h2("How a Trade Happens (Step by Step)"))
steps = [
    ("1. Discovery", "08:30 AM ET", "Scans ~150 stocks for unusual volume or big price moves. Top 10 movers saved to discovered_tickers.json."),
    ("2. Pre-Market", "09:00 AM ET", "Loads the discovered tickers and checks news headlines + gap analysis."),
    ("3. Score", "9:30 AM+", "Each ticker runs through 30+ technical indicators across 5 categories (Trend, Momentum, Volatility, Volume, News). Result: a net score and a confidence number."),
    ("4. Gate Checks", "9:30 AM+", "Signal must pass: kill-switch check, max-trades cap, stop-width limit, regime filter, sector cap, portfolio exposure cap, and a 90-second freshness check."),
    ("5. Claude AI", "9:30 AM+", "If USE_CLAUDE=true, Claude reviews every signal and provides entry price, stop loss, take profit, and a risk/reward ratio."),
    ("6. Order", "9:30 AM+", "A bracket order is placed on Alpaca: limit entry + stop loss + take profit. Position size is 2% of portfolio, scaled by confidence and VIX."),
    ("7. Monitor", "Ongoing", "Open positions are checked every scan cycle for stop/target breach, time-based exits, and trailing stops."),
    ("8. EOD Summary", "4:15 PM ET", "Closed trades are reconciled, P&L logged to the database, and a daily summary written."),
]
col_w = [1.1*inch, 1.0*inch, 4.7*inch]
tbl_data = [["Step", "When", "What Happens"]] + [[s[0], s[1], s[2]] for s in steps]
story.append(table(tbl_data, col_widths=col_w))

# ── Scoring ───────────────────────────────────────────────────────────────────
story.append(h2("Scoring Engine"))
story.append(body(
    "Every ticker earns bull (+) and bear (−) points. "
    "<b>Net Score = Bull − Bear</b>. "
    "<b>Confidence = Net ÷ 150</b> (capped at 1.0)."
))
score_data = [
    ["Category", "Indicators"],
    ["Trend", "EMA 9/21/50/200 alignment, ADX, MACD, Parabolic SAR, VWAP"],
    ["Momentum", "RSI, Stochastic RSI, CCI, Williams %R"],
    ["Volatility", "Bollinger Bands (%B), Keltner Channels, ATR, BB squeeze"],
    ["Volume", "OBV, Volume Ratio, MFI"],
    ["News", "yfinance headlines, TextBlob sentiment, hype/boost keywords"],
    ["Fundamentals", "Revenue growth, EPS beats, institutional ownership, short interest"],
]
story.append(table(score_data, col_widths=[1.4*inch, 5.4*inch]))
story.append(Spacer(1, 6))
story.append(body("<b>Trade signal thresholds:</b>"))
story.append(bullet("BUY: Net ≥ 65 AND Confidence ≥ 0.47"))
story.append(bullet("SHORT: Net ≥ 70 AND Confidence ≥ 0.70"))
story.append(bullet("HOLD: Everything else"))

# ── Risk Management ───────────────────────────────────────────────────────────
story.append(h2("Risk Management"))
risk_data = [
    ["Rule", "Value"],
    ["Base risk per trade", "2% of portfolio"],
    ["Max position size", "10% of portfolio"],
    ["Max total exposure", "60% of portfolio"],
    ["Max new entries per session", "2"],
    ["Max sector positions", "2 per sector"],
    ["Stop loss", "1.5× ATR below entry"],
    ["Take profit", "3.75× ATR above entry (~2.5× R/R)"],
    ["Kill switch", "Halt all orders if daily P&L < −3%"],
    ["Stop width cap", "Block order if stop > 8% from entry"],
    ["VIX scaling", "Position size reduced as VIX rises; new longs halted above VIX 35"],
]
story.append(table(risk_data, col_widths=[2.8*inch, 4.0*inch]))

# ── Strategies ────────────────────────────────────────────────────────────────
story.append(h2("Strategies"))
strat_data = [
    ["Strategy", "Trigger", "Hold Period"],
    ["trend_follow", "Full EMA stack aligned + ADX > 22 + MACD rising", "2–10 days"],
    ["breakout", "Price breaks R1 or 52-week high on volume", "2–10 days"],
    ["squeeze_breakout", "BB squeeze resolves + Keltner breakout", "2–10 days"],
    ["breakdown", "Price breaks S1 or 52-week low (short)", "2–10 days"],
    ["mean_reversion", "RSI/BB/CCI deeply oversold — bounce expected", "Same day to 2 days"],
    ["news_momentum", "Strong news catalyst + EMA alignment or volume surge", "Same day to 2 days"],
]
story.append(table(strat_data, col_widths=[1.4*inch, 3.6*inch, 1.8*inch]))

# ── GitHub Actions Schedule ───────────────────────────────────────────────────
story.append(h2("Daily Schedule (GitHub Actions)"))
sched_data = [
    ["Workflow", "Time (EDT)", "What It Does"],
    ["discovery.yml", "8:30 AM", "Screens ~150 stocks for active movers, saves top 10"],
    ["premarket.yml", "9:00 AM", "News and gap analysis on discovered tickers"],
    ["trading_day.yml", "9:30 AM", "Continuous scan + score + execute loop all day"],
    ["eod_summary.yml", "4:15 PM", "Reconcile positions, log daily P&L, write summary"],
    ["test_ai_filter.yml", "Manual", "Test Claude filter on any tickers via workflow_dispatch"],
]
story.append(table(sched_data, col_widths=[1.6*inch, 1.1*inch, 4.1*inch]))
story.append(body(
    "Workflows run on GitHub's free tier — no server or cloud account needed. "
    "The bot determines if the market is open by querying the Alpaca calendar API, "
    "so it automatically handles holidays, early closes, and edge cases like Juneteenth."
))

# ── Data Flow ─────────────────────────────────────────────────────────────────
story.append(h2("Data Flow"))
story.append(body("<b>Inputs:</b>"))
story.append(bullet("Alpaca: live prices, portfolio positions, order placement"))
story.append(bullet("yfinance: 2 years of OHLCV bars for indicator calculation"))
story.append(bullet("yfinance / NewsAPI: latest headlines for sentiment scoring"))
story.append(body("<b>Outputs:</b>"))
story.append(bullet("data/live_feed.json — decision feed for the Vercel dashboard (updated every scan)"))
story.append(bullet("data/trades.db — SQLite trade log (or Turso if TURSO_DATABASE_URL is set)"))
story.append(bullet("GitHub Actions logs — full indicator + reasoning trace per scan"))
story.append(bullet("discovered_tickers.json — top movers from each discovery session"))

# ── Dashboard ─────────────────────────────────────────────────────────────────
story.append(h2("Live Dashboard (Vercel)"))
story.append(body(
    "A single-file HTML dashboard lives in vercel-dashboard/index.html. "
    "Deploy it to Vercel for free in ~2 minutes (connect repo → set Root Directory to vercel-dashboard → Deploy). "
    "It auto-updates every 30 seconds by reading live_feed.json from GitHub."
))
story.append(body("Three views:"))
story.append(bullet("Portfolio — open positions with unrealized P&L and hold time"))
story.append(bullet("Trades — closed trade history with entry/exit prices and realized P&L"))
story.append(bullet("Bot Status — real-time decision feed for every ticker scanned"))

# ── Setup ─────────────────────────────────────────────────────────────────────
story.append(h2("Setup in 5 Minutes"))
story.append(body("1. Fork or clone the repo to your GitHub account."))
story.append(body("2. Go to Settings → Secrets → Actions and add:"))
secrets_data = [
    ["Secret", "Where to Get It"],
    ["ALPACA_API_KEY", "alpaca.markets → Paper Trading → API Keys"],
    ["ALPACA_SECRET_KEY", "Same page"],
    ["ANTHROPIC_API_KEY", "console.anthropic.com (optional — only needed if USE_CLAUDE=true)"],
    ["NEWS_API_KEY", "newsapi.org — free tier, optional"],
    ["TURSO_DATABASE_URL", "turso.tech — optional, for persistent remote trade history"],
    ["TURSO_AUTH_TOKEN", "Same as above"],
]
story.append(table(secrets_data, col_widths=[2.2*inch, 4.6*inch]))
story.append(body("3. Enable write permissions: Settings → Actions → General → Workflow permissions → Read and write."))
story.append(body("4. Push any change to main — the workflows will start running automatically on the next market day."))

# ── Cost ──────────────────────────────────────────────────────────────────────
story.append(h2("Cost"))
cost_data = [
    ["Service", "Cost"],
    ["Alpaca Paper Trading", "Free"],
    ["GitHub Actions", "Free (public repos)"],
    ["yfinance data", "Free"],
    ["Vercel dashboard", "Free"],
    ["NewsAPI (optional)", "Free (100 req/day)"],
    ["Claude AI (optional)", "~$0.01–0.05 per scan if USE_CLAUDE=true"],
    ["Turso DB (optional)", "Free tier covers trading volume easily"],
    ["Total", "~$0 to $1/month"],
]
story.append(table(cost_data, col_widths=[2.8*inch, 4.0*inch]))

# ── Disclaimer ────────────────────────────────────────────────────────────────
story.append(Spacer(1, 12))
story.append(hr())
story.append(Paragraph(
    "<b>Disclaimer:</b> This bot trades on Alpaca Paper Trading by default — no real money. "
    "It is provided for educational and research purposes only and is not financial advice. "
    "Never trade real money without fully understanding the risks.",
    ParagraphStyle("Disclaimer", parent=body_style, textColor=colors.HexColor("#888888"), fontSize=8.5)
))

doc.build(story)
print(f"PDF written to {OUTPUT}")
