"""Rich CLI dashboard — standalone trade viewer."""

import json
import sys
from datetime import datetime
from pathlib import Path

# Allow running from repo root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel   import Panel
from rich.table   import Table
from rich.text    import Text

from bot.logger import (
    get_all_trades,
    get_daily_summaries,
    get_last_scan,
    get_open_trades,
    get_trades_today,
    init_db,
)

console = Console()


def _fmt_pnl(val) -> Text:
    if val is None:
        return Text("—", style="dim")
    v = float(val)
    style = "green" if v >= 0 else "red"
    return Text(f"${v:,.2f}", style=style)


def _fmt_pct(val) -> Text:
    if val is None:
        return Text("—", style="dim")
    v = float(val) * 100
    style = "green" if v >= 0 else "red"
    return Text(f"{v:.2f}%", style=style)


def render_header(portfolio_value: float = 0, cash: float = 0, daily_pnl: float = 0) -> None:
    pnl_color = "green" if daily_pnl >= 0 else "red"
    console.print(Panel(
        f"[bold]Portfolio Value:[/bold] ${portfolio_value:,.2f}   "
        f"[bold]Cash:[/bold] ${cash:,.2f}   "
        f"[bold]Daily P&L:[/bold] [{pnl_color}]${daily_pnl:,.2f}[/{pnl_color}]   "
        f"[bold]As of:[/bold] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        title="[bold cyan]Trading Bot Dashboard[/bold cyan]",
        border_style="cyan",
    ))


def render_open_positions(open_trades: list) -> None:
    table = Table(title="[bold]Open Positions[/bold]", show_lines=True)
    table.add_column("Ticker",      style="bold")
    table.add_column("Strategy")
    table.add_column("Horizon")
    table.add_column("Entry",       justify="right")
    table.add_column("Current",     justify="right")
    table.add_column("Shares",      justify="right")
    table.add_column("Unreal P&L",  justify="right")
    table.add_column("Unreal %",    justify="right")
    table.add_column("Stop",        justify="right")
    table.add_column("Target",      justify="right")

    for t in open_trades:
        entry   = float(t.get("entry_price") or 0)
        current = float(t.get("current_price") or entry)
        shares  = int(t.get("quantity") or 0)
        stop    = t.get("stop_loss")
        target  = t.get("take_profit")
        action  = t.get("action", "buy")

        if action == "buy":
            unreal_pnl = (current - entry) * shares
        else:
            unreal_pnl = (entry - current) * shares
        unreal_pct = unreal_pnl / (entry * shares) if (entry * shares) else 0

        # Row color
        near_stop = stop and action == "buy" and (current - float(stop)) / current < 0.10
        row_style = "green" if unreal_pnl > 0 else ("yellow" if near_stop else "red")

        table.add_row(
            t.get("ticker", ""),
            t.get("strategy", ""),
            t.get("time_horizon", ""),
            f"${entry:.2f}",
            f"${current:.2f}",
            str(shares),
            _fmt_pnl(unreal_pnl),
            _fmt_pct(unreal_pct),
            f"${float(stop):.2f}" if stop else "—",
            f"${float(target):.2f}" if target else "—",
            style=row_style,
        )

    if not open_trades:
        table.add_row(*["—"] * 10)
    console.print(table)


def render_closed_trades(trades: list) -> None:
    closed = [t for t in trades if t.get("pnl_dollar") is not None]
    table = Table(title="[bold]Today's Closed Trades[/bold]", show_lines=True)
    table.add_column("Ticker",    style="bold")
    table.add_column("Action")
    table.add_column("Entry",     justify="right")
    table.add_column("Exit",      justify="right")
    table.add_column("P&L $",     justify="right")
    table.add_column("P&L %",     justify="right")
    table.add_column("Strategy")
    table.add_column("Result")

    for t in closed:
        pnl = float(t.get("pnl_dollar") or 0)
        win = pnl > 0
        result_text = Text("WIN", style="bold green") if win else Text("LOSS", style="bold red")
        table.add_row(
            t.get("ticker", ""),
            t.get("action", ""),
            f"${float(t.get('entry_price') or 0):.2f}",
            f"${float(t.get('exit_price') or 0):.2f}",
            _fmt_pnl(pnl),
            _fmt_pct(t.get("pnl_pct")),
            t.get("strategy", ""),
            result_text,
        )

    if not closed:
        table.add_row(*["—"] * 8)
    console.print(table)


def render_all_time_stats(all_trades: list) -> None:
    closed = [t for t in all_trades if t.get("pnl_dollar") is not None]
    if not closed:
        console.print(Panel("[dim]No closed trades yet[/dim]", title="All-Time Stats"))
        return

    total    = len(closed)
    winners  = [t for t in closed if (t.get("pnl_dollar") or 0) > 0]
    losers   = [t for t in closed if (t.get("pnl_dollar") or 0) <= 0]
    win_rate = len(winners) / total if total else 0
    total_pnl = sum(float(t.get("pnl_dollar") or 0) for t in closed)
    avg_win   = sum(float(t.get("pnl_dollar") or 0) for t in winners) / len(winners) if winners else 0
    avg_loss  = sum(float(t.get("pnl_dollar") or 0) for t in losers)  / len(losers)  if losers  else 0
    gross_win  = sum(float(t.get("pnl_dollar") or 0) for t in winners)
    gross_loss = abs(sum(float(t.get("pnl_dollar") or 0) for t in losers))
    profit_factor = gross_win / gross_loss if gross_loss > 0 else float("inf")

    pnl_style = "green" if total_pnl >= 0 else "red"
    console.print(Panel(
        f"[bold]Total Trades:[/bold]    {total}\n"
        f"[bold]Win Rate:[/bold]        {win_rate*100:.1f}% ({len(winners)}W / {len(losers)}L)\n"
        f"[bold]Total P&L:[/bold]       [{pnl_style}]${total_pnl:,.2f}[/{pnl_style}]\n"
        f"[bold]Avg Winner:[/bold]      ${avg_win:,.2f}\n"
        f"[bold]Avg Loser:[/bold]       ${avg_loss:,.2f}\n"
        f"[bold]Profit Factor:[/bold]   {profit_factor:.2f}",
        title="[bold]All-Time Stats[/bold]",
        border_style="magenta",
    ))


def render_7day_pnl(summaries: list) -> None:
    if not summaries:
        return
    table = Table(title="[bold]Last 7 Days P&L[/bold]")
    table.add_column("Date")
    table.add_column("P&L",    justify="right")
    table.add_column("Win%",   justify="right")
    table.add_column("Trades", justify="right")
    table.add_column("Bar")

    max_abs = max(abs(float(s.get("gross_pnl") or 0)) for s in summaries) or 1

    for s in reversed(summaries):
        pnl = float(s.get("gross_pnl") or 0)
        wr  = float(s.get("win_rate") or 0) * 100
        n   = s.get("total_trades") or 0
        bar_len = int(abs(pnl) / max_abs * 20)
        bar_char = "█" * bar_len
        bar_style = "green" if pnl >= 0 else "red"
        table.add_row(
            s["date"],
            _fmt_pnl(pnl),
            f"{wr:.1f}%",
            str(n),
            Text(bar_char, style=bar_style),
        )
    console.print(table)


def render_scan_summary(scan: dict) -> None:
    if not scan:
        return
    console.print(Panel(
        f"[bold]Last Scan:[/bold] {scan.get('timestamp','')}\n"
        f"[bold]Session:[/bold]   {scan.get('session','')}\n"
        f"[bold]Scanned:[/bold]   {scan.get('tickers_scanned',0)} tickers\n"
        f"[bold]Signals:[/bold]   {scan.get('signals_generated',0)} generated\n"
        f"[bold]Executed:[/bold]  {scan.get('trades_executed',0)} trades\n"
        f"[bold]Bull:[/bold]      {scan.get('total_bull_signals',0)}   "
        f"[bold]Bear:[/bold] {scan.get('total_bear_signals',0)}",
        title="[bold]Last Scan[/bold]",
        border_style="blue",
    ))


def main() -> None:
    init_db()
    open_trades  = get_open_trades()
    today_trades = get_trades_today()
    all_trades   = get_all_trades()
    summaries    = get_daily_summaries(7)
    last_scan    = get_last_scan()

    # Compute daily P&L from today's closed trades
    daily_pnl = sum(float(t.get("pnl_dollar") or 0) for t in today_trades if t.get("pnl_dollar") is not None)

    # Try to get live portfolio value from Alpaca
    portfolio_value = 0.0
    cash = 0.0
    try:
        from dotenv import load_dotenv
        load_dotenv()
        from bot.trader import build_client, get_account
        client  = build_client()
        account = get_account(client)
        portfolio_value = account.get("portfolio_value", 0)
        cash = account.get("cash", 0)
    except Exception:
        pass

    render_header(portfolio_value, cash, daily_pnl)
    render_open_positions(open_trades)
    render_closed_trades(today_trades)
    render_all_time_stats(all_trades)
    render_7day_pnl(summaries)
    render_scan_summary(last_scan)


if __name__ == "__main__":
    main()
