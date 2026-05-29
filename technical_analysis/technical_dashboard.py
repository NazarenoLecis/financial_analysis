"""Combined technical-analysis dashboard.

This script collects several common technical indicators into one view:

- closing price;
- trading volume;
- 20, 50, and 200 day simple moving averages;
- RSI;
- MACD.

It is useful when the user wants a quick visual summary instead of running each
indicator script separately.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

# Add project root to import path so direct file execution can import utils.py
# and the shared indicator functions.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from technical_analysis.indicators import macd, relative_strength_index, simple_moving_average
from utils import RichHelpFormatter, fetch_price_history


def build_dashboard(ticker: str, start_date: str, end_date: str | None):
    """Create one table with price, volume, moving averages, RSI, and MACD."""

    # The dashboard uses daily Close and Volume as the base data.
    prices = fetch_price_history(ticker, start_date, end_date)
    dashboard = prices[["Close", "Volume"]].copy()

    # Common trend windows: 20 trading days is roughly one month, 50 is medium
    # term, and 200 is long term.
    dashboard["SMA_20"] = simple_moving_average(dashboard["Close"], 20)
    dashboard["SMA_50"] = simple_moving_average(dashboard["Close"], 50)
    dashboard["SMA_200"] = simple_moving_average(dashboard["Close"], 200)
    # Add momentum indicators.
    dashboard["RSI"] = relative_strength_index(dashboard["Close"], 14)
    return dashboard.join(macd(dashboard["Close"]))


def plot_dashboard(ticker: str, dashboard) -> None:
    """Show a multi-panel technical dashboard."""

    # Four panels keep each concept readable: price trend, volume, RSI, and MACD.
    fig, axes = plt.subplots(4, 1, figsize=(13, 10), sharex=True, gridspec_kw={"height_ratios": [3, 1, 1, 1]})

    dashboard[["Close", "SMA_20", "SMA_50", "SMA_200"]].plot(ax=axes[0])
    axes[0].set_title(f"{ticker} Technical Dashboard")
    axes[0].set_ylabel("Price")

    dashboard["Volume"].plot(ax=axes[1], kind="bar", color="gray")
    axes[1].set_title("Volume")
    axes[1].set_ylabel("Shares")

    dashboard["RSI"].plot(ax=axes[2], color="purple")
    axes[2].axhline(70, color="red", linestyle="--", linewidth=1)
    axes[2].axhline(30, color="green", linestyle="--", linewidth=1)
    axes[2].set_title("RSI")
    axes[2].set_ylabel("RSI")

    dashboard[["macd", "signal"]].plot(ax=axes[3])
    axes[3].bar(dashboard.index, dashboard["histogram"], color="gray", alpha=0.4)
    axes[3].set_title("MACD")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show a combined technical dashboard for one stock.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  --ticker accepts one Yahoo Finance symbol, e.g. AAPL.
  --start-date and --end-date use YYYY-MM-DD format.

Data used:
  This is a price and volume time-series script. It downloads OHLCV data from
  yfinance and calculates SMA, RSI, and MACD indicators.

Output:
  Printed table shows the latest rows.
  Chart panels show price/SMA, volume, RSI, and MACD.

Examples:
  python technical_analysis/technical_dashboard.py
  python technical_analysis/technical_dashboard.py --ticker MSFT
  python technical_analysis/technical_dashboard.py --ticker AAPL --start-date 2023-01-01 --no-plot
""",
    )

    # Defaults make the script runnable from VS Code without arguments.
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker symbol.")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date for price history, in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Optional end date for price history, in YYYY-MM-DD format.")
    parser.add_argument("--no-plot", action="store_true", help="Print output only and do not open a matplotlib chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dashboard = build_dashboard(args.ticker, args.start_date, args.end_date)
    print(dashboard.tail().to_string(float_format=lambda value: f"{value:.2f}"))

    if not args.no_plot:
        plot_dashboard(args.ticker, dashboard)


if __name__ == "__main__":
    main()
