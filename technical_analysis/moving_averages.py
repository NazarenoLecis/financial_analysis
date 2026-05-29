"""Moving-average technical analysis.

Moving averages smooth daily prices so trends are easier to see. This script
calculates simple moving averages and gives a basic trend label:

- bullish when the shortest moving average is above the longest one;
- bearish when the shortest moving average is below the longest one.
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

from technical_analysis.indicators import simple_moving_average
from utils import RichHelpFormatter, fetch_price_history


def analyze_moving_averages(ticker: str, start_date: str, end_date: str | None, windows: list[int]):
    """Calculate moving averages and a simple trend status."""

    # Download price history and use the Close column for moving averages.
    prices = fetch_price_history(ticker, start_date, end_date)
    close = prices["Close"]
    analysis = prices[["Close"]].copy()

    # Add one moving-average column for each requested window.
    for window in windows:
        analysis[f"SMA_{window}"] = simple_moving_average(close, window)

    # Compare the shortest and longest moving averages for a simple trend label.
    short_window, long_window = min(windows), max(windows)
    latest = analysis.dropna().iloc[-1]
    status = "bullish" if latest[f"SMA_{short_window}"] > latest[f"SMA_{long_window}"] else "bearish"
    return analysis, status


def plot_moving_averages(ticker: str, analysis) -> None:
    """Show price with selected moving averages."""

    # Plot Close and all SMA columns on the same axis so crossings are visible.
    analysis.plot(figsize=(12, 6))
    plt.title(f"{ticker} Moving Averages")
    plt.ylabel("Price")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chart a stock price with one or more simple moving averages.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  --ticker accepts one Yahoo Finance symbol, e.g. AAPL.
  --start-date and --end-date use YYYY-MM-DD format.
  --windows accepts one or more integer moving-average windows in trading days.

Data used:
  This is a price time-series script. It uses daily close prices from yfinance.

Interpretation:
  Latest status is bullish when the shortest SMA is above the longest SMA.
  Latest status is bearish when the shortest SMA is below the longest SMA.

Examples:
  python technical_analysis/moving_averages.py
  python technical_analysis/moving_averages.py --ticker MSFT --windows 20 50 200
  python technical_analysis/moving_averages.py --ticker AAPL --start-date 2023-01-01 --no-plot
""",
    )

    # Defaults make the script runnable from VS Code without arguments.
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker symbol.")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date for price history, in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Optional end date for price history, in YYYY-MM-DD format.")
    parser.add_argument("--windows", nargs="+", type=int, default=[20, 50, 200], help="One or more SMA windows in trading days.")
    parser.add_argument("--no-plot", action="store_true", help="Print output only and do not open a matplotlib chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis, status = analyze_moving_averages(args.ticker, args.start_date, args.end_date, args.windows)
    print(f"Latest moving-average status: {status}")
    print(analysis.tail().to_string(float_format=lambda value: f"{value:.2f}"))

    if not args.no_plot:
        plot_moving_averages(args.ticker, analysis)


if __name__ == "__main__":
    main()
