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
from utils import fetch_price_history


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
    parser = argparse.ArgumentParser(description="Chart price with simple moving averages.")

    # Defaults make the script runnable from VS Code without arguments.
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--windows", nargs="+", type=int, default=[20, 50, 200])
    parser.add_argument("--no-plot", action="store_true")
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
