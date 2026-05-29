"""MACD technical analysis.

MACD stands for Moving Average Convergence Divergence. It compares a fast
exponential moving average with a slower one. The signal line is another moving
average of the MACD line.

When MACD is above the signal line, this script labels the latest status as
bullish. When it is below, it labels the status as bearish.
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

from technical_analysis.indicators import macd
from utils import RichHelpFormatter, fetch_price_history


def analyze_macd(ticker: str, start_date: str, end_date: str | None, fast: int, slow: int, signal: int):
    """Calculate MACD and identify whether momentum is positive or negative."""

    # MACD is calculated from closing prices.
    prices = fetch_price_history(ticker, start_date, end_date)
    macd_frame = macd(prices["Close"], fast, slow, signal)

    # Keep the close price together with MACD values so the printed table is
    # easy to read.
    analysis = prices[["Close"]].join(macd_frame)
    latest = analysis.dropna().iloc[-1]

    # Simple interpretation of the latest crossover relationship.
    status = "bullish" if latest["macd"] > latest["signal"] else "bearish"
    return analysis, status


def plot_macd(ticker: str, analysis) -> None:
    """Show price, MACD line, signal line, and histogram."""

    # Price is shown above, MACD details below.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    analysis["Close"].plot(ax=axes[0])
    axes[0].set_title(f"{ticker} Price")
    axes[0].set_ylabel("Price")

    analysis[["macd", "signal"]].plot(ax=axes[1])

    # Histogram bars show the distance between the MACD line and signal line.
    axes[1].bar(analysis.index, analysis["histogram"], color="gray", alpha=0.4)
    axes[1].set_title("MACD")
    axes[1].set_ylabel("Value")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate and chart MACD for a stock price time series.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  --ticker accepts one Yahoo Finance symbol, e.g. AAPL.
  --start-date and --end-date use YYYY-MM-DD format.
  --fast, --slow, and --signal are EMA periods in trading days.

Data used:
  This is a price time-series script. It uses daily close prices from yfinance.

Interpretation:
  Latest status is bullish when MACD is above the signal line.
  Latest status is bearish when MACD is below the signal line.

Examples:
  python technical_analysis/macd_analysis.py
  python technical_analysis/macd_analysis.py --ticker MSFT --fast 12 --slow 26 --signal 9
  python technical_analysis/macd_analysis.py --ticker AAPL --start-date 2023-01-01 --no-plot
""",
    )

    # Defaults make the script runnable from VS Code without arguments.
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker symbol.")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date for price history, in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Optional end date for price history, in YYYY-MM-DD format.")
    parser.add_argument("--fast", type=int, default=12, help="Fast EMA period.")
    parser.add_argument("--slow", type=int, default=26, help="Slow EMA period.")
    parser.add_argument("--signal", type=int, default=9, help="Signal-line EMA period.")
    parser.add_argument("--no-plot", action="store_true", help="Print output only and do not open a matplotlib chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis, status = analyze_macd(args.ticker, args.start_date, args.end_date, args.fast, args.slow, args.signal)
    print(f"Latest MACD status: {status}")
    print(analysis.tail().to_string(float_format=lambda value: f"{value:.2f}"))

    if not args.no_plot:
        plot_macd(args.ticker, analysis)


if __name__ == "__main__":
    main()
