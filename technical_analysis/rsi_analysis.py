"""Relative Strength Index analysis.

RSI is a momentum indicator that compares recent gains with recent losses. It is
commonly shown on a 0 to 100 scale:

- above 70 is often called overbought;
- below 30 is often called oversold;
- between 30 and 70 is usually treated as neutral.
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

from technical_analysis.indicators import relative_strength_index
from utils import RichHelpFormatter, fetch_price_history


def analyze_rsi(ticker: str, start_date: str, end_date: str | None, window: int):
    """Calculate RSI and label the latest momentum zone."""

    # Download the price history and keep only Close for the RSI calculation.
    prices = fetch_price_history(ticker, start_date, end_date)
    analysis = prices[["Close"]].copy()

    # Add RSI as a second column so price and indicator stay aligned by date.
    analysis["RSI"] = relative_strength_index(analysis["Close"], window)
    latest_rsi = analysis["RSI"].dropna().iloc[-1]

    # These thresholds are common conventions, not guaranteed trading signals.
    if latest_rsi >= 70:
        status = "overbought"
    elif latest_rsi <= 30:
        status = "oversold"
    else:
        status = "neutral"

    return analysis, status


def plot_rsi(ticker: str, analysis) -> None:
    """Show price and RSI with standard overbought/oversold levels."""

    # Two panels make it easier to see price and RSI separately.
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    analysis["Close"].plot(ax=axes[0])
    axes[0].set_title(f"{ticker} Price")
    axes[0].set_ylabel("Price")

    analysis["RSI"].plot(ax=axes[1], color="purple")

    # Horizontal guide lines mark the common overbought and oversold zones.
    axes[1].axhline(70, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=1)
    axes[1].set_title("Relative Strength Index")
    axes[1].set_ylabel("RSI")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate and chart the Relative Strength Index.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  --ticker accepts one Yahoo Finance symbol, e.g. AAPL.
  --start-date and --end-date use YYYY-MM-DD format.
  --window is the number of trading days used for average gains/losses.

Data used:
  This is a price time-series script. It uses daily close prices from yfinance.

Interpretation:
  RSI >= 70 is labelled overbought.
  RSI <= 30 is labelled oversold.
  Values between 30 and 70 are labelled neutral.

Examples:
  python technical_analysis/rsi_analysis.py
  python technical_analysis/rsi_analysis.py --ticker MSFT --window 14
  python technical_analysis/rsi_analysis.py --ticker AAPL --start-date 2023-01-01 --no-plot
""",
    )

    # Defaults make the script work from VS Code with no arguments.
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker symbol.")
    parser.add_argument("--start-date", default="2020-01-01", help="Start date for price history, in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Optional end date for price history, in YYYY-MM-DD format.")
    parser.add_argument("--window", type=int, default=14, help="Rolling window for RSI gains/losses.")
    parser.add_argument("--no-plot", action="store_true", help="Print output only and do not open a matplotlib chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis, status = analyze_rsi(args.ticker, args.start_date, args.end_date, args.window)
    print(f"Latest RSI status: {status}")
    print(analysis.tail().to_string(float_format=lambda value: f"{value:.2f}"))

    if not args.no_plot:
        plot_rsi(args.ticker, analysis)


if __name__ == "__main__":
    main()
