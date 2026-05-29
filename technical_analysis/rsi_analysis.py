import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from technical_analysis.indicators import relative_strength_index
from utils import fetch_price_history


def analyze_rsi(ticker: str, start_date: str, end_date: str | None, window: int):
    """Calculate RSI and label the latest momentum zone."""

    prices = fetch_price_history(ticker, start_date, end_date)
    analysis = prices[["Close"]].copy()
    analysis["RSI"] = relative_strength_index(analysis["Close"], window)
    latest_rsi = analysis["RSI"].dropna().iloc[-1]

    if latest_rsi >= 70:
        status = "overbought"
    elif latest_rsi <= 30:
        status = "oversold"
    else:
        status = "neutral"

    return analysis, status


def plot_rsi(ticker: str, analysis) -> None:
    """Show price and RSI with standard overbought/oversold levels."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    analysis["Close"].plot(ax=axes[0])
    axes[0].set_title(f"{ticker} Price")
    axes[0].set_ylabel("Price")

    analysis["RSI"].plot(ax=axes[1], color="purple")
    axes[1].axhline(70, color="red", linestyle="--", linewidth=1)
    axes[1].axhline(30, color="green", linestyle="--", linewidth=1)
    axes[1].set_title("Relative Strength Index")
    axes[1].set_ylabel("RSI")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate and chart RSI.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--window", type=int, default=14)
    parser.add_argument("--no-plot", action="store_true")
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
