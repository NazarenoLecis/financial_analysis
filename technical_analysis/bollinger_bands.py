import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from technical_analysis.indicators import bollinger_bands
from utils import fetch_price_history


def analyze_bollinger(ticker: str, start_date: str, end_date: str | None, window: int, num_std: float):
    """Calculate Bollinger bands and classify the latest close."""

    prices = fetch_price_history(ticker, start_date, end_date)
    bands = bollinger_bands(prices["Close"], window, num_std)
    analysis = prices[["Close"]].join(bands)
    latest = analysis.dropna().iloc[-1]

    if latest["Close"] > latest["upper"]:
        status = "above upper band"
    elif latest["Close"] < latest["lower"]:
        status = "below lower band"
    else:
        status = "inside bands"

    return analysis, status


def plot_bollinger(ticker: str, analysis) -> None:
    """Show price with Bollinger upper, middle, and lower bands."""

    plt.figure(figsize=(12, 6))
    plt.plot(analysis.index, analysis["Close"], label="Close")
    plt.plot(analysis.index, analysis["middle"], label="Middle Band")
    plt.plot(analysis.index, analysis["upper"], label="Upper Band")
    plt.plot(analysis.index, analysis["lower"], label="Lower Band")
    plt.fill_between(analysis.index, analysis["lower"], analysis["upper"], color="gray", alpha=0.15)
    plt.title(f"{ticker} Bollinger Bands")
    plt.ylabel("Price")
    plt.legend()
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calculate and chart Bollinger bands.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--start-date", default="2020-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--num-std", type=float, default=2.0)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis, status = analyze_bollinger(args.ticker, args.start_date, args.end_date, args.window, args.num_std)
    print(f"Latest Bollinger status: {status}")
    print(analysis.tail().to_string(float_format=lambda value: f"{value:.2f}"))

    if not args.no_plot:
        plot_bollinger(args.ticker, analysis)


if __name__ == "__main__":
    main()
