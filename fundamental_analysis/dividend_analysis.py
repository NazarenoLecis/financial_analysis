"""Dividend history analysis.

This script looks at a company's dividend payments, groups them by year, and
calculates dividend growth and current dividend yield. It is useful for income
investors who want to see whether dividends have been stable and growing.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

# Add project root to import path so direct file execution can import utils.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import fetch_price_history, yahoo_symbol


def calculate_dividend_history(ticker: str, cagr_years: int = 5) -> tuple[pd.DataFrame, dict[str, float]]:
    """Summarize annual dividends, growth, and current dividend yield."""

    yahoo_ticker = yahoo_symbol(ticker)
    stock = yf.Ticker(yahoo_ticker)

    # yfinance returns individual dividend payments with dates. We group those
    # individual payments into annual totals below.
    dividends = stock.dividends
    if dividends.empty:
        raise ValueError(f"No dividend history found for {ticker}")

    # Sum all dividend payments in each calendar year.
    annual = dividends.groupby(dividends.index.year).sum().rename("annual_dividend").to_frame()
    annual.index.name = "year"

    # pct_change shows year-over-year dividend growth.
    annual["dividend_growth"] = annual["annual_dividend"].pct_change()

    # Exclude the current calendar year from growth analysis because it is usually incomplete.
    current_year = pd.Timestamp.today(tz=dividends.index.tz).year if dividends.index.tz else pd.Timestamp.today().year
    completed_annual = annual[annual.index < current_year]
    if completed_annual.empty:
        completed_annual = annual

    # Download recent prices so dividend yield can be calculated using the
    # latest close price.
    prices = fetch_price_history(ticker, period_start_from_history(completed_annual))
    latest_price = float(prices["Close"].dropna().iloc[-1])

    # Trailing 12-month dividend is better for current yield than the current
    # calendar year, because the current year may not be complete.
    trailing_start = pd.Timestamp.today(tz=dividends.index.tz) - pd.DateOffset(years=1)
    current_dividend = float(dividends[dividends.index >= trailing_start].sum())
    # Use a recent CAGR window by default. This avoids making old dividend gaps
    # dominate the growth calculation.
    cagr_window = completed_annual.tail(cagr_years + 1)
    first_dividend = float(cagr_window["annual_dividend"].iloc[0])
    last_completed_dividend = float(completed_annual["annual_dividend"].iloc[-1])
    years = max(len(cagr_window) - 1, 1)
    recent_dividend_cagr = (
        (last_completed_dividend / first_dividend) ** (1 / years) - 1
        if first_dividend > 0
        else pd.NA
    )

    summary = {
        "latest_price": latest_price,
        "trailing_12_month_dividend": current_dividend,
        "last_completed_year_dividend": last_completed_dividend,
        "current_dividend_yield": current_dividend / latest_price,
        f"{years}_year_dividend_cagr": recent_dividend_cagr,
    }
    return completed_annual, summary


def period_start_from_history(annual: pd.DataFrame) -> str:
    """Start the price lookup near the dividend history to keep downloads small."""

    return f"{int(annual.index.min())}-01-01"


def plot_dividends(ticker: str, annual: pd.DataFrame) -> None:
    """Show annual dividends and year-over-year dividend growth."""

    # Top panel: dollars of dividends per share. Bottom panel: annual growth.
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    annual["annual_dividend"].plot(kind="bar", ax=axes[0])
    axes[0].set_title(f"{ticker} Annual Dividends")
    axes[0].set_ylabel("Dividend per Share")

    annual["dividend_growth"].plot(kind="bar", ax=axes[1], color="gray")
    axes[1].set_title("Dividend Growth")
    axes[1].set_ylabel("YoY Growth")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze dividend history for one ticker.")

    # Defaults make this script work by pressing Run Python File in VS Code.
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--cagr-years", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annual, summary = calculate_dividend_history(args.ticker, args.cagr_years)
    print(annual.to_string(float_format=lambda value: f"{value:.4f}"))
    print("\nSummary:")
    for label, value in summary.items():
        print(f"{label}: {value:.4f}")

    if not args.no_plot:
        plot_dividends(args.ticker, annual)


if __name__ == "__main__":
    main()
