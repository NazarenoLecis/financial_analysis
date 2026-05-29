"""Financial statement charts.

This script pulls the main annual income statement, cash flow, and balance sheet
lines for one company. It prints the data and can show three simple matplotlib
charts.

The goal is to help users visually inspect revenue, profits, cash generation,
assets, liabilities, and equity over time.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Add project root to import path so direct file execution can import utils.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import RichHelpFormatter, fetch_financial_data, statement_series


def build_statement_frame(ticker: str) -> pd.DataFrame:
    """Collect the main financial statement lines into one time-series table."""

    data = fetch_financial_data(ticker)
    periods = data.periods

    # Not every company reports every field with the same label, so these use aliases.
    # Each column below becomes one line in the printed table and charts.
    frame = pd.DataFrame(
        {
            "Revenue": statement_series(data.income_statement, periods, ["Total Revenue", "Revenue"]),
            "Gross Profit": statement_series(data.income_statement, periods, "Gross Profit"),
            "Operating Income": statement_series(data.income_statement, periods, ["Operating Income", "EBIT"]),
            "Net Income": statement_series(data.income_statement, periods, "Net Income"),
            "Operating Cash Flow": statement_series(
                data.cash_flow,
                periods,
                ["Operating Cash Flow", "Total Cash From Operating Activities"],
            ),
            "Free Cash Flow": statement_series(data.cash_flow, periods, "Free Cash Flow"),
            "Total Assets": statement_series(data.balance_sheet, periods, "Total Assets"),
            "Total Liabilities": statement_series(
                data.balance_sheet,
                periods,
                ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"],
            ),
            "Stockholders Equity": statement_series(
                data.balance_sheet,
                periods,
                ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
            ),
        }
    )
    # Drop rows and columns that are fully empty. yfinance sometimes returns an
    # extra year in one statement that does not exist in the others.
    return frame.dropna(axis=1, how="all").dropna(axis=0, how="all")


def plot_statement_charts(ticker: str, frame: pd.DataFrame) -> None:
    """Show income, cash-flow, and balance-sheet charts for one company."""

    # Use three panels so the user can compare related statement lines without
    # putting every metric on one crowded chart.
    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    income_columns = [column for column in ["Revenue", "Gross Profit", "Operating Income", "Net Income"] if column in frame]
    cash_columns = [column for column in ["Operating Cash Flow", "Free Cash Flow"] if column in frame]
    balance_columns = [
        column
        for column in ["Total Assets", "Total Liabilities", "Stockholders Equity"]
        if column in frame
    ]

    frame[income_columns].plot(ax=axes[0], marker="o")
    axes[0].set_title(f"{ticker} Income Statement")
    axes[0].set_ylabel("USD")

    frame[cash_columns].plot(ax=axes[1], marker="o")
    axes[1].set_title("Cash Flow")
    axes[1].set_ylabel("USD")

    frame[balance_columns].plot(ax=axes[2], marker="o")
    axes[2].set_title("Balance Sheet")
    axes[2].set_ylabel("USD")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print and chart annual financial statement time series.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  --ticker accepts one Yahoo Finance symbol, e.g. AAPL.

Data used:
  This is a time-series script. It uses annual income statement, cash flow, and
  balance sheet values from yfinance.

Printed table:
  Rows are fiscal years.
  Columns are major statement lines such as revenue, net income, free cash flow,
  total assets, total liabilities, and stockholders equity.

Examples:
  python fundamental_analysis/financial_statement_charts.py
  python fundamental_analysis/financial_statement_charts.py --ticker MSFT
  python fundamental_analysis/financial_statement_charts.py --ticker AAPL --no-plot
""",
    )

    # Defaults make this script work from VS Code without command-line options.
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker symbol.")
    parser.add_argument("--no-plot", action="store_true", help="Print output only and do not open a matplotlib chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = build_statement_frame(args.ticker)
    print(frame.sort_index(ascending=False).to_string(float_format=lambda value: f"{value:,.0f}"))

    if not args.no_plot:
        plot_statement_charts(args.ticker, frame)


if __name__ == "__main__":
    main()
