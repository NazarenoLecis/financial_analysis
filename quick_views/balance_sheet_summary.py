"""Balance sheet summary for a public company.

The balance sheet shows what a company owns, what it owes, and what is left for
shareholders at a specific fiscal period.

Basic structure:

Assets = Liabilities + Equity

This script prints an easy-to-read snapshot of the latest available balance
sheet and calculates a few simple ratios:

- current ratio: short-term assets divided by short-term liabilities;
- debt to equity: total debt divided by shareholder equity;
- liabilities to assets: how much of the asset base is financed by liabilities;
- cash to assets: how much of assets are held as cash and short-term investments.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

# Add the project root to the import path so this file can be run directly from
# VS Code while still importing utils.py from the parent folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import fetch_financial_data, safe_divide, statement_series, statement_value


BALANCE_SHEET_FIELDS = {
    "Cash And Short Term Investments": [
        "Cash Cash Equivalents And Short Term Investments",
        "Cash And Cash Equivalents",
        "Cash Cash Equivalents",
    ],
    "Receivables": ["Receivables", "Accounts Receivable", "Net Receivables"],
    "Inventory": "Inventory",
    "Current Assets": ["Current Assets", "Total Current Assets"],
    "Total Assets": "Total Assets",
    "Current Liabilities": ["Current Liabilities", "Total Current Liabilities"],
    "Total Debt": ["Total Debt", "Long Term Debt And Capital Lease Obligation", "Long Term Debt"],
    "Total Liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"],
    "Stockholders Equity": [
        "Stockholders Equity",
        "Total Equity Gross Minority Interest",
        "Common Stock Equity",
    ],
    "Retained Earnings": "Retained Earnings",
    "Working Capital": "Working Capital",
}


def optional_statement_value(statement: pd.DataFrame, period, aliases) -> Any:
    """Read a balance sheet value, returning missing instead of stopping the script."""

    return statement_value(statement, period, aliases, required=False, default=pd.NA)


def safe_ratio(numerator, denominator, label: str) -> Any:
    """Calculate a ratio only when both values are available and the denominator is not zero."""

    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return pd.NA
    return safe_divide(float(numerator), float(denominator), label)


def build_balance_sheet_snapshot(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build latest balance sheet details, ratios, and historical statement lines."""

    data = fetch_financial_data(ticker)
    latest_period = data.periods[0]

    # Read the latest value for each field. Some companies do not report every
    # line item, so optional values are displayed as missing instead of crashing.
    latest_values = {
        label: optional_statement_value(data.balance_sheet, latest_period, aliases)
        for label, aliases in BALANCE_SHEET_FIELDS.items()
    }

    # If yfinance does not provide Working Capital directly, calculate it from
    # current assets and current liabilities.
    if pd.isna(latest_values["Working Capital"]):
        latest_values["Working Capital"] = (
            latest_values["Current Assets"] - latest_values["Current Liabilities"]
            if not pd.isna(latest_values["Current Assets"]) and not pd.isna(latest_values["Current Liabilities"])
            else pd.NA
        )

    snapshot = pd.DataFrame(
        {
            "metric": latest_values.keys(),
            "value": latest_values.values(),
        }
    )

    ratios = pd.DataFrame(
        [
            {
                "metric": "Current Ratio",
                "value": safe_ratio(
                    latest_values["Current Assets"],
                    latest_values["Current Liabilities"],
                    "current ratio",
                ),
            },
            {
                "metric": "Debt To Equity",
                "value": safe_ratio(
                    latest_values["Total Debt"],
                    latest_values["Stockholders Equity"],
                    "debt to equity",
                ),
            },
            {
                "metric": "Liabilities To Assets",
                "value": safe_ratio(
                    latest_values["Total Liabilities"],
                    latest_values["Total Assets"],
                    "liabilities to assets",
                ),
            },
            {
                "metric": "Cash To Assets",
                "value": safe_ratio(
                    latest_values["Cash And Short Term Investments"],
                    latest_values["Total Assets"],
                    "cash to assets",
                ),
            },
        ]
    )

    # Build a historical table so the user can see whether assets, liabilities,
    # and equity are rising or falling over time.
    history = pd.DataFrame(
        {
            label: statement_series(data.balance_sheet, data.periods, aliases)
            for label, aliases in BALANCE_SHEET_FIELDS.items()
        }
    )
    history = history.dropna(axis=1, how="all").dropna(axis=0, how="all")

    return snapshot, ratios, history


def plot_balance_sheet(ticker: str, history: pd.DataFrame, years: int) -> None:
    """Show balance sheet trends for the selected number of recent years."""

    recent = history.tail(years)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    # First chart: the big accounting structure of assets, liabilities, equity.
    major_columns = [
        column
        for column in ["Total Assets", "Total Liabilities", "Stockholders Equity"]
        if column in recent
    ]
    recent[major_columns].plot(ax=axes[0], marker="o")
    axes[0].set_title(f"{ticker} Balance Sheet Structure")
    axes[0].set_ylabel("USD")

    # Second chart: short-term liquidity items.
    liquidity_columns = [
        column
        for column in ["Current Assets", "Current Liabilities", "Cash And Short Term Investments", "Working Capital"]
        if column in recent
    ]
    recent[liquidity_columns].plot(ax=axes[1], marker="o")
    axes[1].set_title("Liquidity View")
    axes[1].set_ylabel("USD")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize the latest balance sheet for a public company.")

    # Defaults make the script useful by simply pressing Run Python File in VS Code.
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--years", type=int, default=4)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    snapshot, ratios, history = build_balance_sheet_snapshot(args.ticker)

    print(f"Latest balance sheet snapshot for {args.ticker}:")
    print(snapshot.to_string(index=False, float_format=lambda value: f"{value:,.0f}"))

    print("\nBalance sheet ratios:")
    print(ratios.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    print(f"\nRecent balance sheet history ({args.years} years):")
    print(history.tail(args.years).to_string(float_format=lambda value: f"{value:,.0f}"))

    if not args.no_plot:
        plot_balance_sheet(args.ticker, history, args.years)


if __name__ == "__main__":
    main()
