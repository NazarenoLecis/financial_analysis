"""General fundamental ratio analysis.

This script calculates a compact set of ratios that beginners often use to
understand a company's liquidity, leverage, profitability, efficiency, and
valuation.

The default run compares AAPL and MSFT so there is useful output even when the
file is started directly from VS Code without command-line arguments.
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

from utils import RichHelpFormatter, fetch_financial_data, fetch_index_tickers, market_cap_from_stock, safe_divide, statement_value


def calculate_ratios(ticker: str) -> dict[str, float | str]:
    """Calculate a practical set of liquidity, leverage, profitability, and valuation ratios."""

    data = fetch_financial_data(ticker)
    period = data.periods[0]

    # Balance-sheet fields describe what the company owns and owes at the end
    # of the fiscal year.
    current_assets = statement_value(data.balance_sheet, period, ["Current Assets", "Total Current Assets"])
    current_liabilities = statement_value(
        data.balance_sheet,
        period,
        ["Current Liabilities", "Total Current Liabilities"],
    )
    total_assets = statement_value(data.balance_sheet, period, "Total Assets")
    total_liabilities = statement_value(
        data.balance_sheet,
        period,
        ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"],
    )
    equity = statement_value(
        data.balance_sheet,
        period,
        ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
    )
    # Income-statement fields describe activity during the fiscal year.
    revenue = statement_value(data.income_statement, period, ["Total Revenue", "Revenue"])
    cost_of_revenue = statement_value(data.income_statement, period, ["Cost Of Revenue", "Cost Of Goods Sold"])
    operating_income = statement_value(data.income_statement, period, ["Operating Income", "EBIT"])
    net_income = statement_value(data.income_statement, period, "Net Income")
    market_cap = market_cap_from_stock(data.stock, data.ticker)

    gross_profit = revenue - cost_of_revenue

    return {
        "ticker": ticker,
        "period": str(period.date()) if hasattr(period, "date") else str(period),
        # Liquidity: can the company cover short-term obligations?
        "current_ratio": safe_divide(current_assets, current_liabilities, "current ratio"),

        # Leverage: how much debt-like obligation exists relative to equity?
        "debt_to_equity": safe_divide(total_liabilities, equity, "debt to equity"),

        # Profitability: how much profit remains at different stages?
        "gross_margin": safe_divide(gross_profit, revenue, "gross margin"),
        "operating_margin": safe_divide(operating_income, revenue, "operating margin"),
        "net_margin": safe_divide(net_income, revenue, "net margin"),

        # Returns: how efficiently assets and equity generate profit?
        "return_on_assets": safe_divide(net_income, total_assets, "return on assets"),
        "return_on_equity": safe_divide(net_income, equity, "return on equity"),
        "asset_turnover": safe_divide(revenue, total_assets, "asset turnover"),

        # Valuation: how the market price compares with sales, book value, and
        # earnings.
        "price_to_sales": safe_divide(market_cap, revenue, "price to sales"),
        "price_to_book": safe_divide(market_cap, equity, "price to book"),
        "earnings_yield": safe_divide(net_income, market_cap, "earnings yield"),
    }


def plot_ratios(results: pd.DataFrame) -> None:
    """Show comparable ratio bars for all tickers in the run."""

    plot_columns = [
        "current_ratio",
        "debt_to_equity",
        "gross_margin",
        "operating_margin",
        "net_margin",
        "return_on_assets",
        "return_on_equity",
    ]
    results.set_index("ticker")[plot_columns].plot(kind="bar", figsize=(12, 6))
    plt.title("Fundamental Ratio Comparison")
    plt.ylabel("Ratio")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate liquidity, leverage, profitability, efficiency, and valuation ratios.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  --tickers accepts one or more Yahoo Finance symbols, e.g. AAPL MSFT NVDA.
  --index accepts one of: sp500, nasdaq100.
  --plot opens a comparison chart with matplotlib.

Data used:
  Ratios use the latest available annual financial statements plus current
  market capitalization from yfinance.

Examples:
  python fundamental_analysis/ratio_analysis.py
  python fundamental_analysis/ratio_analysis.py --tickers AAPL MSFT NVDA --plot
  python fundamental_analysis/ratio_analysis.py --index sp500 --limit 5
""",
    )
    parser.add_argument("--index", choices=["sp500", "nasdaq100"], default="sp500", help="Index universe to use when running in index mode.")
    parser.add_argument("--tickers", nargs="+", help="One or more Yahoo Finance ticker symbols. Overrides --index.")
    parser.add_argument("--limit", type=int, help="Maximum number of index tickers to process.")
    parser.add_argument("--plot", action="store_true", help="Show a matplotlib ratio comparison chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Running without arguments compares AAPL and MSFT. If --index or --limit is
    # used, fetch tickers from the selected index instead.
    use_index = args.tickers is None and ("--index" in sys.argv or "--limit" in sys.argv)
    tickers = fetch_index_tickers(args.index) if use_index else (args.tickers or ["AAPL", "MSFT"])
    if args.limit:
        tickers = tickers[: args.limit]

    rows = []
    errors = {}
    for ticker in tickers:
        try:
            # Each ticker produces one row in the comparison table.
            rows.append(calculate_ratios(ticker))
        except Exception as exc:
            errors[ticker] = str(exc)
            print(f"{ticker}: failed ({exc})")

    results = pd.DataFrame(rows)
    if not results.empty:
        print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if errors:
        print("\nErrors:")
        for ticker, error in errors.items():
            print(f"{ticker}: {error}")

    if args.plot and not results.empty:
        plot_ratios(results)


if __name__ == "__main__":
    main()
