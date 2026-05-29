"""Altman Z-score analysis.

The Altman Z-score is a classic bankruptcy-risk indicator. It combines five
ratios from the balance sheet, income statement, and market capitalization.

Higher scores usually suggest lower financial distress risk. A common rule of
thumb is:

- above 3.0: safer zone;
- 1.8 to 3.0: grey zone;
- below 1.8: distress zone.

This script defaults to AAPL when run directly from VS Code, but you can pass
`--tickers` or `--index` from the command line.
"""

import argparse
import sys
from pathlib import Path

# When this file is run from inside the fundamental_analysis folder, Python
# would not normally see utils.py in the project root. This adds the project root
# to the import path so direct file execution works.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import fetch_financial_data, fetch_index_tickers, market_cap_from_stock, safe_divide, statement_value


def calculate_altman_z_score(ticker: str) -> float:
    """Calculate the original public-company Altman Z-score for one ticker."""

    data = fetch_financial_data(ticker)

    # yfinance orders annual financial statements newest first. period 0 is the
    # most recent fiscal year available.
    period = data.periods[0]

    # Altman's five factors combine liquidity, retained earnings, profitability,
    # market leverage, and asset turnover.
    current_assets = statement_value(data.balance_sheet, period, ["Current Assets", "Total Current Assets"])
    current_liabilities = statement_value(
        data.balance_sheet,
        period,
        ["Current Liabilities", "Total Current Liabilities"],
    )
    total_assets = statement_value(data.balance_sheet, period, "Total Assets")
    retained_earnings = statement_value(data.balance_sheet, period, "Retained Earnings")
    ebit = statement_value(data.income_statement, period, ["EBIT", "Operating Income"])
    total_liabilities = statement_value(
        data.balance_sheet,
        period,
        ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"],
    )
    sales = statement_value(data.income_statement, period, ["Total Revenue", "Revenue"])

    # X1: working capital / total assets. This measures short-term liquidity.
    working_capital = current_assets - current_liabilities
    x1 = safe_divide(working_capital, total_assets, "working capital / total assets")

    # X2: retained earnings / total assets. This captures accumulated profits
    # kept in the business over time.
    x2 = safe_divide(retained_earnings, total_assets, "retained earnings / total assets")

    # X3: EBIT / total assets. This measures operating profitability relative
    # to the asset base.
    x3 = safe_divide(ebit, total_assets, "EBIT / total assets")

    # X4: market value of equity / total liabilities. This compares what the
    # market thinks the company is worth with what it owes.
    x4 = safe_divide(
        market_cap_from_stock(data.stock, data.ticker),
        total_liabilities,
        "market value of equity / total liabilities",
    )

    # X5: sales / total assets. This measures how efficiently assets generate
    # revenue.
    x5 = safe_divide(sales, total_assets, "sales / total assets")

    # Original Altman public manufacturing-company weights.
    return 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + x5


def parse_args() -> argparse.Namespace:
    """Parse command-line options for batch or single-ticker scoring."""

    parser = argparse.ArgumentParser(description="Calculate Altman Z-scores.")
    parser.add_argument("--index", choices=["sp500", "nasdaq100"], default="sp500")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols to analyze. Overrides --index.")
    parser.add_argument("--limit", type=int, help="Limit the number of tickers processed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Beginner-friendly behavior: running the file with no arguments analyzes
    # AAPL. If the user passes --index or --limit, use the selected index list.
    use_index = args.tickers is None and ("--index" in sys.argv or "--limit" in sys.argv)
    tickers = fetch_index_tickers(args.index) if use_index else (args.tickers or ["AAPL"])
    if args.limit:
        tickers = tickers[: args.limit]

    results: dict[str, float] = {}
    errors: dict[str, str] = {}

    # Keep processing even when yfinance lacks a field for a specific company.
    for ticker in tickers:
        try:
            z_score = calculate_altman_z_score(ticker)
            results[ticker] = z_score
            print(f"{ticker}: Altman Z-score = {z_score:.4f}")
        except Exception as exc:
            errors[ticker] = str(exc)
            print(f"{ticker}: failed ({exc})")

    print("\nAltman Z-score results:")
    for ticker, z_score in results.items():
        print(f"{ticker}: {z_score:.4f}")

    if errors:
        print("\nErrors:")
        for ticker, error in errors.items():
            print(f"{ticker}: {error}")


if __name__ == "__main__":
    main()
