import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import fetch_financial_data, fetch_index_tickers, market_cap_from_stock, safe_divide, statement_value


def calculate_altman_z_score(ticker: str) -> float:
    """Calculate the original public-company Altman Z-score for one ticker."""

    data = fetch_financial_data(ticker)
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

    working_capital = current_assets - current_liabilities
    x1 = safe_divide(working_capital, total_assets, "working capital / total assets")
    x2 = safe_divide(retained_earnings, total_assets, "retained earnings / total assets")
    x3 = safe_divide(ebit, total_assets, "EBIT / total assets")
    x4 = safe_divide(
        market_cap_from_stock(data.stock, data.ticker),
        total_liabilities,
        "market value of equity / total liabilities",
    )
    x5 = safe_divide(sales, total_assets, "sales / total assets")

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
