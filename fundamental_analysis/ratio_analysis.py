import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import fetch_financial_data, fetch_index_tickers, market_cap_from_stock, safe_divide, statement_value


def calculate_ratios(ticker: str) -> dict[str, float | str]:
    """Calculate a practical set of liquidity, leverage, profitability, and valuation ratios."""

    data = fetch_financial_data(ticker)
    period = data.periods[0]

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
    revenue = statement_value(data.income_statement, period, ["Total Revenue", "Revenue"])
    cost_of_revenue = statement_value(data.income_statement, period, ["Cost Of Revenue", "Cost Of Goods Sold"])
    operating_income = statement_value(data.income_statement, period, ["Operating Income", "EBIT"])
    net_income = statement_value(data.income_statement, period, "Net Income")
    market_cap = market_cap_from_stock(data.stock, data.ticker)

    gross_profit = revenue - cost_of_revenue

    return {
        "ticker": ticker,
        "period": str(period.date()) if hasattr(period, "date") else str(period),
        "current_ratio": safe_divide(current_assets, current_liabilities, "current ratio"),
        "debt_to_equity": safe_divide(total_liabilities, equity, "debt to equity"),
        "gross_margin": safe_divide(gross_profit, revenue, "gross margin"),
        "operating_margin": safe_divide(operating_income, revenue, "operating margin"),
        "net_margin": safe_divide(net_income, revenue, "net margin"),
        "return_on_assets": safe_divide(net_income, total_assets, "return on assets"),
        "return_on_equity": safe_divide(net_income, equity, "return on equity"),
        "asset_turnover": safe_divide(revenue, total_assets, "asset turnover"),
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
    parser = argparse.ArgumentParser(description="Calculate fundamental ratios for one or more tickers.")
    parser.add_argument("--index", choices=["sp500", "nasdaq100"], default="sp500")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols to analyze. Overrides --index.")
    parser.add_argument("--limit", type=int, help="Limit the number of index tickers processed.")
    parser.add_argument("--plot", action="store_true", help="Show a matplotlib ratio comparison chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    use_index = args.tickers is None and ("--index" in sys.argv or "--limit" in sys.argv)
    tickers = fetch_index_tickers(args.index) if use_index else (args.tickers or ["AAPL", "MSFT"])
    if args.limit:
        tickers = tickers[: args.limit]

    rows = []
    errors = {}
    for ticker in tickers:
        try:
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
