import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import fetch_financial_data, fetch_index_tickers, require_periods, safe_divide, statement_value


def calculate_profitability(data) -> int:
    """Score the four Piotroski profitability signals."""

    current, previous = data.periods[0], data.periods[1]

    net_income = statement_value(data.income_statement, current, "Net Income")
    net_income_previous = statement_value(data.income_statement, previous, "Net Income")
    operating_cash_flow = statement_value(
        data.cash_flow,
        current,
        ["Operating Cash Flow", "Total Cash From Operating Activities"],
    )
    total_assets = statement_value(data.balance_sheet, current, "Total Assets")
    total_assets_previous = statement_value(data.balance_sheet, previous, "Total Assets")

    roa = safe_divide(net_income, total_assets, "current ROA")
    roa_previous = safe_divide(net_income_previous, total_assets_previous, "previous ROA")

    positive_roa_score = int(roa > 0)
    positive_cash_flow_score = int(operating_cash_flow > 0)
    roa_improvement_score = int(roa > roa_previous)
    accruals_score = int(operating_cash_flow > net_income)

    return positive_roa_score + positive_cash_flow_score + roa_improvement_score + accruals_score


def calculate_leverage_liquidity_and_dilution(data) -> int:
    """Score leverage reduction, liquidity improvement, and lack of share dilution."""

    current, previous = data.periods[0], data.periods[1]

    long_term_debt = statement_value(data.balance_sheet, current, "Long Term Debt", required=False, default=0)
    long_term_debt_previous = statement_value(data.balance_sheet, previous, "Long Term Debt", required=False, default=0)
    debt_score = int(long_term_debt <= long_term_debt_previous)

    current_assets = statement_value(data.balance_sheet, current, ["Current Assets", "Total Current Assets"])
    current_liabilities = statement_value(
        data.balance_sheet,
        current,
        ["Current Liabilities", "Total Current Liabilities"],
    )
    current_assets_previous = statement_value(
        data.balance_sheet,
        previous,
        ["Current Assets", "Total Current Assets"],
    )
    current_liabilities_previous = statement_value(
        data.balance_sheet,
        previous,
        ["Current Liabilities", "Total Current Liabilities"],
    )
    current_ratio = safe_divide(current_assets, current_liabilities, "current ratio")
    current_ratio_previous = safe_divide(
        current_assets_previous,
        current_liabilities_previous,
        "previous current ratio",
    )
    current_ratio_score = int(current_ratio > current_ratio_previous)

    shares = statement_value(
        data.balance_sheet,
        current,
        ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
    )
    shares_previous = statement_value(
        data.balance_sheet,
        previous,
        ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
    )
    dilution_score = int(shares <= shares_previous)

    return debt_score + current_ratio_score + dilution_score


def calculate_operational_efficiency(data) -> int:
    """Score gross margin and asset turnover improvements."""

    current, previous, two_years_ago = data.periods[0], data.periods[1], data.periods[2]

    sales = statement_value(data.income_statement, current, ["Total Revenue", "Revenue"])
    sales_previous = statement_value(data.income_statement, previous, ["Total Revenue", "Revenue"])
    cost_of_revenue = statement_value(data.income_statement, current, ["Cost Of Revenue", "Cost Of Goods Sold"])
    cost_of_revenue_previous = statement_value(
        data.income_statement,
        previous,
        ["Cost Of Revenue", "Cost Of Goods Sold"],
    )

    gross_margin = safe_divide(sales - cost_of_revenue, sales, "current gross margin")
    gross_margin_previous = safe_divide(
        sales_previous - cost_of_revenue_previous,
        sales_previous,
        "previous gross margin",
    )
    gross_margin_score = int(gross_margin > gross_margin_previous)

    total_assets = statement_value(data.balance_sheet, current, "Total Assets")
    total_assets_previous = statement_value(data.balance_sheet, previous, "Total Assets")
    total_assets_two_years_ago = statement_value(data.balance_sheet, two_years_ago, "Total Assets")
    average_assets = (total_assets + total_assets_previous) / 2
    average_assets_previous = (total_assets_previous + total_assets_two_years_ago) / 2
    asset_turnover = safe_divide(sales, average_assets, "current asset turnover")
    asset_turnover_previous = safe_divide(
        sales_previous,
        average_assets_previous,
        "previous asset turnover",
    )
    asset_turnover_score = int(asset_turnover > asset_turnover_previous)

    return gross_margin_score + asset_turnover_score


def calculate_piotroski_score(ticker: str) -> int:
    """Calculate the full 0-9 Piotroski F-score for one ticker."""

    data = fetch_financial_data(ticker)
    require_periods(data, 3)
    return (
        calculate_profitability(data)
        + calculate_leverage_liquidity_and_dilution(data)
        + calculate_operational_efficiency(data)
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line options for batch or single-ticker scoring."""

    parser = argparse.ArgumentParser(description="Calculate Piotroski F-scores.")
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

    results: dict[str, int] = {}
    errors: dict[str, str] = {}

    # Continue through the list even if a company is missing a required field.
    for ticker in tickers:
        try:
            f_score = calculate_piotroski_score(ticker)
            results[ticker] = f_score
            print(f"{ticker}: Piotroski F-score = {f_score}")
        except Exception as exc:
            errors[ticker] = str(exc)
            print(f"{ticker}: failed ({exc})")

    print("\nPiotroski F-score results:")
    for ticker, f_score in results.items():
        print(f"{ticker}: {f_score}")

    if errors:
        print("\nErrors:")
        for ticker, error in errors.items():
            print(f"{ticker}: {error}")


if __name__ == "__main__":
    main()
