"""Beneish M-score analysis.

The Beneish M-score is designed to flag possible earnings manipulation. It does
not prove fraud. It highlights accounting patterns that deserve further review.

The model compares the most recent fiscal year with the prior year across eight
variables, such as receivables, gross margin, asset quality, sales growth, and
accruals.
"""

import argparse
import sys
from pathlib import Path

# Allow this script to import utils.py when it is run directly from VS Code.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import fetch_financial_data, fetch_index_tickers, require_periods, safe_divide, statement_value


def calculate_beneish_m_score(ticker: str) -> float:
    """Calculate the eight-variable Beneish M-score for earnings manipulation risk."""

    data = fetch_financial_data(ticker)

    # Beneish needs two years because most variables are year-over-year indexes.
    require_periods(data, 2)
    current, previous = data.periods[0], data.periods[1]

    # The Beneish model compares current-year ratios with prior-year ratios.
    # Large changes in these ratios can indicate aggressive accounting.
    receivables = statement_value(data.balance_sheet, current, ["Accounts Receivable", "Net Receivables"])
    receivables_previous = statement_value(data.balance_sheet, previous, ["Accounts Receivable", "Net Receivables"])
    sales = statement_value(data.income_statement, current, ["Total Revenue", "Revenue"])
    sales_previous = statement_value(data.income_statement, previous, ["Total Revenue", "Revenue"])
    cost_of_revenue = statement_value(data.income_statement, current, ["Cost Of Revenue", "Cost Of Goods Sold"])
    cost_of_revenue_previous = statement_value(
        data.income_statement,
        previous,
        ["Cost Of Revenue", "Cost Of Goods Sold"],
    )

    total_assets = statement_value(data.balance_sheet, current, "Total Assets")
    total_assets_previous = statement_value(data.balance_sheet, previous, "Total Assets")
    current_assets = statement_value(data.balance_sheet, current, ["Current Assets", "Total Current Assets"])
    current_assets_previous = statement_value(
        data.balance_sheet,
        previous,
        ["Current Assets", "Total Current Assets"],
    )
    ppe = statement_value(data.balance_sheet, current, ["Net PPE", "Property Plant Equipment"])
    ppe_previous = statement_value(data.balance_sheet, previous, ["Net PPE", "Property Plant Equipment"])

    depreciation = depreciation_value(data, current)
    depreciation_previous = depreciation_value(data, previous)
    sga_expense = statement_value(
        data.income_statement,
        current,
        ["Selling General Administrative", "Selling General And Administration", "SG&A Expense"],
    )
    sga_expense_previous = statement_value(
        data.income_statement,
        previous,
        ["Selling General Administrative", "Selling General And Administration", "SG&A Expense"],
    )
    total_liabilities = statement_value(
        data.balance_sheet,
        current,
        ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"],
    )
    total_liabilities_previous = statement_value(
        data.balance_sheet,
        previous,
        ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"],
    )
    net_income = statement_value(data.income_statement, current, "Net Income")
    operating_cash_flow = statement_value(
        data.cash_flow,
        current,
        ["Operating Cash Flow", "Total Cash From Operating Activities"],
    )

    # DSRI: Days Sales in Receivables Index. If receivables grow faster than
    # sales, revenue recognition may deserve extra scrutiny.
    dsri = safe_divide(
        safe_divide(receivables, sales, "current receivables / sales"),
        safe_divide(receivables_previous, sales_previous, "previous receivables / sales"),
        "DSRI",
    )
    # GMI: Gross Margin Index. A weakening margin can create pressure to manage
    # earnings.
    gross_margin = safe_divide(sales - cost_of_revenue, sales, "current gross margin")
    gross_margin_previous = safe_divide(
        sales_previous - cost_of_revenue_previous,
        sales_previous,
        "previous gross margin",
    )
    gmi = safe_divide(gross_margin_previous, gross_margin, "GMI")

    # AQI: Asset Quality Index. This looks at assets other than current assets
    # and PPE. A rise can suggest more intangible or less certain assets.
    asset_quality = 1 - safe_divide(current_assets + ppe, total_assets, "current asset quality")
    asset_quality_previous = 1 - safe_divide(
        current_assets_previous + ppe_previous,
        total_assets_previous,
        "previous asset quality",
    )
    aqi = safe_divide(asset_quality, asset_quality_previous, "AQI")

    # SGI: Sales Growth Index. Fast-growing firms can face pressure to keep
    # growth looking strong.
    sgi = safe_divide(sales, sales_previous, "SGI")

    # DEPI: Depreciation Index. A change can indicate a longer assumed useful
    # life of assets, which lowers depreciation expense.
    depreciation_rate = safe_divide(depreciation, depreciation + ppe, "current depreciation rate")
    depreciation_rate_previous = safe_divide(
        depreciation_previous,
        depreciation_previous + ppe_previous,
        "previous depreciation rate",
    )
    depi = safe_divide(depreciation_rate_previous, depreciation_rate, "DEPI")
    # SGAI: Sales, General, and Administrative Expense Index. Rising SG&A
    # relative to sales can signal declining efficiency.
    sgai = safe_divide(
        safe_divide(sga_expense, sales, "current SGA / sales"),
        safe_divide(sga_expense_previous, sales_previous, "previous SGA / sales"),
        "SGAI",
    )
    # LVGI: Leverage Index. Higher leverage can increase earnings-management
    # incentives because debt covenants may become more important.
    lvgi = safe_divide(
        safe_divide(total_liabilities, total_assets, "current leverage"),
        safe_divide(total_liabilities_previous, total_assets_previous, "previous leverage"),
        "LVGI",
    )
    # TATA: Total Accruals to Total Assets. Higher accruals mean earnings are
    # less supported by operating cash flow.
    tata = safe_divide(net_income - operating_cash_flow, total_assets, "TATA")

    # Beneish's published eight-variable formula. A common threshold is around
    # -1.78: scores above that are often considered higher manipulation risk.
    return (
        -4.84
        + 0.92 * dsri
        + 0.528 * gmi
        + 0.404 * aqi
        + 0.892 * sgi
        + 0.115 * depi
        - 0.172 * sgai
        + 4.679 * tata
        - 0.327 * lvgi
    )


def depreciation_value(data, period) -> float:
    """Read depreciation from cash flow first, then income statement as a fallback."""

    try:
        # Cash flow statements usually report depreciation and amortization
        # together as a non-cash charge.
        return statement_value(data.cash_flow, period, ["Depreciation And Amortization", "Depreciation"])
    except Exception:
        return statement_value(data.income_statement, period, ["Depreciation And Amortization", "Depreciation"])


def parse_args() -> argparse.Namespace:
    """Parse command-line options for batch or single-ticker scoring."""

    parser = argparse.ArgumentParser(description="Calculate Beneish M-scores.")
    parser.add_argument("--index", choices=["sp500", "nasdaq100"], default="sp500")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols to analyze. Overrides --index.")
    parser.add_argument("--limit", type=int, help="Limit the number of tickers processed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # With no arguments, analyze AAPL so VS Code direct-run works. Passing
    # --index or --limit switches the script into index-batch mode.
    use_index = args.tickers is None and ("--index" in sys.argv or "--limit" in sys.argv)
    tickers = fetch_index_tickers(args.index) if use_index else (args.tickers or ["AAPL"])
    if args.limit:
        tickers = tickers[: args.limit]

    results: dict[str, float] = {}
    errors: dict[str, str] = {}

    # Missing fields are reported per ticker instead of stopping the full run.
    for ticker in tickers:
        try:
            m_score = calculate_beneish_m_score(ticker)
            results[ticker] = m_score
            print(f"{ticker}: Beneish M-score = {m_score:.4f}")
        except Exception as exc:
            errors[ticker] = str(exc)
            print(f"{ticker}: failed ({exc})")

    print("\nBeneish M-score results:")
    for ticker, m_score in results.items():
        print(f"{ticker}: {m_score:.4f}")

    if errors:
        print("\nErrors:")
        for ticker, error in errors.items():
            print(f"{ticker}: {error}")


if __name__ == "__main__":
    main()
