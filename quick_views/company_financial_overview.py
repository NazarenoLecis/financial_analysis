"""Company financial overview for quick, beginner-friendly inspection.

This module is intentionally broader than a normal "balance sheet summary".
It combines the most important annual income statement, cash flow statement,
and balance sheet lines, then calculates practical indicators such as margins,
ROA, ROE, liquidity ratios, leverage ratios, and turnover ratios.

The goal is not to replace the more specific fundamental-analysis scripts.
Instead, this file gives users a clean first look at one public company:

- important variables are shown in USD billions, so the unit is obvious;
- indicators are shown as percentages or simple ratios, so interpretation is
  easier than looking at raw accounting values;
- charts are split into related panels instead of mixing everything together.
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter

# Add the project root to the import path so this file can be run directly from
# VS Code while still importing utils.py from the parent folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import RichHelpFormatter, fetch_financial_data, statement_value


USD_BILLION = 1_000_000_000


@dataclass(frozen=True)
class CompanyFinancialOverview:
    """All data needed by the quick-view scripts and notebooks."""

    ticker: str
    variables: pd.DataFrame
    indicators: pd.DataFrame


# yfinance statement labels vary by company. Each variable below has a list of
# aliases so the code can still find the value when Yahoo uses a different row
# name for the same accounting concept.
VARIABLE_FIELDS = {
    # Income statement: performance during the year.
    "Revenue": ("income_statement", ["Total Revenue", "Revenue", "Operating Revenue"]),
    "Gross Profit": ("income_statement", "Gross Profit"),
    "Operating Income": ("income_statement", ["Operating Income", "EBIT"]),
    "EBITDA": ("income_statement", ["EBITDA", "Normalized EBITDA"]),
    "EBIT": ("income_statement", ["EBIT", "Operating Income"]),
    "Net Income": ("income_statement", "Net Income"),
    "R&D Expense": ("income_statement", ["Research And Development", "Research Development"]),
    "SG&A Expense": ("income_statement", ["Selling General And Administration", "Selling General Administrative"]),
    "Cost Of Revenue": ("income_statement", ["Cost Of Revenue", "Reconciled Cost Of Revenue", "Cost Of Goods Sold"]),

    # Cash flow statement: cash generated or consumed during the year.
    "Operating Cash Flow": ("cash_flow", ["Operating Cash Flow", "Total Cash From Operating Activities"]),
    "Capital Expenditure": ("cash_flow", ["Capital Expenditure", "Capital Expenditures"]),
    "Free Cash Flow": ("cash_flow", "Free Cash Flow"),

    # Balance sheet: what the company owns and owes at fiscal year-end.
    "Cash And Short Term Investments": (
        "balance_sheet",
        [
            "Cash Cash Equivalents And Short Term Investments",
            "Cash And Cash Equivalents",
            "Cash Cash Equivalents",
        ],
    ),
    "Receivables": ("balance_sheet", ["Receivables", "Accounts Receivable", "Net Receivables"]),
    "Inventory": ("balance_sheet", "Inventory"),
    "Current Assets": ("balance_sheet", ["Current Assets", "Total Current Assets"]),
    "Total Assets": ("balance_sheet", "Total Assets"),
    "Current Liabilities": ("balance_sheet", ["Current Liabilities", "Total Current Liabilities"]),
    "Short Term Debt": (
        "balance_sheet",
        [
            "Current Debt And Capital Lease Obligation",
            "Current Debt",
            "Short Term Debt",
            "Other Current Borrowings",
        ],
    ),
    "Long Term Debt": (
        "balance_sheet",
        ["Long Term Debt And Capital Lease Obligation", "Long Term Debt"],
    ),
    "Total Debt": ("balance_sheet", ["Total Debt", "Total Debt Net Minority Interest"]),
    "Net Debt": ("balance_sheet", "Net Debt"),
    "Total Liabilities": (
        "balance_sheet",
        ["Total Liabilities Net Minority Interest", "Total Liabilities", "Total Liab"],
    ),
    "Stockholders Equity": (
        "balance_sheet",
        ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
    ),
    "Invested Capital": ("balance_sheet", "Invested Capital"),
    "Working Capital": ("balance_sheet", "Working Capital"),
}


KEY_VARIABLE_GROUPS = {
    "Sales And Profit": ["Revenue", "Gross Profit", "Operating Income", "Net Income"],
    "Cash Flow": ["Operating Cash Flow", "Capital Expenditure", "Free Cash Flow"],
    "Debt And Cash": [
        "Cash And Short Term Investments",
        "Short Term Debt",
        "Long Term Debt",
        "Total Debt",
        "Net Debt",
    ],
    "Balance Sheet Size": ["Total Assets", "Total Liabilities", "Stockholders Equity", "Working Capital"],
}


PERCENT_INDICATORS = [
    "Gross Margin",
    "Operating Margin",
    "Net Margin",
    "ROA",
    "ROE",
    "ROI / ROIC Approx",
    "Working Capital To Revenue",
    "Debt To Assets",
    "Liabilities To Assets",
]


RATIO_INDICATORS = [
    "Current Ratio",
    "Quick Ratio",
    "Cash Ratio",
    "Debt To Equity",
    "Net Debt To EBITDA",
    "Asset Turnover",
    "Inventory Turnover",
    "Receivables Turnover",
]


INDICATOR_GROUPS = [
    ("Profit Margins", ["Gross Margin", "Operating Margin", "Net Margin"], "percent"),
    ("Returns", ["ROA", "ROE", "ROI / ROIC Approx"], "percent"),
    ("Liquidity Ratios", ["Current Ratio", "Quick Ratio", "Cash Ratio"], "ratio"),
    (
        "Debt And Liability Share",
        ["Debt To Assets", "Liabilities To Assets", "Working Capital To Revenue"],
        "percent",
    ),
    ("Debt Coverage", ["Debt To Equity", "Net Debt To EBITDA"], "ratio"),
    ("Efficiency Turnover", ["Asset Turnover", "Inventory Turnover", "Receivables Turnover"], "ratio"),
]


def build_company_financial_overview(ticker: str) -> CompanyFinancialOverview:
    """Collect annual statement variables and calculate annual indicators."""

    data = fetch_financial_data(ticker)
    periods = _combined_periods(
        data.balance_sheet,
        data.income_statement,
        data.cash_flow,
    )

    variables = pd.DataFrame(
        {
            label: _statement_series(getattr(data, statement_name), periods, aliases)
            for label, (statement_name, aliases) in VARIABLE_FIELDS.items()
        }
    )

    # Some statement providers do not report working capital directly. It is
    # current assets minus current liabilities, so calculate it when needed.
    if "Working Capital" in variables:
        calculated_working_capital = variables["Current Assets"] - variables["Current Liabilities"]
        variables["Working Capital"] = variables["Working Capital"].fillna(calculated_working_capital)

    # Free cash flow is commonly operating cash flow minus capital expenditure.
    # yfinance usually reports capital expenditure as a negative number, so add
    # it to operating cash flow when the direct Free Cash Flow line is missing.
    if "Free Cash Flow" in variables:
        calculated_fcf = variables["Operating Cash Flow"] + variables["Capital Expenditure"]
        variables["Free Cash Flow"] = variables["Free Cash Flow"].fillna(calculated_fcf)

    # If total debt is missing, build an approximate total from short-term and
    # long-term debt. This is useful because companies do not always report the
    # exact same balance-sheet row names.
    if "Total Debt" in variables:
        debt_parts_available = variables["Short Term Debt"].notna() | variables["Long Term Debt"].notna()
        calculated_total_debt = variables["Short Term Debt"].add(variables["Long Term Debt"], fill_value=0)
        calculated_total_debt = calculated_total_debt.where(debt_parts_available)
        variables["Total Debt"] = variables["Total Debt"].fillna(calculated_total_debt)

    # Net debt is debt minus cash. Negative net debt means the company has more
    # cash and short-term investments than debt.
    if "Net Debt" in variables:
        calculated_net_debt = variables["Total Debt"] - variables["Cash And Short Term Investments"]
        variables["Net Debt"] = variables["Net Debt"].fillna(calculated_net_debt)

    if "Invested Capital" in variables:
        calculated_invested_capital = (
            variables["Total Debt"]
            + variables["Stockholders Equity"]
            - variables["Cash And Short Term Investments"]
        )
        variables["Invested Capital"] = variables["Invested Capital"].fillna(calculated_invested_capital)

    variables = variables.dropna(axis=1, how="all").dropna(axis=0, how="all")
    indicators = calculate_indicators(variables)

    return CompanyFinancialOverview(ticker=ticker, variables=variables, indicators=indicators)


def calculate_indicators(variables: pd.DataFrame) -> pd.DataFrame:
    """Calculate common profitability, liquidity, leverage, and efficiency ratios."""

    revenue = _column(variables, "Revenue")
    gross_profit = _column(variables, "Gross Profit")
    operating_income = _column(variables, "Operating Income")
    ebit = _column(variables, "EBIT").fillna(operating_income)
    ebitda = _column(variables, "EBITDA")
    net_income = _column(variables, "Net Income")
    cost_of_revenue = _column(variables, "Cost Of Revenue")

    cash = _column(variables, "Cash And Short Term Investments")
    receivables = _column(variables, "Receivables")
    inventory = _column(variables, "Inventory")
    current_assets = _column(variables, "Current Assets")
    total_assets = _column(variables, "Total Assets")
    current_liabilities = _column(variables, "Current Liabilities")
    total_debt = _column(variables, "Total Debt")
    net_debt = _column(variables, "Net Debt")
    total_liabilities = _column(variables, "Total Liabilities")
    equity = _column(variables, "Stockholders Equity")
    invested_capital = _column(variables, "Invested Capital")
    working_capital = _column(variables, "Working Capital")

    indicators = pd.DataFrame(index=variables.index)

    # Profitability: how much profit is generated from sales, assets, equity,
    # and invested capital.
    indicators["Gross Margin"] = _safe_series_divide(gross_profit, revenue)
    indicators["Operating Margin"] = _safe_series_divide(operating_income, revenue)
    indicators["Net Margin"] = _safe_series_divide(net_income, revenue)
    indicators["ROA"] = _safe_series_divide(net_income, total_assets)
    indicators["ROE"] = _safe_series_divide(net_income, equity)
    indicators["ROI / ROIC Approx"] = _safe_series_divide(ebit, invested_capital)

    # Liquidity: ability to cover short-term obligations.
    indicators["Current Ratio"] = _safe_series_divide(current_assets, current_liabilities)
    indicators["Quick Ratio"] = _safe_series_divide(cash + receivables, current_liabilities)
    indicators["Cash Ratio"] = _safe_series_divide(cash, current_liabilities)
    indicators["Working Capital To Revenue"] = _safe_series_divide(working_capital, revenue)

    # Leverage: how much of the business is funded by debt or liabilities.
    indicators["Debt To Equity"] = _safe_series_divide(total_debt, equity)
    indicators["Debt To Assets"] = _safe_series_divide(total_debt, total_assets)
    indicators["Liabilities To Assets"] = _safe_series_divide(total_liabilities, total_assets)
    indicators["Net Debt To EBITDA"] = _safe_series_divide(net_debt, ebitda)

    # Efficiency: how effectively assets, inventory, and receivables support
    # sales. Higher turnover usually means the company uses that item faster.
    indicators["Asset Turnover"] = _safe_series_divide(revenue, total_assets)
    indicators["Inventory Turnover"] = _safe_series_divide(cost_of_revenue, inventory)
    indicators["Receivables Turnover"] = _safe_series_divide(revenue, receivables)

    return indicators.dropna(axis=1, how="all").dropna(axis=0, how="all")


def latest_values(frame: pd.DataFrame) -> pd.Series:
    """Return the latest non-empty row from a time-series DataFrame."""

    clean = frame.dropna(axis=0, how="all")
    if clean.empty:
        return pd.Series(dtype="Float64")
    return clean.iloc[-1].dropna()


def to_billions(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Scale raw USD values to USD billions for readable tables and charts."""

    return frame / USD_BILLION


def format_billions_frame(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return display-friendly text values in USD billions."""

    return to_billions(frame).map(lambda value: "" if pd.isna(value) else f"{value:,.1f}")


def format_indicator_frame(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Return display-friendly text values for indicators."""

    if isinstance(frame, pd.Series):
        return pd.Series(
            {
                label: _format_indicator_value(label, value)
                for label, value in frame.items()
            },
            name=frame.name,
        )

    formatted = frame.copy()
    for column in formatted.columns:
        formatted[column] = formatted[column].map(lambda value, label=column: _format_indicator_value(label, value))
    return formatted


def plot_key_variables(ticker: str, variables: pd.DataFrame, years: int = 4) -> None:
    """Plot important statement variables in clear USD-billions panels."""

    recent = variables.tail(years)
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    axes = axes.flatten()

    for axis, (title, columns) in zip(axes, KEY_VARIABLE_GROUPS.items()):
        _plot_money_panel(axis, recent, columns, title)

    fig.suptitle(f"{ticker} Key Financial Variables", fontsize=14)
    plt.tight_layout()
    plt.show()


def plot_indicators(ticker: str, indicators: pd.DataFrame, years: int = 4) -> None:
    """Plot financial indicators in groups that use clear ratio units."""

    recent = indicators.tail(years)
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for axis, (title, columns, unit) in zip(axes, INDICATOR_GROUPS):
        _plot_ratio_panel(axis, recent, columns, title, percent_axis=unit == "percent")

    fig.suptitle(f"{ticker} Financial Indicators", fontsize=14)
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show key financial variables and indicators for one public company.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  --ticker accepts one Yahoo Finance symbol, e.g. AAPL, MSFT, NVDA.
  --years controls how many recent annual fiscal periods are printed and plotted.
  --view accepts variables, indicators, or all.

Data used:
  Annual income statement, cash flow statement, and balance sheet values from
  yfinance. Variables are displayed in USD billions. Indicators are displayed
  as percentages or simple ratios depending on the metric.

Examples:
  python quick_views/company_financial_overview.py
  python quick_views/company_financial_overview.py --ticker MSFT --view variables
  python quick_views/company_financial_overview.py --ticker AAPL --view indicators --years 5 --no-plot
""",
    )
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker symbol.")
    parser.add_argument("--years", type=int, default=4, help="Number of recent annual periods to show.")
    parser.add_argument(
        "--view",
        choices=["variables", "indicators", "all"],
        default="all",
        help="Choose whether to show statement variables, financial indicators, or both.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Print output only and do not open matplotlib charts.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overview = build_company_financial_overview(args.ticker)

    if args.view in {"variables", "all"}:
        variables = overview.variables.tail(args.years)
        print(f"\n{args.ticker} key financial variables (USD billions):")
        print(to_billions(variables).to_string(float_format=lambda value: f"{value:,.1f}"))
        if not args.no_plot:
            plot_key_variables(args.ticker, overview.variables, args.years)

    if args.view in {"indicators", "all"}:
        indicators = overview.indicators.tail(args.years)
        print(f"\n{args.ticker} financial indicators:")
        print(indicators.to_string(float_format=lambda value: f"{value:.4f}"))
        if not args.no_plot:
            plot_indicators(args.ticker, overview.indicators, args.years)


def _combined_periods(*statements: pd.DataFrame) -> pd.DatetimeIndex:
    """Return all annual periods reported across the available statements."""

    periods = set()
    for statement in statements:
        periods.update(pd.to_datetime(statement.columns))
    return pd.DatetimeIndex(sorted(periods))


def _statement_series(
    statement: pd.DataFrame,
    periods: pd.DatetimeIndex,
    aliases: str | Iterable[str],
) -> pd.Series:
    """Read one statement line across all periods, returning missing when absent."""

    values = [
        statement_value(statement, period, aliases, required=False, default=pd.NA)
        for period in periods
    ]
    return pd.Series(values, index=periods, dtype="Float64")


def _column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric column or an all-missing series when it is unavailable."""

    if column in frame:
        return frame[column].astype("Float64")
    return pd.Series(pd.NA, index=frame.index, dtype="Float64")


def _safe_series_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divide two series while treating zero denominators as missing values."""

    clean_denominator = denominator.where(denominator != 0)
    return numerator / clean_denominator


def _format_indicator_value(label: str, value) -> str:
    """Format percentages and ratios differently for notebook display."""

    if pd.isna(value):
        return ""
    if label in PERCENT_INDICATORS:
        return f"{value:.2%}"
    return f"{value:.2f}x"


def _plot_money_panel(axis, frame: pd.DataFrame, columns: list[str], title: str) -> None:
    """Plot related accounting values after scaling them to USD billions."""

    available = _available_columns(frame, columns)
    if not available:
        _show_missing_panel(axis, title)
        return

    plot_frame = _with_year_index(to_billions(frame[available]))
    plot_frame.plot(ax=axis, marker="o")
    axis.set_title(title)
    axis.set_ylabel("USD billions")
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.grid(True, axis="y", alpha=0.25)


def _plot_ratio_panel(
    axis,
    frame: pd.DataFrame,
    columns: list[str],
    title: str,
    *,
    percent_axis: bool = False,
) -> None:
    """Plot related financial indicators with percentage or ratio units."""

    available = _available_columns(frame, columns)
    if not available:
        _show_missing_panel(axis, title)
        return

    plot_frame = _with_year_index(frame[available])
    plot_frame.plot(ax=axis, marker="o")
    axis.set_title(title)
    axis.set_ylabel("Percent" if percent_axis else "Ratio")
    if percent_axis:
        axis.yaxis.set_major_formatter(PercentFormatter(1.0))
    axis.axhline(0, color="black", linewidth=0.8)
    axis.grid(True, axis="y", alpha=0.25)


def _available_columns(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return columns that exist and have at least one non-missing value."""

    return [
        column
        for column in columns
        if column in frame and not frame[column].dropna().empty
    ]


def _with_year_index(frame: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Use fiscal years as x-axis labels instead of full timestamp strings."""

    result = frame.copy()
    result.index = [str(period.year) for period in result.index]
    return result


def _show_missing_panel(axis, title: str) -> None:
    """Keep a chart panel readable when a company has no data for that group."""

    axis.set_title(title)
    axis.text(0.5, 0.5, "No data available", ha="center", va="center")
    axis.set_axis_off()


if __name__ == "__main__":
    main()
