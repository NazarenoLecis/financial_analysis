import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (
    fetch_financial_data,
    market_cap_from_stock,
    safe_divide,
    shares_outstanding_from_stock,
    statement_value,
)


def latest_free_cash_flow(data) -> float:
    """Read free cash flow directly, or estimate it from operating cash flow and capex."""

    period = data.periods[0]
    free_cash_flow = statement_value(data.cash_flow, period, "Free Cash Flow", required=False)
    if free_cash_flow is not None:
        return free_cash_flow

    operating_cash_flow = statement_value(
        data.cash_flow,
        period,
        ["Operating Cash Flow", "Total Cash From Operating Activities"],
    )
    capital_expenditure = statement_value(data.cash_flow, period, "Capital Expenditure")
    return operating_cash_flow + capital_expenditure


def run_dcf(
    ticker: str,
    growth_rate: float,
    discount_rate: float,
    terminal_growth_rate: float,
    years: int,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Project free cash flow and estimate intrinsic value per share."""

    if terminal_growth_rate >= discount_rate:
        raise ValueError("terminal_growth_rate must be lower than discount_rate")

    data = fetch_financial_data(ticker)
    starting_fcf = latest_free_cash_flow(data)

    rows = []
    for year in range(1, years + 1):
        projected_fcf = starting_fcf * (1 + growth_rate) ** year
        present_value = projected_fcf / (1 + discount_rate) ** year
        rows.append({"year": year, "projected_fcf": projected_fcf, "present_value": present_value})

    projection = pd.DataFrame(rows)
    terminal_fcf = projection.iloc[-1]["projected_fcf"] * (1 + terminal_growth_rate)
    terminal_value = terminal_fcf / (discount_rate - terminal_growth_rate)
    terminal_present_value = terminal_value / (1 + discount_rate) ** years
    equity_value = projection["present_value"].sum() + terminal_present_value
    shares_outstanding = shares_outstanding_from_stock(data.stock, data.ticker)
    intrinsic_value_per_share = safe_divide(equity_value, shares_outstanding, "intrinsic value per share")
    market_cap = market_cap_from_stock(data.stock, data.ticker)

    summary = {
        "starting_fcf": starting_fcf,
        "terminal_value": terminal_value,
        "terminal_present_value": terminal_present_value,
        "equity_value": equity_value,
        "shares_outstanding": shares_outstanding,
        "intrinsic_value_per_share": intrinsic_value_per_share,
        "market_cap": market_cap,
        "upside_to_market_cap": safe_divide(equity_value - market_cap, market_cap, "upside to market cap"),
    }
    return projection, summary


def plot_dcf(ticker: str, projection: pd.DataFrame) -> None:
    """Show projected and discounted free cash flows."""

    projection.set_index("year")[["projected_fcf", "present_value"]].plot(kind="bar", figsize=(10, 6))
    plt.title(f"{ticker} DCF Projection")
    plt.xlabel("Projection Year")
    plt.ylabel("USD")
    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple discounted cash flow valuation.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--growth-rate", type=float, default=0.05)
    parser.add_argument("--discount-rate", type=float, default=0.10)
    parser.add_argument("--terminal-growth-rate", type=float, default=0.025)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    projection, summary = run_dcf(
        args.ticker,
        args.growth_rate,
        args.discount_rate,
        args.terminal_growth_rate,
        args.years,
    )
    print(projection.to_string(index=False, float_format=lambda value: f"{value:,.0f}"))
    print("\nSummary:")
    for label, value in summary.items():
        print(f"{label}: {value:,.4f}")

    if not args.no_plot:
        plot_dcf(args.ticker, projection)


if __name__ == "__main__":
    main()
