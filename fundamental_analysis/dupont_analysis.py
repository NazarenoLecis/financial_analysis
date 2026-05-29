import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import MissingFinancialField, fetch_financial_data, safe_divide, statement_value


def calculate_dupont(ticker: str) -> pd.DataFrame:
    """Break ROE into net margin, asset turnover, and equity multiplier by year."""

    data = fetch_financial_data(ticker)
    rows = []

    for period in data.periods:
        try:
            revenue = statement_value(data.income_statement, period, ["Total Revenue", "Revenue"])
            net_income = statement_value(data.income_statement, period, "Net Income")
            total_assets = statement_value(data.balance_sheet, period, "Total Assets")
            equity = statement_value(
                data.balance_sheet,
                period,
                ["Stockholders Equity", "Total Equity Gross Minority Interest", "Common Stock Equity"],
            )
        except MissingFinancialField:
            # Some yfinance statements expose an extra balance-sheet year without matching income data.
            continue

        net_margin = safe_divide(net_income, revenue, "net margin")
        asset_turnover = safe_divide(revenue, total_assets, "asset turnover")
        equity_multiplier = safe_divide(total_assets, equity, "equity multiplier")

        rows.append(
            {
                "period": pd.to_datetime(period),
                "net_margin": net_margin,
                "asset_turnover": asset_turnover,
                "equity_multiplier": equity_multiplier,
                "roe": net_margin * asset_turnover * equity_multiplier,
            }
        )

    if not rows:
        raise ValueError(f"No complete DuPont periods found for {ticker}")
    return pd.DataFrame(rows).sort_values("period")


def plot_dupont(ticker: str, frame: pd.DataFrame) -> None:
    """Show the DuPont components and resulting ROE over time."""

    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    frame.set_index("period")[["net_margin", "asset_turnover", "equity_multiplier"]].plot(
        ax=axes[0],
        marker="o",
    )
    axes[0].set_title(f"{ticker} DuPont Components")
    axes[0].set_ylabel("Ratio")

    frame.set_index("period")["roe"].plot(ax=axes[1], marker="o", color="black")
    axes[1].set_title("Return on Equity")
    axes[1].set_ylabel("ROE")

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a DuPont ROE decomposition.")
    parser.add_argument("--ticker", default="AAPL")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = calculate_dupont(args.ticker)
    print(frame.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if not args.no_plot:
        plot_dupont(args.ticker, frame)


if __name__ == "__main__":
    main()
