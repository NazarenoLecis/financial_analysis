"""Quick stock snapshot.

This script gives a fast visual overview of one or more stocks. It is meant to
answer simple questions such as:

- which stock performed best over the selected period?
- which stock was more volatile?
- how large was the worst drawdown?

The chart normalizes every stock to 100 at the first available date. This makes
stocks with very different share prices comparable on the same chart. For
example, a move from 100 to 120 means a 20 percent gain, regardless of whether
the original stock price was 20 dollars or 500 dollars.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

# Add the project root to the import path so this file can be run directly from
# VS Code while still importing utils.py from the parent folder.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import RichHelpFormatter, extract_close_prices, yahoo_symbol


def build_stock_snapshot(
    tickers: list[str],
    start_date: str,
    end_date: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Download prices and calculate quick comparison metrics."""

    # Convert tickers to Yahoo Finance format before downloading. This handles
    # symbols such as BRK.B, which Yahoo writes as BRK-B.
    yahoo_tickers = [yahoo_symbol(ticker) for ticker in tickers]

    # Download all requested stocks in one request. This is faster than calling
    # yfinance separately for each ticker.
    downloaded = yf.download(
        yahoo_tickers,
        start=start_date,
        end=end_date,
        progress=False,
        auto_adjust=False,
    )
    prices = extract_close_prices(downloaded, yahoo_tickers).sort_index().ffill().dropna(how="all")

    if prices.empty:
        raise ValueError("No price data available for the selected tickers and dates")

    # Normalize each stock to 100 at its first available value. This turns the
    # chart into a percentage-performance comparison.
    first_prices = prices.apply(lambda column: column.dropna().iloc[0])
    normalized = prices.divide(first_prices) * 100

    daily_returns = prices.pct_change(fill_method=None).dropna(how="all")
    summary_rows = []

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        normalized_series = normalized[ticker].dropna()
        returns = daily_returns[ticker].dropna() if ticker in daily_returns else pd.Series(dtype=float)

        # Drawdown measures the fall from a previous high. A max drawdown of
        # -0.20 means the stock was down 20 percent from its prior peak at the
        # worst point in this period.
        drawdown = normalized_series / normalized_series.cummax() - 1

        summary_rows.append(
            {
                "ticker": ticker,
                "first_close": series.iloc[0],
                "latest_close": series.iloc[-1],
                "period_return": series.iloc[-1] / series.iloc[0] - 1,
                "annualized_volatility": returns.std() * np.sqrt(252) if not returns.empty else pd.NA,
                "max_drawdown": drawdown.min(),
                "period_high": series.max(),
                "period_low": series.min(),
            }
        )

    summary = pd.DataFrame(summary_rows)
    return prices, normalized, summary


def plot_snapshot(normalized: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Show normalized performance and period returns."""

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})

    # Top chart: each line starts at 100, so the lines show relative performance.
    normalized.plot(ax=axes[0])
    axes[0].set_title("Normalized Stock Performance")
    axes[0].set_ylabel("Start = 100")

    # Bottom chart: final percentage return for each ticker.
    summary.set_index("ticker")["period_return"].plot(kind="bar", ax=axes[1], color="steelblue")
    axes[1].set_title("Total Return Over Selected Period")
    axes[1].set_ylabel("Return")
    axes[1].axhline(0, color="black", linewidth=1)

    plt.tight_layout()
    plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a quick price-performance snapshot for one or more stocks.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  --tickers accepts one or more Yahoo Finance symbols, e.g. AAPL MSFT NVDA.
  --start-date and --end-date use YYYY-MM-DD format.
  Leave --end-date empty to use the latest available market data.

Data used:
  This is a price time-series view. It downloads historical close prices and
  normalizes every stock to 100 on the first date so performance is comparable.

Printed metrics:
  first_close, latest_close, total period return, annualized volatility,
  maximum drawdown, period high, and period low.

Examples:
  python quick_views/stock_snapshot.py
  python quick_views/stock_snapshot.py --tickers AAPL MSFT NVDA GOOGL
  python quick_views/stock_snapshot.py --tickers AAPL MSFT --start-date 2023-01-01 --no-plot
""",
    )

    # Defaults make the script useful by simply pressing Run Python File in VS Code.
    parser.add_argument("--tickers", nargs="+", default=["AAPL", "MSFT", "NVDA", "GOOGL"], help="One or more Yahoo Finance ticker symbols.")
    parser.add_argument("--start-date", default="2024-01-01", help="Start date for the price time series, in YYYY-MM-DD format.")
    parser.add_argument("--end-date", help="Optional end date for the price time series, in YYYY-MM-DD format.")
    parser.add_argument("--no-plot", action="store_true", help="Print output only and do not open a matplotlib chart.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, normalized, summary = build_stock_snapshot(args.tickers, args.start_date, args.end_date)

    print("Quick stock snapshot:")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))

    if not args.no_plot:
        plot_snapshot(normalized, summary)


if __name__ == "__main__":
    main()
