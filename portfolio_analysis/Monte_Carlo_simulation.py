import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from utils import extract_close_prices, fetch_index_tickers, yahoo_symbol


def run_simulation(
    tickers: list[str],
    start_date: str,
    end_date: str,
    simulations: int,
    risk_free_rate: float,
    seed: int,
) -> pd.DataFrame:
    """Generate random long-only portfolios and return their risk/return metrics."""

    yahoo_tickers = [yahoo_symbol(ticker) for ticker in tickers]
    downloaded = yf.download(yahoo_tickers, start=start_date, end=end_date, progress=False)
    prices = extract_close_prices(downloaded, yahoo_tickers).sort_index()
    returns = prices.pct_change(fill_method=None).dropna()

    if returns.empty:
        raise ValueError("Not enough price data to calculate returns")

    rng = np.random.default_rng(seed)
    num_stocks = len(prices.columns)
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()
    results = np.zeros((3 + num_stocks, simulations))

    # Each simulation creates weights that sum to 1, then annualizes return and volatility.
    for i in range(simulations):
        weights = rng.random(num_stocks)
        weights /= np.sum(weights)

        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(weights.T @ cov_matrix @ weights) * np.sqrt(252)
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe_ratio
        results[3:, i] = weights

    columns = ["return", "volatility", "sharpe"] + prices.columns.tolist()
    return pd.DataFrame(results.T, columns=columns)


def plot_results(results: pd.DataFrame) -> None:
    """Plot the simulated portfolios and highlight the best Sharpe/min-volatility portfolios."""

    max_sharpe_port = results.iloc[results["sharpe"].idxmax()]
    min_vol_port = results.iloc[results["volatility"].idxmin()]

    plt.scatter(results.volatility, results["return"], c=results.sharpe, cmap="RdYlBu")
    plt.xlabel("Volatility")
    plt.ylabel("Returns")
    plt.colorbar(label="Sharpe Ratio")
    plt.scatter(
        max_sharpe_port["volatility"],
        max_sharpe_port["return"],
        marker=(5, 1, 0),
        color="r",
        s=100,
        label="Max Sharpe Ratio",
    )
    plt.scatter(
        min_vol_port["volatility"],
        min_vol_port["return"],
        marker=(5, 1, 0),
        color="g",
        s=100,
        label="Min Volatility",
    )
    plt.legend()
    plt.show()


def parse_args() -> argparse.Namespace:
    """Parse command-line options for reproducible portfolio simulations."""

    parser = argparse.ArgumentParser(description="Run a Monte Carlo portfolio simulation.")
    parser.add_argument("--index", choices=["sp500", "nasdaq100"], default="sp500")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols to analyze. Overrides random selection.")
    parser.add_argument("--num-stocks", type=int, default=4, help="Random stock count when --tickers is omitted.")
    parser.add_argument("--start-date", default="2010-01-01")
    parser.add_argument("--end-date", default=datetime.today().strftime("%Y-%m-%d"))
    parser.add_argument("--simulations", type=int, default=25000)
    parser.add_argument("--risk-free-rate", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tickers:
        tickers = args.tickers
    elif "--index" in sys.argv:
        # Choose a reproducible random subset when no explicit tickers are provided.
        rng = np.random.default_rng(args.seed)
        tickers = rng.choice(fetch_index_tickers(args.index), args.num_stocks, replace=False).tolist()
    else:
        tickers = ["AAPL", "MSFT", "NVDA", "GOOGL"][: args.num_stocks]

    results = run_simulation(
        tickers=tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        simulations=args.simulations,
        risk_free_rate=args.risk_free_rate,
        seed=args.seed,
    )
    max_sharpe_port = results.iloc[results["sharpe"].idxmax()]
    min_vol_port = results.iloc[results["volatility"].idxmin()]

    print("Selected stocks:", ", ".join(tickers))
    print("\nPortfolio with maximum Sharpe ratio:")
    print(max_sharpe_port)
    print("\nPortfolio with minimum volatility:")
    print(min_vol_port)

    if not args.no_plot:
        plot_results(results)


if __name__ == "__main__":
    main()
