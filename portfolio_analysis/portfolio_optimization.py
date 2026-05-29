"""Portfolio optimization.

This script searches for portfolio weights that maximize the Sharpe ratio. The
Sharpe ratio compares expected return with volatility after subtracting the
risk-free rate.

If SciPy is installed, the script uses a numerical optimizer. If SciPy is not
available, it falls back to a Monte Carlo random search so the script still
runs.
"""

import argparse
import datetime as dt
import sys
from pathlib import Path

# Add project root to import path so direct file execution can import utils.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from utils import extract_close_prices, fetch_index_tickers, yahoo_symbol


def fetch_price_data(tickers: list[str], start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    """Download close prices for a group of tickers in one yfinance request."""

    # Convert symbols such as BRK.B to Yahoo Finance format before downloading.
    yahoo_tickers = [yahoo_symbol(ticker) for ticker in tickers]
    downloaded = yf.download(yahoo_tickers, start=start_date, end=end_date, progress=False)
    prices = extract_close_prices(downloaded, yahoo_tickers)
    # Drop tickers with missing price history so optimization uses aligned data.
    return prices.dropna(axis=1, how="any")


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price levels into daily log returns."""

    # Log returns are commonly used in portfolio math because they add cleanly
    # over time and behave well in statistical calculations.
    return np.log(prices / prices.shift(1)).dropna()


def portfolio_std_dev(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Calculate annualized portfolio volatility from weights and covariance."""

    # Matrix multiplication combines individual asset risk with how assets move
    # together.
    variance = weights.T @ cov_matrix @ weights
    return float(np.sqrt(variance))


def portfolio_expected_return(weights: np.ndarray, log_returns: pd.DataFrame) -> float:
    """Calculate annualized expected return from historical mean log returns."""

    # Multiply each asset's average return by its portfolio weight, then
    # annualize with 252 trading days.
    return float(np.sum(log_returns.mean() * weights) * 252)


def portfolio_sharpe_ratio(
    weights: np.ndarray,
    log_returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
) -> float:
    """Calculate the Sharpe ratio after subtracting the chosen risk-free rate."""

    volatility = portfolio_std_dev(weights, cov_matrix)
    return (portfolio_expected_return(weights, log_returns) - risk_free_rate) / volatility


def select_tickers(log_returns: pd.DataFrame, method: str = "low_volatility", top_n: int = 10) -> list[str]:
    """Choose a smaller investable universe before optimizing the portfolio."""

    # If there are fewer assets than requested, use all available assets.
    if top_n > len(log_returns.columns):
        top_n = len(log_returns.columns)

    if method == "low_volatility":
        # Pick stocks with the smallest annualized volatility.
        volatility = log_returns.std() * np.sqrt(252)
        return volatility.nsmallest(top_n).index.tolist()
    if method == "high_return":
        # Pick stocks with the largest historical annualized average return.
        mean_returns = log_returns.mean() * 252
        return mean_returns.nlargest(top_n).index.tolist()
    if method == "random":
        return np.random.choice(log_returns.columns.to_list(), top_n, replace=False).tolist()
    raise ValueError("Invalid selection method. Choose 'low_volatility', 'high_return', or 'random'.")


def optimize_portfolio(
    log_returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    weight_bounds: tuple[float, float] = (0, 0.4),
    fallback_simulations: int = 10000,
) -> tuple[np.ndarray, str]:
    """Optimize weights with SciPy when available, otherwise use a random search fallback."""

    try:
        # SciPy's SLSQP optimizer can handle equality constraints and bounds,
        # which is exactly what portfolio weights need.
        from scipy.optimize import minimize
    except ImportError:
        return monte_carlo_best_weights(log_returns, cov_matrix, risk_free_rate, weight_bounds, fallback_simulations), "monte_carlo"

    num_assets = len(log_returns.columns)

    # Start from equal weights. The optimizer then adjusts them.
    initial_weights = np.array([1 / num_assets] * num_assets)

    # Constraint: all weights must add up to 1, meaning 100 percent invested.
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}

    # Bounds: default is no short selling and max 40 percent in one asset.
    bounds = [weight_bounds for _ in range(num_assets)]

    optimized_results = minimize(
        # minimize tries to make a value smaller. We use negative Sharpe so
        # minimizing it is equivalent to maximizing the Sharpe ratio.
        lambda weights: -portfolio_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate),
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if optimized_results.success:
        return optimized_results.x, "scipy_slsqp"

    return monte_carlo_best_weights(log_returns, cov_matrix, risk_free_rate, weight_bounds, fallback_simulations), "monte_carlo"


def monte_carlo_best_weights(
    log_returns: pd.DataFrame,
    cov_matrix: pd.DataFrame,
    risk_free_rate: float,
    weight_bounds: tuple[float, float],
    simulations: int,
) -> np.ndarray:
    """Search random portfolios for the highest Sharpe ratio within weight bounds."""

    # Fallback method for environments without SciPy.
    rng = np.random.default_rng(42)
    num_assets = len(log_returns.columns)
    lower, upper = weight_bounds
    best_weights = np.array([1 / num_assets] * num_assets)
    best_sharpe = portfolio_sharpe_ratio(best_weights, log_returns, cov_matrix, risk_free_rate)

    for _ in range(simulations):
        # Generate a random long-only portfolio.
        weights = rng.random(num_assets)
        weights /= weights.sum()

        # Skip portfolios that violate min or max weight bounds.
        if np.any(weights < lower) or np.any(weights > upper):
            continue
        sharpe = portfolio_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = weights

    return best_weights


def parse_args() -> argparse.Namespace:
    """Parse command-line options for portfolio optimization."""

    parser = argparse.ArgumentParser(description="Optimize a long-only equity portfolio.")

    # Defaults make the script runnable from VS Code without arguments.
    parser.add_argument("--index", choices=["sp500", "nasdaq100"], default="sp500")
    parser.add_argument("--tickers", nargs="+", help="Ticker symbols to analyze. Overrides --index selection.")
    parser.add_argument("--selection", choices=["low_volatility", "high_return", "random"], default="low_volatility")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--years", type=int, default=5)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--min-weight", type=float, default=0.0)
    parser.add_argument("--max-weight", type=float, default=0.4)
    parser.add_argument("--limit", type=int, help="Limit index tickers before downloading prices.")
    parser.add_argument("--fallback-simulations", type=int, default=10000)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # With no arguments, run a small default portfolio. Passing --index or
    # --limit switches the script into index-selection mode.
    use_index = args.tickers is None and ("--index" in sys.argv or "--limit" in sys.argv)
    all_tickers = fetch_index_tickers(args.index) if use_index else (args.tickers or ["AAPL", "MSFT", "NVDA", "GOOGL"])
    if args.limit:
        all_tickers = all_tickers[: args.limit]

    # Use the requested number of years of daily price history.
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=args.years * 365)
    prices = fetch_price_data(all_tickers, start_date, end_date)
    log_returns = calculate_log_returns(prices)

    # Explicit tickers are used directly; otherwise select the best candidates from the index.
    if args.tickers or not use_index:
        selected_tickers = [yahoo_symbol(ticker) for ticker in all_tickers]
    else:
        selected_tickers = select_tickers(log_returns, method=args.selection, top_n=args.top_n)
    log_returns = log_returns[selected_tickers]

    # Annualize the covariance matrix because expected return is annualized too.
    cov_matrix = log_returns.cov() * 252

    optimal_weights, method = optimize_portfolio(
        log_returns,
        cov_matrix,
        args.risk_free_rate,
        weight_bounds=(args.min_weight, args.max_weight),
        fallback_simulations=args.fallback_simulations,
    )

    portfolio_return = portfolio_expected_return(optimal_weights, log_returns)
    volatility = portfolio_std_dev(optimal_weights, cov_matrix)
    sharpe_ratio = portfolio_sharpe_ratio(optimal_weights, log_returns, cov_matrix, args.risk_free_rate)

    print(f"Optimization method: {method}")
    print("\nOptimal portfolio weights:")
    for ticker, weight in zip(selected_tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")

    print(f"\nExpected annual return: {portfolio_return:.4f}")
    print(f"Expected volatility: {volatility:.4f}")
    print(f"Sharpe ratio: {sharpe_ratio:.4f}")

    if not args.no_plot:
        # Bar chart makes the final allocation easy to inspect.
        plt.figure(figsize=(10, 6))
        plt.bar(selected_tickers, optimal_weights)
        plt.xlabel("Assets")
        plt.ylabel("Optimal Weights")
        plt.title("Optimal Portfolio Weights")
        plt.show()


if __name__ == "__main__":
    main()
