import argparse
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from utils import extract_close_prices, fetch_index_tickers, yahoo_symbol


def fetch_price_data(tickers: list[str], start_date: dt.datetime, end_date: dt.datetime) -> pd.DataFrame:
    """Download close prices for a group of tickers in one yfinance request."""

    yahoo_tickers = [yahoo_symbol(ticker) for ticker in tickers]
    downloaded = yf.download(yahoo_tickers, start=start_date, end=end_date, progress=False)
    prices = extract_close_prices(downloaded, yahoo_tickers)
    return prices.dropna(axis=1, how="any")


def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert price levels into daily log returns."""

    return np.log(prices / prices.shift(1)).dropna()


def portfolio_std_dev(weights: np.ndarray, cov_matrix: pd.DataFrame) -> float:
    """Calculate annualized portfolio volatility from weights and covariance."""

    variance = weights.T @ cov_matrix @ weights
    return float(np.sqrt(variance))


def portfolio_expected_return(weights: np.ndarray, log_returns: pd.DataFrame) -> float:
    """Calculate annualized expected return from historical mean log returns."""

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

    if top_n > len(log_returns.columns):
        raise ValueError(f"top_n={top_n} exceeds available assets ({len(log_returns.columns)})")

    if method == "low_volatility":
        volatility = log_returns.std() * np.sqrt(252)
        return volatility.nsmallest(top_n).index.tolist()
    if method == "high_return":
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
        from scipy.optimize import minimize
    except ImportError:
        return monte_carlo_best_weights(log_returns, cov_matrix, risk_free_rate, weight_bounds, fallback_simulations), "monte_carlo"

    num_assets = len(log_returns.columns)
    initial_weights = np.array([1 / num_assets] * num_assets)
    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = [weight_bounds for _ in range(num_assets)]

    optimized_results = minimize(
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

    rng = np.random.default_rng(42)
    num_assets = len(log_returns.columns)
    lower, upper = weight_bounds
    best_weights = np.array([1 / num_assets] * num_assets)
    best_sharpe = portfolio_sharpe_ratio(best_weights, log_returns, cov_matrix, risk_free_rate)

    for _ in range(simulations):
        weights = rng.random(num_assets)
        weights /= weights.sum()
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
    all_tickers = args.tickers or fetch_index_tickers(args.index)
    if args.limit:
        all_tickers = all_tickers[: args.limit]

    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=args.years * 365)
    prices = fetch_price_data(all_tickers, start_date, end_date)
    log_returns = calculate_log_returns(prices)

    # Explicit tickers are used directly; otherwise select the best candidates from the index.
    selected_tickers = [yahoo_symbol(ticker) for ticker in args.tickers] if args.tickers else select_tickers(
        log_returns,
        method=args.selection,
        top_n=args.top_n,
    )
    log_returns = log_returns[selected_tickers]
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
        plt.figure(figsize=(10, 6))
        plt.bar(selected_tickers, optimal_weights)
        plt.xlabel("Assets")
        plt.ylabel("Optimal Weights")
        plt.title("Optimal Portfolio Weights")
        plt.show()


if __name__ == "__main__":
    main()
