import numpy as np
import pandas as pd
import datetime as dt
import yfinance as yf
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Fetch S&P 500 tickers dynamically
def fetch_sp500_tickers():
    """
    Fetch the list of S&P 500 companies from Wikipedia.

    Returns:
        list: A list of ticker symbols.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)[0]
    return sp500_table['Symbol'].tolist()

# Fetch adjusted closing prices for selected tickers
def fetch_price_data(tickers, start_date, end_date):
    """
    Fetch historical adjusted closing prices for a list of tickers.

    Args:
        tickers (list): List of stock ticker symbols.
        start_date (datetime): Start date for historical data.
        end_date (datetime): End date for historical data.

    Returns:
        DataFrame: Adjusted closing prices of the stocks.
    """
    adj_close_df = pd.DataFrame()
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            adj_close_df[ticker] = data['Adj Close']
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
    adj_close_df.dropna(axis=1, inplace=True)  # Drop columns with missing data
    return adj_close_df


# Calculate portfolio metrics
def calculate_log_returns(adj_close_df):
    """
    Calculate log returns for a DataFrame of adjusted closing prices.

    Args:
        adj_close_df (DataFrame): Adjusted closing prices.

    Returns:
        DataFrame: Log returns of the stocks.
    """
    return np.log(adj_close_df / adj_close_df.shift(1)).dropna()

def portfolio_std_dev(weights, cov_matrix):
    variance = weights.T @ cov_matrix @ weights
    return np.sqrt(variance)

def portfolio_expected_return(weights, log_returns):
    return np.sum(log_returns.mean() * weights) * 252

def portfolio_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
    return (portfolio_expected_return(weights, log_returns) - risk_free_rate) / portfolio_std_dev(weights, cov_matrix)

# Select tickers dynamically based on criteria
def select_tickers(adj_close_df, log_returns, method='low_volatility', top_n=10):
    """
    Select tickers dynamically based on specified criteria.

    Args:
        adj_close_df (DataFrame): Adjusted closing prices.
        log_returns (DataFrame): Log returns of the stocks.
        method (str): Selection criteria ('low_volatility', 'high_return', 'random').
        top_n (int): Number of tickers to select.

    Returns:
        list: Selected tickers.
    """
    if method == 'low_volatility':
        volatility = log_returns.std() * np.sqrt(252)
        selected_tickers = volatility.nsmallest(top_n).index.tolist()
    elif method == 'high_return':
        mean_returns = log_returns.mean() * 252
        selected_tickers = mean_returns.nlargest(top_n).index.tolist()
    elif method == 'random':
        selected_tickers = adj_close_df.columns.to_list()
        selected_tickers = np.random.choice(selected_tickers, top_n, replace=False).tolist()
    else:
        raise ValueError("Invalid selection method. Choose 'low_volatility', 'high_return', or 'random'.")
    return selected_tickers

# Optimize portfolio
def optimize_portfolio(log_returns, cov_matrix, risk_free_rate, bounds=(0, 0.4)):
    num_assets = len(log_returns.columns)
    initial_weights = np.array([1 / num_assets] * num_assets)  # Equal weights initially

    # Constraints
    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}  # Weights must sum to 1
    bounds = [(bounds[0], bounds[1]) for _ in range(num_assets)]  # No short selling, max allocation per asset

    # Perform optimization to maximize Sharpe Ratio
    optimized_results = minimize(
        lambda weights: -portfolio_sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate),
        initial_weights,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )

    if not optimized_results.success:
        raise ValueError("Optimization failed: " + optimized_results.message)

    return optimized_results.x

# Main program
if __name__ == "__main__":
    # Step 1: Fetch S&P 500 tickers
    all_tickers = fetch_sp500_tickers()

    # Step 2: Fetch historical price data
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=5 * 365)  # 5 years of data
    adj_close_df = fetch_price_data(all_tickers, start_date, end_date)

    # Step 3: Calculate log returns
    log_returns = calculate_log_returns(adj_close_df)
    cov_matrix = log_returns.cov() * 252  # Annualized covariance matrix

    # Step 4: Dynamically select top tickers (e.g., based on low volatility)
    selected_tickers = select_tickers(adj_close_df, log_returns, method='low_volatility', top_n=10)
    print(f"Selected Tickers: {selected_tickers}")

    # Filter data for selected tickers
    log_returns = log_returns[selected_tickers]
    cov_matrix = log_returns.cov() * 252

    # Step 5: Optimize portfolio
    risk_free_rate = 0.02
    optimal_weights = optimize_portfolio(log_returns, cov_matrix, risk_free_rate, bounds=(0, 0.4))

    # Calculate portfolio metrics
    optimal_portfolio_return = portfolio_expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = portfolio_std_dev(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = portfolio_sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

    # Step 6: Print and visualize results
    print("\nOptimal Portfolio Weights:")
    for ticker, weight in zip(selected_tickers, optimal_weights):
        print(f"{ticker}: {weight:.4f}")

    print(f"\nExpected Annual Return: {optimal_portfolio_return:.4f}")
    print(f"Expected Volatility: {optimal_portfolio_volatility:.4f}")
    print(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

    # Plot optimal weights
    plt.figure(figsize=(10, 6))
    plt.bar(selected_tickers, optimal_weights)
    plt.xlabel('Assets')
    plt.ylabel('Optimal Weights')
    plt.title('Optimal Portfolio Weights')
    plt.show()
