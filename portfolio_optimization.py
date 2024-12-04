import numpy as np
import pandas as pd
import yfinance as yf  # For fetching stock data
import matplotlib.pyplot as plt  # For plotting results
from datetime import datetime  # For handling dates

# Fetch the list of S&P 500 companies from Wikipedia
def fetch_sp500_tickers():
    """
    Fetch the list of S&P 500 companies from Wikipedia.

    Returns:
        list: A list of ticker symbols.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    sp500_table = pd.read_html(url)[0]  # Use pandas to read the table
    return sp500_table['Symbol'].tolist()  # Extract the 'Symbol' column as a list

# Fetch historical adjusted closing prices for a list of tickers
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
    adj_close_df = pd.DataFrame()  # Initialize an empty DataFrame
    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            adj_close_df[ticker] = data['Adj Close']  # Append 'Adj Close' to the DataFrame
        except Exception as e:
            print(f"Failed to fetch data for {ticker}: {e}")
    adj_close_df.dropna(axis=1, inplace=True)  # Drop columns with missing data
    return adj_close_df

# Monte Carlo simulation for portfolio optimization
def simulate_portfolios(returns, mean_daily_returns, cov_matrix, num_portfolios=25000, risk_free_rate=0.02):
    """
    Simulate multiple portfolios to determine their returns, volatility, and Sharpe Ratio.

    Args:
        returns (DataFrame): Daily returns of stocks.
        mean_daily_returns (Series): Mean daily returns of stocks.
        cov_matrix (DataFrame): Covariance matrix of returns.
        num_portfolios (int): Number of portfolios to simulate.
        risk_free_rate (float): Risk-free rate for Sharpe Ratio calculation.

    Returns:
        np.ndarray: Results of the simulation (returns, volatility, Sharpe Ratio, weights).
    """
    num_stocks = len(mean_daily_returns)  # Number of stocks in the portfolio
    results = np.zeros((3 + num_stocks, num_portfolios))  # Initialize results array

    # Monte Carlo Simulation:
    # Randomly generate weights for stocks in the portfolio and calculate portfolio performance.
    for i in range(num_portfolios):
        weights = np.random.random(num_stocks)  # Generate random weights
        weights /= np.sum(weights)  # Normalize weights to sum to 1

        # Portfolio annualized return = sum(weight * daily mean return * trading days)
        portfolio_return = np.sum(mean_daily_returns * weights) * 252

        # Portfolio volatility = sqrt(weights^T * covariance matrix * weights) * sqrt(trading days)
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

        # Sharpe Ratio = (portfolio return - risk-free rate) / portfolio volatility
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std_dev

        # Store results
        results[0, i] = portfolio_return
        results[1, i] = portfolio_std_dev
        results[2, i] = sharpe_ratio
        results[3:, i] = weights  # Store weights for this portfolio

    return results

# Convert simulation results into a DataFrame
def results_to_dataframe(results, stocks):
    """
    Convert simulation results into a DataFrame for easier analysis.

    Args:
        results (np.ndarray): Results of the Monte Carlo simulation.
        stocks (list): List of stock tickers.

    Returns:
        DataFrame: Results DataFrame with returns, volatility, Sharpe Ratio, and weights.
    """
    columns = ['ret', 'stdev', 'sharpe'] + stocks  # Columns for returns, volatility, Sharpe Ratio, and weights
    return pd.DataFrame(results.T, columns=columns)

# Identify portfolios with the highest Sharpe Ratio and minimum volatility
def identify_optimal_portfolios(results_frame):
    """
    Identify the optimal portfolios: one with the maximum Sharpe Ratio and one with the minimum volatility.

    Args:
        results_frame (DataFrame): Results DataFrame.

    Returns:
        tuple: DataFrames for the maximum Sharpe Ratio and minimum volatility portfolios.
    """
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]
    return max_sharpe_port, min_vol_port

# Plot portfolio results
def plot_portfolios(results_frame, max_sharpe_port, min_vol_port):
    """
    Plot portfolios based on volatility and returns, with color indicating Sharpe Ratio.

    Args:
        results_frame (DataFrame): Results DataFrame.
        max_sharpe_port (DataFrame): Maximum Sharpe Ratio portfolio.
        min_vol_port (DataFrame): Minimum volatility portfolio.
    """
    plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, cmap='RdYlBu')
    plt.xlabel('Volatility (Risk)')
    plt.ylabel('Returns')
    plt.colorbar(label='Sharpe Ratio')
    plt.xlim(left=0)
    plt.ylim(bottom=0)

    # Highlight the maximum Sharpe Ratio portfolio
    plt.scatter(max_sharpe_port['stdev'], max_sharpe_port['ret'], marker=(5, 1, 0), color='r', s=100, label='Max Sharpe Ratio')

    # Highlight the minimum volatility portfolio
    plt.scatter(min_vol_port['stdev'], min_vol_port['ret'], marker=(5, 1, 0), color='g', s=100, label='Min Volatility')

    plt.legend()
    plt.show()

# Script execution starts here
if __name__ == "__main__":
    # Step 1: Fetch S&P 500 stock tickers
    sp500_tickers = fetch_sp500_tickers()

    # Step 2: Allow the user to specify the number of stocks to include
    try:
        num_stocks = int(input("Enter the number of stocks for the portfolio (default 4): ") or 4)
    except ValueError:
        print("Invalid input. Using default of 4 stocks.")
        num_stocks = 4

    # Step 3: Randomly select stocks from the S&P 500
    np.random.seed(42)
    selected_tickers = np.random.choice(sp500_tickers, num_stocks, replace=False).tolist()
    print(f"Selected Tickers: {selected_tickers}")

    # Step 4: Fetch historical price data
    start_date = '2010-01-01'
    end_date = datetime.today().strftime('%Y-%m-%d')
    data = fetch_price_data(selected_tickers, start_date, end_date)

    # Step 5: Calculate daily returns, mean returns, and covariance matrix
    returns = data.pct_change()
    mean_daily_returns = returns.mean()
    cov_matrix = returns.cov()

    # Step 6: Perform Monte Carlo simulation to generate portfolios
    results = simulate_portfolios(returns, mean_daily_returns, cov_matrix)
    results_frame = results_to_dataframe(results, selected_tickers)

    # Step 7: Identify optimal portfolios
    max_sharpe_port, min_vol_port = identify_optimal_portfolios(results_frame)

    # Step 8: Display the results
    print("\nPortfolio with Maximum Sharpe Ratio:")
    print(max_sharpe_port)
    print("\nPortfolio with Minimum Volatility:")
    print(min_vol_port)

    # Step 9: Plot the portfolios
    plot_portfolios(results_frame, max_sharpe_port, min_vol_port)
