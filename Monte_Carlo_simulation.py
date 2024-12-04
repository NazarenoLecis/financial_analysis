import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime
import requests
from bs4 import BeautifulSoup


# Function to fetch the list of S&P 500 stocks from Wikipedia
def get_sp500_stocks():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})
    tickers = [row.find_all('td')[0].text.strip() for row in table.find_all('tr')[1:]]
    return tickers


# Fetch the list of S&P 500 stocks
sp500_stocks = get_sp500_stocks()

# Select a random subset of stocks (e.g., 4 stocks)
np.random.seed(42)  # For reproducibility
num_stocks = 4
stocks = np.random.choice(sp500_stocks, num_stocks, replace=False).tolist()

print(f"Selected Stocks: {stocks}")

# Download daily adjusted closing prices for selected stocks
start_date = '2010-01-01'
end_date = datetime.today().strftime('%Y-%m-%d')
data = yf.download(stocks, start=start_date, end=end_date)['Adj Close']
data.sort_index(inplace=True)  # Ensure data is sorted by date

# Convert daily prices into daily returns
returns = data.pct_change()

# Calculate the mean daily returns and the covariance matrix of daily returns
mean_daily_returns = returns.mean()
cov_matrix = returns.cov()

# Set the number of simulations for the Monte Carlo approach
num_portfolios = 25000

# Initialize an array to store results (returns, volatility, Sharpe ratio, and weights)
results = np.zeros((3 + num_stocks, num_portfolios))

# Run the simulation for the given number of portfolios
for i in range(num_portfolios):
    # Randomly generate portfolio weights
    weights = np.random.random(num_stocks)

    # Normalize weights to ensure they sum to 1
    weights /= np.sum(weights)

    # Calculate portfolio return (annualized)
    portfolio_return = np.sum(mean_daily_returns * weights) * 252

    # Calculate portfolio volatility (annualized standard deviation)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

    # Store results in the results array
    results[0, i] = portfolio_return  # Portfolio return
    results[1, i] = portfolio_std_dev  # Portfolio standard deviation (volatility)
    results[2, i] = results[0, i] / results[1, i]  # Sharpe ratio (excluding risk-free rate)

    # Store the weights of each stock in the results array
    results[3:, i] = weights

# Convert the results array into a Pandas DataFrame for easier analysis
columns = ['ret', 'stdev', 'sharpe'] + stocks
results_frame = pd.DataFrame(results.T, columns=columns)

# Identify the portfolio with the maximum Sharpe Ratio
max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]

# Identify the portfolio with the minimum standard deviation (volatility)
min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

# Create a scatter plot of portfolios colored by Sharpe Ratio
plt.scatter(results_frame.stdev, results_frame.ret, c=results_frame.sharpe, cmap='RdYlBu')
plt.xlabel('Volatility')  # X-axis: Portfolio volatility (risk)
plt.ylabel('Returns')  # Y-axis: Portfolio returns
plt.colorbar(label='Sharpe Ratio')  # Color bar to indicate Sharpe Ratio levels

# Highlight the portfolio with the maximum Sharpe Ratio (red star)
plt.scatter(max_sharpe_port['stdev'], max_sharpe_port['ret'], marker=(5, 1, 0), color='r', s=100,
            label='Max Sharpe Ratio')

# Highlight the portfolio with the minimum volatility (green star)
plt.scatter(min_vol_port['stdev'], min_vol_port['ret'], marker=(5, 1, 0), color='g', s=100, label='Min Volatility')

# Add a legend and display the plot
plt.legend()
plt.show()

# Display the selected stocks, max Sharpe portfolio, and min volatility portfolio
print("Selected Stocks:", stocks)
print("\nPortfolio with Maximum Sharpe Ratio:")
print(max_sharpe_port)
print("\nPortfolio with Minimum Volatility:")
print(min_vol_port)
