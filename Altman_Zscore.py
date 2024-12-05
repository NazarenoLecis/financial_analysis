import yfinance as yf
import pandas as pd

# Fetch financial data for a given ticker
def fetch_financial_data(ticker):
    """
  Fetch the balance sheet, income statement, and cash flow statement for a stock.

  Args:
      ticker (str): Stock ticker symbol.

  Returns:
      list: A list containing:
          - Balance sheet (DataFrame)
          - Income statement (DataFrame)
          - Cash flow statement (DataFrame)
          - Columns representing the years of financial data
  """
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.income_stmt
    cash_flow = stock.cash_flow
    years = balance_sheet.columns  # Financial years
    return [balance_sheet, income_statement, cash_flow, years]

# Calculate the Altman Z-Score for a given stock ticker
def calculate_altman_z_score(ticker):
    """
    Calculate the Altman Z-Score for a stock.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float: Altman Z-Score.
    """
    data = fetch_financial_data(ticker)
    balance_sheet, income_statement, cash_flow, years, stock = data

    # Altman Z-Score components calculation

    # X1: Working Capital / Total Assets
    try:
        working_capital = balance_sheet.loc["Total Current Assets", years[0]] - balance_sheet.loc["Total Current Liabilities", years[0]]
    except KeyError:
        working_capital = balance_sheet.loc["Current Assets", years[0]] - balance_sheet.loc["Current Liabilities", years[0]]
    
    total_assets = balance_sheet.loc["Total Assets", years[0]]
    x1 = working_capital / total_assets

    # X2: Retained Earnings / Total Assets
    retained_earnings = balance_sheet.loc["Retained Earnings", years[0]]
    x2 = retained_earnings / total_assets

    # X3: EBIT / Total Assets
    ebit = income_statement.loc["EBIT", years[0]]
    x3 = ebit / total_assets

    # X4: Market Value of Equity / Total Liabilities
    market_value_of_equity = stock.info['marketCap']
    total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest", years[0]]
    x4 = market_value_of_equity / total_liabilities

    # X5: Sales / Total Assets
    sales = income_statement.loc["Total Revenue", years[0]]
    x5 = sales / total_assets

    # Altman Z-Score calculation
    z_score = 1.2 * x1 + 1.4 * x2 + 3.3 * x3 + 0.6 * x4 + 1.0 * x5

    return z_score

# Fetch the list of S&P 500 companies from an alternative source
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol'].tolist()

results = {}
errors = {}

# Calculate the Altman Z-Score for each stock in the S&P 500
for ticker in sp500_tickers:
    try:
        z_score = calculate_altman_z_score(ticker)
        results[ticker] = z_score
        print(f"The Altman Z-Score for {ticker} is {z_score}")
    except Exception as e:
        errors[ticker] = e
        print(f"Failed to calculate the Altman Z-Score for {ticker}: {e}")

# Print results and errors summary
print("\nAltman Z-Scores:")
for ticker, z_score in results.items():
    print(f"{ticker}: {z_score}")

print("\nErrors:")
for ticker, error in errors.items():
    print(f"{ticker}: {error}")
