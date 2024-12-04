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


# Calculate profitability score
def calculate_profitability(data):
    """
  Calculate the profitability score based on net income, cash flow, ROA, and accruals.

  Args:
      data (list): Financial data [balance_sheet, income_statement, cash_flow, years].

  Returns:
      int: Profitability score (0 to 4).
  """
    balance_sheet, income_statement, cash_flow, years = data

    # Score 1: Net Income Growth
    net_income = income_statement[years[0]]['Net Income']
    net_income_previous_year = income_statement[years[1]]['Net Income']
    profit_score = 1 if net_income > net_income_previous_year else 0

    # Score 2: Positive Free Cash Flow
    free_cash_flow = cash_flow[years[0]]['Free Cash Flow']
    cash_flow_score = 1 if free_cash_flow > 0 else 0

    # Score 3: Positive ROA Change
    total_assets = balance_sheet[years[0]]['Total Assets']
    total_assets_previous_year = balance_sheet[years[1]]['Total Assets']
    roa = net_income / total_assets
    roa_previous_year = net_income_previous_year / total_assets_previous_year
    roa_score = 1 if roa > roa_previous_year else 0

    # Score 4: Accruals (Cash flow from operations / total assets)
    accruals = free_cash_flow / total_assets
    accruals_score = 1 if accruals > roa else 0

    return profit_score + cash_flow_score + roa_score + accruals_score


# Calculate leverage and liquidity score
def calculate_leverage(data):
    """
  Calculate the leverage and liquidity score based on debt, current ratio, and share dilution.

  Args:
      data (list): Financial data [balance_sheet, income_statement, cash_flow, years].

  Returns:
      int: Leverage and liquidity score (0 to 3).
  """
    balance_sheet, _, _, years = data

    # Score 5: Decrease in Long-Term Debt
    try:
        long_term_debt = balance_sheet[years[0]]['Long Term Debt']
        long_term_debt_previous_year = balance_sheet[years[1]]['Long Term Debt']
        debt_score = 1 if long_term_debt < long_term_debt_previous_year else 0
    except KeyError:
        debt_score = 1  # Assume no long-term debt available

    # Score 6: Improving Current Ratio (Current Assets / Current Liabilities)
    try:
        current_assets = balance_sheet[years[0]]['Current Assets']
        current_liabilities = balance_sheet[years[0]]['Current Liabilities']
    except KeyError:
        current_assets = balance_sheet[years[0]]['Net Tangible Assets']
        current_liabilities = balance_sheet[years[0]]['Current Debt']

    current_ratio = current_assets / current_liabilities

    try:
        current_assets_previous_year = balance_sheet[years[1]]['Current Assets']
        current_liabilities_previous_year = balance_sheet[years[1]]['Current Liabilities']
    except KeyError:
        current_assets_previous_year = balance_sheet[years[1]]['Net Tangible Assets']
        current_liabilities_previous_year = balance_sheet[years[1]]['Current Debt']

    current_ratio_previous_year = current_assets_previous_year / current_liabilities_previous_year
    current_ratio_score = 1 if current_ratio > current_ratio_previous_year else 0

    # Score 7: No Share Dilution
    shares_issued = balance_sheet[years[0]]['Share Issued']
    shares_issued_previous_year = balance_sheet[years[1]]['Share Issued']
    dilution_score = 1 if shares_issued <= shares_issued_previous_year else 0

    return debt_score + current_ratio_score + dilution_score


# Calculate operational efficiency score
def calculate_operational_efficiency(data):
    """
  Calculate the operational efficiency score based on gross margin and asset turnover.

  Args:
      data (list): Financial data [balance_sheet, income_statement, cash_flow, years].

  Returns:
      int: Operational efficiency score (0 to 2).
  """
    balance_sheet, income_statement, _, years = data

    # Score 8: Improvement in Gross Margin
    try:
        ebitda = income_statement[years[0]]['Normalized EBITDA']
        ebitda_previous_year = income_statement[years[1]]['Normalized EBITDA']
    except KeyError:
        try:
            ebitda = income_statement[years[0]]['EBITDA']
            ebitda_previous_year = income_statement[years[1]]['EBITDA']
        except KeyError:
            ebitda = (income_statement[years[0]]['Net Income'] +
                      income_statement[years[0]]['Reconciled Depreciation'] +
                      income_statement[years[0]]['Tax Provision'])
            ebitda_previous_year = (income_statement[years[1]]['Net Income'] +
                                    income_statement[years[1]]['Reconciled Depreciation'] +
                                    income_statement[years[1]]['Tax Provision'])

    gross_margin_score = 1 if ebitda > ebitda_previous_year else 0

    # Score 9: Improvement in Asset Turnover
    total_assets = balance_sheet[years[0]]['Total Assets']
    total_assets_previous_year = balance_sheet[years[1]]['Total Assets']
    average_assets = (total_assets + total_assets_previous_year) / 2
    revenue = income_statement[years[0]]['Total Revenue']
    asset_turnover = revenue / average_assets

    total_assets_two_years_ago = balance_sheet[years[2]]['Total Assets']
    average_assets_previous_year = (total_assets_previous_year + total_assets_two_years_ago) / 2
    revenue_previous_year = income_statement[years[1]]['Total Revenue']
    asset_turnover_previous_year = revenue_previous_year / average_assets_previous_year

    asset_turnover_score = 1 if asset_turnover > asset_turnover_previous_year else 0

    return gross_margin_score + asset_turnover_score


# Calculate the Piotroski F-score for a given stock ticker
def calculate_piotroski_score(ticker):
    """
  Calculate the Piotroski F-score for a stock.

  Args:
      ticker (str): Stock ticker symbol.

  Returns:
      int: Piotroski F-score (0 to 9).
  """
    data = fetch_financial_data(ticker)
    profitability_score = calculate_profitability(data)
    leverage_score = calculate_leverage(data)
    efficiency_score = calculate_operational_efficiency(data)
    return profitability_score + leverage_score + efficiency_score


# Fetch the list of S&P 500 companies
sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

results = {}
errors = {}

# Calculate the Piotroski F-score for each stock in the S&P 500
for ticker in sp500_tickers.Symbol:
    try:
        f_score = calculate_piotroski_score(ticker)
        results[ticker] = f_score
        print(f"The Piotroski F-score for {ticker} is {f_score}")
    except Exception as e:
        errors[ticker] = e
        print(f"Failed to calculate the Piotroski F-score for {ticker}.")
