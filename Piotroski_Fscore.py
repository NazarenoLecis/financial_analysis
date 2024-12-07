import yfinance as yf
import pandas as pd


# Fetch financial data for a given ticker
def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.income_stmt
    cash_flow = stock.cash_flow
    years = balance_sheet.columns  # Financial years
    return [balance_sheet, income_statement, cash_flow, years]


# Calculate profitability score
def calculate_profitability(data):
    balance_sheet, income_statement, cash_flow, years = data

    net_income = income_statement[years[0]]['Net Income']
    net_income_previous_year = income_statement[years[1]]['Net Income']
    profit_score = 1 if net_income > net_income_previous_year else 0

    free_cash_flow = cash_flow[years[0]]['Free Cash Flow']
    cash_flow_score = 1 if free_cash_flow > 0 else 0

    total_assets = balance_sheet[years[0]]['Total Assets']
    total_assets_previous_year = balance_sheet[years[1]]['Total Assets']
    roa = net_income / total_assets
    roa_previous_year = net_income_previous_year / total_assets_previous_year
    roa_score = 1 if roa > roa_previous_year else 0

    accruals = free_cash_flow / total_assets
    accruals_score = 1 if accruals > roa else 0

    return profit_score + cash_flow_score + roa_score + accruals_score


# Calculate leverage and liquidity score
def calculate_leverage(data):
    balance_sheet, _, _, years = data

    try:
        long_term_debt = balance_sheet[years[0]]['Long Term Debt']
        long_term_debt_previous_year = balance_sheet[years[1]]['Long Term Debt']
        debt_score = 1 if long_term_debt < long_term_debt_previous_year else 0
    except KeyError:
        debt_score = 1

    current_assets = balance_sheet[years[0]]['Current Assets']
    current_liabilities = balance_sheet[years[0]]['Current Liabilities']
    current_ratio = current_assets / current_liabilities

    current_assets_previous_year = balance_sheet[years[1]]['Current Assets']
    current_liabilities_previous_year = balance_sheet[years[1]]['Current Liabilities']
    current_ratio_previous_year = current_assets_previous_year / current_liabilities_previous_year
    current_ratio_score = 1 if current_ratio > current_ratio_previous_year else 0

    shares_issued = balance_sheet[years[0]]['Share Issued']
    shares_issued_previous_year = balance_sheet[years[1]]['Share Issued']
    dilution_score = 1 if shares_issued <= shares_issued_previous_year else 0

    return debt_score + current_ratio_score + dilution_score


# Calculate operational efficiency score
def calculate_operational_efficiency(data):
    balance_sheet, income_statement, _, years = data

    try:
        ebitda = income_statement[years[0]]['EBITDA']
        ebitda_previous_year = income_statement[years[1]]['EBITDA']
    except KeyError:
        ebitda = income_statement[years[0]]['Net Income']
        ebitda_previous_year = income_statement[years[1]]['Net Income']

    gross_margin_score = 1 if ebitda > ebitda_previous_year else 0

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
    data = fetch_financial_data(ticker)
    profitability_score = calculate_profitability(data)
    leverage_score = calculate_leverage(data)
    efficiency_score = calculate_operational_efficiency(data)
    return profitability_score + leverage_score + efficiency_score


# Fetch the list of tickers for S&P 500 and Nasdaq-100
def fetch_sp500_tickers():
    sp500_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return sp500_data.Symbol.tolist()


def fetch_nasdaq100_tickers():
    nasdaq100_data = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[4]
    return nasdaq100_data.Symbol.tolist()


# Main function
def main():
    print("Choose the index for analysis:")
    print("1. S&P 500")
    print("2. Nasdaq-100")
    choice = input("Enter the number corresponding to your choice: ")

    if choice == "1":
        tickers = fetch_sp500_tickers()
        print("You selected the S&P 500.")
    elif choice == "2":
        tickers = fetch_nasdaq100_tickers()
        print("You selected the Nasdaq-100.")
    else:
        print("Invalid choice. Exiting.")
        return

    results = {}
    errors = {}

    for ticker in tickers:
        try:
            f_score = calculate_piotroski_score(ticker)
            results[ticker] = f_score
            print(f"The Piotroski F-score for {ticker} is {f_score}")
        except Exception as e:
            errors[ticker] = e
            print(f"Failed to calculate the Piotroski F-score for {ticker}. Error: {e}")

    print("\nCalculation complete.")
    print("Results:", results)
    if errors:
        print("\nErrors occurred for the following tickers:")
        for ticker, error in errors.items():
            print(f"{ticker}: {error}")


if __name__ == "__main__":
    main()
