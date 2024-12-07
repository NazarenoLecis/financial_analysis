import yfinance as yf
import pandas as pd

# Fetch S&P 500 tickers
def get_sp500_tickers():
    table = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
    sp500_df = table[0]  # The first table contains the S&P 500 tickers
    return sp500_df['Symbol'].tolist()

# Fetch financial data for a given ticker
def fetch_financial_data(ticker):
    stock = yf.Ticker(ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.income_stmt
    cash_flow = stock.cash_flow
    years = balance_sheet.columns  # Financial years
    return [balance_sheet, income_statement, cash_flow, years]

# Calculate the Beneish M-Score for a given stock ticker
def calculate_beneish_m_score(ticker):
    try:
        data = fetch_financial_data(ticker)
        balance_sheet, income_statement, cash_flow, years = data

        # Check if we have at least two years of data
        if len(years) < 2:
            raise ValueError("Not enough financial data available")

        # Helper functions
        def safe_get(df, row, column, default=0):
            try:
                return df.loc[row, column]
            except KeyError:
                return default

        def safe_divide(numerator, denominator):
            return numerator / denominator if denominator != 0 else 0

        # DSRI: Days Sales in Receivables Index
        receivables = safe_get(balance_sheet, "Accounts Receivable", years[0])
        receivables_previous = safe_get(balance_sheet, "Accounts Receivable", years[1])
        sales = safe_get(income_statement, "Total Revenue", years[0])
        sales_previous = safe_get(income_statement, "Total Revenue", years[1])
        dsri = safe_divide((receivables / sales), (receivables_previous / sales_previous))

        # GMI: Gross Margin Index
        gross_margin = safe_divide((sales - safe_get(income_statement, "Cost Of Revenue", years[0])), sales)
        gross_margin_previous = safe_divide((sales_previous - safe_get(income_statement, "Cost Of Revenue", years[1])), sales_previous)
        gmi = safe_divide(gross_margin_previous, gross_margin)

        # AQI: Asset Quality Index
        total_assets = safe_get(balance_sheet, "Total Assets", years[0])
        total_assets_previous = safe_get(balance_sheet, "Total Assets", years[1])
        current_assets = safe_get(balance_sheet, "Current Assets", years[0])
        current_assets_previous = safe_get(balance_sheet, "Current Assets", years[1])
        ppe = safe_get(balance_sheet, "Net PPE", years[0])
        ppe_previous = safe_get(balance_sheet, "Net PPE", years[1])
        aqi = safe_divide((1 - (current_assets + ppe) / total_assets), (1 - (current_assets_previous + ppe_previous) / total_assets_previous))

        # SGI: Sales Growth Index
        sgi = safe_divide(sales, sales_previous)

        # DEPI: Depreciation Index
        depreciation = safe_get(income_statement, "Depreciation", years[0])
        depreciation_previous = safe_get(income_statement, "Depreciation", years[1])
        depi = safe_divide((depreciation_previous / (depreciation_previous + ppe_previous)), (depreciation / (depreciation + ppe)))

        # SGAI: Sales, General, and Administrative Expenses Index
        sga_expense = safe_get(income_statement, "Selling General Administrative", years[0])
        sga_expense_previous = safe_get(income_statement, "Selling General Administrative", years[1])
        sgai = safe_divide((sga_expense / sales), (sga_expense_previous / sales_previous))

        # LVGI: Leverage Index
        total_liabilities = safe_get(balance_sheet, "Total Liabilities Net Minority Interest", years[0])
        total_liabilities_previous = safe_get(balance_sheet, "Total Liabilities Net Minority Interest", years[1])
        lvgi = safe_divide((total_liabilities / total_assets), (total_liabilities_previous / total_assets_previous))

        # TATA: Total Accruals to Total Assets
        net_income = safe_get(income_statement, "Net Income", years[0])
        cash_flow_operations = safe_get(cash_flow, "Total Cash From Operating Activities", years[0])
        tata = safe_divide((net_income - cash_flow_operations), total_assets)

        # Beneish M-Score calculation
        m_score = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi

        return m_score

    except Exception as e:
        raise RuntimeError(f"Failed to calculate M-Score for {ticker}: {e}")

# Fetch the list of S&P 500 companies and calculate Beneish M-Score for each
sp500_tickers = get_sp500_tickers()
errors = {}

for ticker in sp500_tickers:
    try:
        m_score = calculate_beneish_m_score(ticker)
        print(f"The Beneish M-Score for {ticker} is {m_score}")
    except Exception as e:
        errors[ticker] = str(e)
        print(f"Error for {ticker}: {e}")

# Print errors summary
print("\nErrors Summary:")
for ticker, error_message in errors.items():
    print(f"{ticker}: {error_message}")
