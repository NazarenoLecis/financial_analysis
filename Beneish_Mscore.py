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

# Calculate the Beneish M-Score for a given stock ticker
def calculate_beneish_m_score(ticker):
    """
    Calculate the Beneish M-Score for a stock.

    Args:
        ticker (str): Stock ticker symbol.

    Returns:
        float: Beneish M-Score.
    """
    try:
        data = fetch_financial_data(ticker)
        balance_sheet, income_statement, cash_flow, years = data

        # Check if we have at least two years of data
        if len(years) < 2:
            raise ValueError("Not enough financial data available")

        # Beneish M-Score components calculation

        # DSRI: Days Sales in Receivables Index
        receivables = balance_sheet.loc["Net Receivables", years[0]]
        receivables_previous = balance_sheet.loc["Net Receivables", years[1]]
        sales = income_statement.loc["Total Revenue", years[0]]
        sales_previous = income_statement.loc["Total Revenue", years[1]]
        dsri = (receivables / sales) / (receivables_previous / sales_previous)

        # GMI: Gross Margin Index
        gross_margin = (income_statement.loc["Total Revenue", years[0]] - income_statement.loc["Cost Of Revenue", years[0]]) / income_statement.loc["Total Revenue", years[0]]
        gross_margin_previous = (income_statement.loc["Total Revenue", years[1]] - income_statement.loc["Cost Of Revenue", years[1]]) / income_statement.loc["Total Revenue", years[1]]
        gmi = gross_margin_previous / gross_margin

        # AQI: Asset Quality Index
        total_assets = balance_sheet.loc["Total Assets", years[0]]
        total_assets_previous = balance_sheet.loc["Total Assets", years[1]]
        current_assets = balance_sheet.loc.get("Total Current Assets", balance_sheet.loc["Current Assets", years[0]])
        current_assets_previous = balance_sheet.loc.get("Total Current Assets", balance_sheet.loc["Current Assets", years[1]])
        ppe = balance_sheet.loc.get("Property Plant Equipment", balance_sheet.loc["Net PPE", years[0]])
        ppe_previous = balance_sheet.loc.get("Property Plant Equipment", balance_sheet.loc["Net PPE", years[1]])
        aqi = (1 - (current_assets + ppe) / total_assets) / (1 - (current_assets_previous + ppe_previous) / total_assets_previous)

        # SGI: Sales Growth Index
        sgi = sales / sales_previous

        # DEPI: Depreciation Index
        depreciation = income_statement.loc.get("Depreciation", income_statement.loc["Depreciation Amortization", years[0]])
        depreciation_previous = income_statement.loc.get("Depreciation", income_statement.loc["Depreciation Amortization", years[1]])
        depi = (depreciation_previous / (depreciation_previous + ppe_previous)) / (depreciation / (depreciation + ppe))

        # SGAI: Sales, General, and Administrative Expenses Index
        sga_expense = income_statement.loc.get("Selling General Administrative", income_statement.loc["SGA Expense", years[0]])
        sga_expense_previous = income_statement.loc.get("Selling General Administrative", income_statement.loc["SGA Expense", years[1]])
        sgai = (sga_expense / sales) / (sga_expense_previous / sales_previous)

        # LVGI: Leverage Index
        total_liabilities = balance_sheet.loc["Total Liabilities Net Minority Interest", years[0]]
        total_liabilities_previous = balance_sheet.loc["Total Liabilities Net Minority Interest", years[1]]
        lvgi = (total_liabilities / total_assets) / (total_liabilities_previous / total_assets_previous)

        # TATA: Total Accruals to Total Assets
        net_income = income_statement.loc["Net Income", years[0]]
        cash_flow_operations = cash_flow.loc["Total Cash From Operating Activities", years[0]]
        tata = (net_income - cash_flow_operations) / total_assets

        # Beneish M-Score calculation
        m_score = -4.84 + 0.92 * dsri + 0.528 * gmi + 0.404 * aqi + 0.892 * sgi + 0.115 * depi - 0.172 * sgai + 4.679 * tata - 0.327 * lvgi

        return m_score

    except Exception as e:
        print(f"Failed to calculate the Beneish M-Score for {ticker}: {e}")
        return None

# Example usage
ticker = "AAPL"  # Replace with the desired stock ticker
m_score = calculate_beneish_m_score(ticker)
if m_score is not None:
    print(f"The Beneish M-Score for {ticker} is {m_score}")
else:
    print(f"Could not calculate the Beneish M-Score for {ticker}")
