# Financial Analysis

Python scripts for quick stock/company views, fundamental analysis, technical analysis, and portfolio analysis.

The project uses `yfinance` for market and financial statement data. Index constituents are scraped from Wikipedia with `requests` and `beautifulsoup4`.

## Setup

Use Python 3.13 or another recent Python 3 version.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Project Structure

```text
financial_analysis/
  utils.py
  quick_views/
    stock_snapshot.py
    balance_sheet_summary.py
    notebooks/
      stock_snapshot.ipynb
      balance_sheet_summary.ipynb
  fundamental_analysis/
    Altman_Zscore.py
    Beneish_Mscore.py
    Piotroski_Fscore.py
    ratio_analysis.py
    financial_statement_charts.py
    dupont_analysis.py
    dcf_valuation.py
    dividend_analysis.py
  technical_analysis/
    indicators.py
    moving_averages.py
    rsi_analysis.py
    macd_analysis.py
    bollinger_bands.py
    technical_dashboard.py
  portfolio_analysis/
    Monte_Carlo_simulation.py
    portfolio_optimization.py
```

## Running From VS Code

Every runnable script has defaults, so pressing "Run Python File" in VS Code works without arguments. Single-company scripts default to `AAPL`; ratio analysis defaults to `AAPL MSFT`; portfolio scripts default to `AAPL MSFT NVDA GOOGL`.

Use the command-line options below when you want different tickers, index samples, dates, assumptions, or no chart window.

## Command Help

Every runnable script has detailed command-line help:

```powershell
python technical_analysis/rsi_analysis.py --help
python quick_views/stock_snapshot.py --help
python portfolio_analysis/portfolio_optimization.py --help
```

The help text explains accepted variables, possible values, defaults, date formats, examples, and whether the script uses latest financial statement values or historical time-series data.

## Quick Views

These scripts are for fast inspection rather than deeper modelling.

Quick stock snapshot:

```powershell
python quick_views/stock_snapshot.py --tickers AAPL MSFT NVDA GOOGL
```

Balance sheet summary:

```powershell
python quick_views/balance_sheet_summary.py --ticker AAPL
```

Interactive notebooks:

```powershell
jupyter notebook quick_views/notebooks/stock_snapshot.ipynb
jupyter notebook quick_views/notebooks/balance_sheet_summary.ipynb
```

You can also open the notebooks directly in VS Code. Edit the input cell, rerun the following cells, and the tables/charts update inline.

## Fundamental Analysis

Quality and risk scores:

```powershell
python fundamental_analysis/Altman_Zscore.py --tickers AAPL
python fundamental_analysis/Beneish_Mscore.py --tickers AAPL
python fundamental_analysis/Piotroski_Fscore.py --tickers AAPL
```

Ratio analysis:

```powershell
python fundamental_analysis/ratio_analysis.py --tickers AAPL MSFT NVDA
python fundamental_analysis/ratio_analysis.py --index sp500 --limit 5 --plot
```

Financial statement charts:

```powershell
python fundamental_analysis/financial_statement_charts.py --ticker AAPL
```

DuPont ROE decomposition:

```powershell
python fundamental_analysis/dupont_analysis.py --ticker AAPL
```

Discounted cash flow valuation:

```powershell
python fundamental_analysis/dcf_valuation.py --ticker AAPL --growth-rate 0.05 --discount-rate 0.10
```

Dividend history:

```powershell
python fundamental_analysis/dividend_analysis.py --ticker AAPL
```

## Technical Analysis

Moving averages:

```powershell
python technical_analysis/moving_averages.py --ticker AAPL
```

RSI:

```powershell
python technical_analysis/rsi_analysis.py --ticker AAPL
```

MACD:

```powershell
python technical_analysis/macd_analysis.py --ticker AAPL
```

Bollinger bands:

```powershell
python technical_analysis/bollinger_bands.py --ticker AAPL
```

Combined technical dashboard:

```powershell
python technical_analysis/technical_dashboard.py --ticker AAPL
```

Add `--no-plot` to any charting command when you only want printed output.

## Portfolio Analysis

Monte Carlo simulation:

```powershell
python portfolio_analysis/Monte_Carlo_simulation.py --tickers AAPL MSFT NVDA GOOGL --simulations 1000
```

Portfolio optimization:

```powershell
python portfolio_analysis/portfolio_optimization.py --tickers AAPL MSFT NVDA GOOGL
```

## Notes

- Yahoo Finance statement labels vary by company, so the code uses aliases and reports missing fields clearly.
- S&P symbols such as `BRK.B` are converted to Yahoo Finance format such as `BRK-B`.
- `scipy` is used for portfolio optimization when installed. If it is unavailable, the optimizer falls back to a Monte Carlo search.
- This is educational analysis code, not investment advice.
