# Financial Analysis

Python scripts for calculating common equity risk and quality indicators:

- Altman Z-score
- Beneish M-score
- Piotroski F-score
- Monte Carlo portfolio simulation
- Portfolio optimization

The scripts use `yfinance` for market and financial statement data. Index constituents are scraped from Wikipedia with `requests` and `beautifulsoup4`, so `lxml` is not required.

## Setup

Use Python 3.13 or another recent Python 3 version.

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Examples

Run a single ticker:

```powershell
python Altman_Zscore.py --tickers AAPL
python Beneish_Mscore.py --tickers AAPL
python Piotroski_Fscore.py --tickers AAPL
```

Run a small index sample:

```powershell
python Altman_Zscore.py --index sp500 --limit 5
python Beneish_Mscore.py --index sp500 --limit 5
python Piotroski_Fscore.py --index nasdaq100 --limit 5
```

Run portfolio tools without opening charts:

```powershell
python Monte_Carlo_simulation.py --tickers AAPL MSFT NVDA GOOGL --simulations 1000 --no-plot
python portfolio_optimization.py --tickers AAPL MSFT NVDA GOOGL --no-plot
```

Remove `--no-plot` when you want the matplotlib charts.

## Notes

- Yahoo Finance statement labels vary by company, so the code uses field aliases and reports missing fields per ticker.
- S&P symbols such as `BRK.B` are converted to Yahoo Finance format such as `BRK-B`.
- `scipy` is used for portfolio optimization when installed. If it is unavailable, the optimizer falls back to a Monte Carlo search.
- This is educational analysis code, not investment advice.

## Files

- `utils.py`: shared ticker, statement, price, and calculation helpers.
- `Altman_Zscore.py`: Altman Z-score CLI.
- `Beneish_Mscore.py`: Beneish M-score CLI.
- `Piotroski_Fscore.py`: Piotroski F-score CLI.
- `Monte_Carlo_simulation.py`: random portfolio simulation.
- `portfolio_optimization.py`: Sharpe-ratio portfolio optimizer.
