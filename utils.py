"""Shared helpers for ticker scraping, yfinance data access, and safe calculations."""

from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup


USER_AGENT = "financial-analysis/1.0"

INDEX_URLS = {
    "sp500": "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
    "nasdaq100": "https://en.wikipedia.org/wiki/Nasdaq-100",
}


@dataclass(frozen=True)
class FinancialStatements:
    """Container for the annual statements used by the score calculators."""

    ticker: str
    yahoo_ticker: str
    stock: yf.Ticker
    balance_sheet: pd.DataFrame
    income_statement: pd.DataFrame
    cash_flow: pd.DataFrame
    periods: pd.Index


class MissingFinancialField(ValueError):
    pass


def yahoo_symbol(ticker: str) -> str:
    """Convert index tickers such as BRK.B to Yahoo Finance symbols such as BRK-B."""

    return ticker.strip().replace(".", "-")


def fetch_index_tickers(index: str = "sp500") -> list[str]:
    """Fetch index constituents from Wikipedia without requiring pandas/lxml HTML parsing."""

    try:
        url = INDEX_URLS[index.lower()]
    except KeyError as exc:
        raise ValueError(f"Unknown index '{index}'. Choose one of: {', '.join(INDEX_URLS)}") from exc

    response = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    for table in soup.find_all("table"):
        headers = [cell.get_text(strip=True).lower() for cell in table.find_all("th")]
        if "symbol" not in headers and "ticker" not in headers:
            continue

        symbol_column = headers.index("symbol") if "symbol" in headers else headers.index("ticker")
        tickers = []
        for row in table.find_all("tr")[1:]:
            cells = row.find_all(["td", "th"])
            if len(cells) <= symbol_column:
                continue
            ticker = cells[symbol_column].get_text(strip=True).split()[0]
            if ticker:
                tickers.append(ticker)
        if tickers:
            return tickers

    raise RuntimeError(f"Could not find ticker table at {url}")


def fetch_financial_data(ticker: str) -> FinancialStatements:
    """Fetch annual balance sheet, income statement, and cash flow data from yfinance."""

    yahoo_ticker = yahoo_symbol(ticker)
    stock = yf.Ticker(yahoo_ticker)
    balance_sheet = stock.balance_sheet
    income_statement = stock.income_stmt
    cash_flow = stock.cash_flow

    if balance_sheet.empty or income_statement.empty:
        raise ValueError(f"No annual financial statements found for {ticker}")

    return FinancialStatements(
        ticker=ticker,
        yahoo_ticker=yahoo_ticker,
        stock=stock,
        balance_sheet=balance_sheet,
        income_statement=income_statement,
        cash_flow=cash_flow,
        periods=balance_sheet.columns,
    )


def require_periods(data: FinancialStatements, count: int) -> None:
    """Ensure a formula has enough annual periods to compare current and prior years."""

    if len(data.periods) < count:
        raise ValueError(f"{data.ticker} has {len(data.periods)} periods; {count} are required")


def statement_value(
    statement: pd.DataFrame,
    period,
    aliases: str | Iterable[str],
    *,
    required: bool = True,
    default: float | None = None,
) -> float | None:
    """Return a statement field using aliases because yfinance labels vary by company."""

    if isinstance(aliases, str):
        aliases = [aliases]

    for alias in aliases:
        if alias in statement.index and period in statement.columns:
            value = statement.loc[alias, period]
            if pd.notna(value):
                return float(value)

    normalized_index = {_normalize_label(label): label for label in statement.index}
    for alias in aliases:
        label = normalized_index.get(_normalize_label(alias))
        if label is None:
            continue
        if period not in statement.columns:
            continue
        value = statement.loc[label, period]
        if pd.notna(value):
            return float(value)

    if required:
        raise MissingFinancialField(f"Missing field: {', '.join(aliases)}")
    return default


def safe_divide(numerator: float, denominator: float, label: str) -> float:
    """Divide with a useful error message instead of silently returning invalid ratios."""

    if denominator == 0:
        raise ZeroDivisionError(f"Cannot divide by zero while calculating {label}")
    return numerator / denominator


def market_cap_from_stock(stock: yf.Ticker, ticker: str) -> float:
    """Fetch market capitalization from fast_info first, then the slower info endpoint."""

    try:
        value = stock.fast_info.get("market_cap")
    except Exception:
        value = None

    if value is None:
        value = stock.info.get("marketCap")

    if value is None:
        raise ValueError(f"No market capitalization found for {ticker}")
    return float(value)


def shares_outstanding_from_stock(stock: yf.Ticker, ticker: str) -> float:
    """Fetch shares outstanding for per-share valuation models."""

    try:
        value = stock.fast_info.get("shares")
    except Exception:
        value = None

    if value is None:
        value = stock.info.get("sharesOutstanding")

    if value is None:
        raise ValueError(f"No shares outstanding found for {ticker}")
    return float(value)


def extract_close_prices(downloaded: pd.DataFrame, tickers: list[str] | None = None) -> pd.DataFrame:
    """Extract adjusted close prices, falling back to close prices when yfinance omits them."""

    if downloaded.empty:
        raise ValueError("No price data returned from yfinance")

    if isinstance(downloaded.columns, pd.MultiIndex):
        price_field = "Adj Close" if "Adj Close" in downloaded.columns.get_level_values(0) else "Close"
        prices = downloaded[price_field]
    else:
        price_field = "Adj Close" if "Adj Close" in downloaded.columns else "Close"
        prices = downloaded[[price_field]].copy()
        if tickers and len(tickers) == 1:
            prices = prices.rename(columns={price_field: tickers[0]})

    return prices.dropna(axis=1, how="all")


def fetch_price_history(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download OHLCV price history for one ticker."""

    yahoo_ticker = yahoo_symbol(ticker)
    data = yf.download(
        yahoo_ticker,
        start=start_date,
        end=end_date,
        interval=interval,
        progress=False,
        auto_adjust=False,
        multi_level_index=False,
    )
    if data.empty:
        raise ValueError(f"No price data returned for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data = _flatten_single_ticker_download(data, yahoo_ticker)

    return data.dropna(how="all")


def statement_series(
    statement: pd.DataFrame,
    periods: pd.Index,
    aliases: str | Iterable[str],
    *,
    required: bool = False,
) -> pd.Series:
    """Build a time series from a financial statement row across available periods."""

    values = [
        statement_value(statement, period, aliases, required=required, default=pd.NA)
        for period in periods
    ]
    series = pd.Series(values, index=pd.to_datetime(periods), dtype="Float64")
    return series.sort_index()


def period_labels(index: pd.Index) -> list[str]:
    """Return readable year labels for statement chart x-axes."""

    return [str(getattr(period, "year", period)) for period in index]


def _flatten_single_ticker_download(data: pd.DataFrame, yahoo_ticker: str) -> pd.DataFrame:
    if "Ticker" in data.columns.names:
        return data.xs(yahoo_ticker, level="Ticker", axis=1, drop_level=True)

    if yahoo_ticker in data.columns.get_level_values(-1):
        return data.xs(yahoo_ticker, level=-1, axis=1, drop_level=True)

    if yahoo_ticker in data.columns.get_level_values(0):
        return data.xs(yahoo_ticker, level=0, axis=1, drop_level=True)

    data.columns = data.columns.get_level_values(0)
    return data


def _normalize_label(label) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())
