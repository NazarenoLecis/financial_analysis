"""Data-loading helpers for strategy tests."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import yfinance as yf

from utils import extract_close_prices, fetch_index_tickers, yahoo_symbol


def resolve_ticker_universe(
    *,
    tickers: list[str] | None,
    index: str,
    limit: int | None = None,
) -> list[str]:
    """Return explicit tickers or index constituents."""

    universe = tickers if tickers else fetch_index_tickers(index)
    if limit:
        universe = universe[:limit]
    return [yahoo_symbol(ticker) for ticker in universe]


def fetch_close_prices(
    tickers: list[str],
    start_date: str,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download adjusted close prices for a ticker list."""

    if not tickers:
        raise ValueError("At least one ticker is required")

    yahoo_tickers = [yahoo_symbol(ticker) for ticker in tickers]
    end = end_date or dt.datetime.today().strftime("%Y-%m-%d")
    downloaded = yf.download(yahoo_tickers, start=start_date, end=end, progress=False, auto_adjust=False)
    prices = extract_close_prices(downloaded, yahoo_tickers)
    prices = prices.sort_index().ffill().dropna(axis=1, how="all")

    if prices.empty:
        raise ValueError("No usable close prices returned from yfinance")

    return prices
