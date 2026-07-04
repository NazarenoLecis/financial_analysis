"""Convert technical indicators into dated portfolio weights for backtests."""

from __future__ import annotations

import pandas as pd

from technical_analysis.indicators import relative_strength_index


def _equal_weight_active_signals(active: pd.DataFrame) -> pd.DataFrame:
    """Convert boolean signals into equal-weight target portfolios."""

    active = active.fillna(False).astype(bool)
    counts = active.sum(axis=1).replace(0, pd.NA)
    return active.astype(float).div(counts, axis=0).fillna(0.0)


def sma_crossover_weights(
    prices: pd.DataFrame,
    *,
    short_window: int = 50,
    long_window: int = 200,
) -> pd.DataFrame:
    """Long assets whose short moving average is above the long moving average."""

    if short_window >= long_window:
        raise ValueError("short_window must be smaller than long_window")

    short_sma = prices.rolling(short_window).mean()
    long_sma = prices.rolling(long_window).mean()
    return _equal_weight_active_signals(short_sma > long_sma)


def time_series_momentum_weights(
    prices: pd.DataFrame,
    *,
    lookback_days: int = 252,
    skip_days: int = 21,
) -> pd.DataFrame:
    """Long assets with positive own past return over a lookback window."""

    if lookback_days <= skip_days:
        raise ValueError("lookback_days must be greater than skip_days")

    formation_prices = prices.shift(skip_days)
    past_returns = formation_prices / prices.shift(lookback_days) - 1.0
    return _equal_weight_active_signals(past_returns > 0)


def rsi_reversal_weights(
    prices: pd.DataFrame,
    *,
    window: int = 14,
    lower: float = 30.0,
    upper: float = 70.0,
) -> pd.DataFrame:
    """Contrarian RSI rule for empirical testing."""

    if lower >= upper:
        raise ValueError("lower must be smaller than upper")

    rsi = prices.apply(lambda column: relative_strength_index(column, window))
    weights = _equal_weight_active_signals(rsi < lower)
    weights = weights.mask(rsi > upper, 0.0)
    return weights.fillna(0.0)
