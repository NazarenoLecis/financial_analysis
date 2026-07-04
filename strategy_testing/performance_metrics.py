"""Performance metrics for strategy backtests.

The functions in this module are deliberately small and transparent. They report
the metrics that are needed to judge whether a trading or screening rule adds
value after risk and costs instead of only showing attractive charts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


TRADING_DAYS_PER_YEAR = 252


def equity_curve(returns: pd.Series, initial_value: float = 1.0) -> pd.Series:
    """Convert periodic returns into a cumulative equity curve."""

    clean_returns = returns.fillna(0.0).astype(float)
    return initial_value * (1.0 + clean_returns).cumprod()


def annualized_return(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calculate compound annual growth rate from periodic returns."""

    clean_returns = returns.dropna().astype(float)
    if clean_returns.empty:
        return math.nan

    total_return = float((1.0 + clean_returns).prod())
    years = len(clean_returns) / periods_per_year
    if years <= 0 or total_return <= 0:
        return math.nan
    return total_return ** (1.0 / years) - 1.0


def annualized_volatility(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    """Calculate annualized standard deviation of periodic returns."""

    clean_returns = returns.dropna().astype(float)
    if clean_returns.empty:
        return math.nan
    return float(clean_returns.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized Sharpe ratio using a constant annual risk-free rate."""

    clean_returns = returns.dropna().astype(float)
    if clean_returns.empty:
        return math.nan

    periodic_risk_free = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess_returns = clean_returns - periodic_risk_free
    volatility = excess_returns.std(ddof=1)
    if volatility == 0 or np.isnan(volatility):
        return math.nan
    return float(excess_returns.mean() / volatility * np.sqrt(periods_per_year))


def max_drawdown(returns: pd.Series) -> float:
    """Calculate the worst peak-to-trough loss of an equity curve."""

    curve = equity_curve(returns)
    if curve.empty:
        return math.nan
    running_max = curve.cummax()
    drawdowns = curve / running_max - 1.0
    return float(drawdowns.min())


def hit_rate(returns: pd.Series) -> float:
    """Share of periods with positive returns."""

    clean_returns = returns.dropna().astype(float)
    if clean_returns.empty:
        return math.nan
    return float((clean_returns > 0).mean())


def information_ratio(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Calculate annualized information ratio versus a benchmark."""

    aligned = pd.concat([strategy_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return math.nan

    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    tracking_error = active_returns.std(ddof=1)
    if tracking_error == 0 or np.isnan(tracking_error):
        return math.nan
    return float(active_returns.mean() / tracking_error * np.sqrt(periods_per_year))


def summarize_returns(
    returns: pd.Series,
    *,
    benchmark_returns: pd.Series | None = None,
    turnover: pd.Series | None = None,
    cost_drag: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> pd.Series:
    """Return a compact metrics table for one strategy return series."""

    clean_returns = returns.dropna().astype(float)
    summary = {
        "periods": float(len(clean_returns)),
        "annual_return": annualized_return(clean_returns, periods_per_year),
        "annual_volatility": annualized_volatility(clean_returns, periods_per_year),
        "sharpe_ratio": sharpe_ratio(clean_returns, risk_free_rate, periods_per_year),
        "max_drawdown": max_drawdown(clean_returns),
        "hit_rate": hit_rate(clean_returns),
    }

    if turnover is not None:
        summary["average_turnover"] = float(turnover.reindex(clean_returns.index).fillna(0.0).mean())
        summary["total_turnover"] = float(turnover.reindex(clean_returns.index).fillna(0.0).sum())

    if cost_drag is not None:
        summary["total_cost_drag"] = float(cost_drag.reindex(clean_returns.index).fillna(0.0).sum())
        summary["average_cost_drag"] = float(cost_drag.reindex(clean_returns.index).fillna(0.0).mean())

    if benchmark_returns is not None:
        aligned = pd.concat([clean_returns, benchmark_returns], axis=1, join="inner").dropna()
        if not aligned.empty:
            active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
            summary["benchmark_annual_return"] = annualized_return(aligned.iloc[:, 1], periods_per_year)
            summary["annual_excess_return"] = summary["annual_return"] - summary["benchmark_annual_return"]
            summary["tracking_error"] = annualized_volatility(active_returns, periods_per_year)
            summary["information_ratio"] = information_ratio(aligned.iloc[:, 0], aligned.iloc[:, 1], periods_per_year)

    return pd.Series(summary, dtype="float64")
