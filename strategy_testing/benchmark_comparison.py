"""Benchmark helpers for strategy comparisons."""

from __future__ import annotations

import pandas as pd

from strategy_testing.backtest_engine import price_returns
from strategy_testing.performance_metrics import summarize_returns


def equal_weight_benchmark_returns(prices: pd.DataFrame) -> pd.Series:
    """Calculate daily returns of a simple equal-weight benchmark.

    The benchmark uses constant equal weights each day. It is deliberately simple
    so active strategy results are compared with a transparent passive rule over
    the same downloaded universe.
    """

    returns = price_returns(prices)
    if returns.empty:
        return pd.Series(dtype="float64", name="equal_weight_benchmark")

    weights = pd.Series(1.0 / len(returns.columns), index=returns.columns)
    return (returns * weights).sum(axis=1).rename("equal_weight_benchmark")


def compare_to_benchmark(
    strategy_returns: pd.Series,
    benchmark_returns: pd.Series,
    *,
    risk_free_rate: float = 0.0,
) -> pd.DataFrame:
    """Return side-by-side metrics for a strategy and benchmark."""

    strategy_summary = summarize_returns(
        strategy_returns,
        benchmark_returns=benchmark_returns,
        risk_free_rate=risk_free_rate,
    ).rename("strategy")
    benchmark_summary = summarize_returns(
        benchmark_returns,
        risk_free_rate=risk_free_rate,
    ).rename("benchmark")

    return pd.concat([strategy_summary, benchmark_summary], axis=1)
