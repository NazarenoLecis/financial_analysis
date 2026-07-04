"""Reusable backtest engine for technical and fundamental strategies.

The engine receives prices and target weights. It shifts the weights by one
period before applying returns, which prevents a signal calculated with today's
close from trading at that same close. The result is a simple daily close-to-close
backtest suitable for comparing strategy rules, not an execution simulator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from strategy_testing.cost_model import CostModel, portfolio_turnover, transaction_cost_drag
from strategy_testing.performance_metrics import equity_curve, summarize_returns


@dataclass(frozen=True)
class BacktestResult:
    """Container returned by `run_backtest`."""

    strategy_name: str
    gross_returns: pd.Series
    net_returns: pd.Series
    asset_returns: pd.DataFrame
    weights: pd.DataFrame
    executed_weights: pd.DataFrame
    turnover: pd.Series
    cost_drag: pd.Series
    equity_curve_gross: pd.Series
    equity_curve_net: pd.Series
    summary: pd.Series


def price_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Convert a price table into simple daily returns."""

    clean_prices = prices.sort_index().ffill().dropna(how="all")
    return clean_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="all")


def normalize_weights(weights: pd.DataFrame, allow_leverage: bool = False) -> pd.DataFrame:
    """Clean target weights and optionally cap gross exposure at 100 percent."""

    clean_weights = weights.sort_index().replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    if allow_leverage:
        return clean_weights

    gross_exposure = clean_weights.abs().sum(axis=1)
    scale = gross_exposure.where(gross_exposure > 1.0, 1.0)
    return clean_weights.div(scale.replace(0.0, 1.0), axis=0)


def run_backtest(
    prices: pd.DataFrame,
    target_weights: pd.DataFrame,
    *,
    strategy_name: str = "strategy",
    cost_model: CostModel | None = None,
    benchmark_returns: pd.Series | None = None,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
    allow_leverage: bool = False,
) -> BacktestResult:
    """Run a close-to-close portfolio backtest.

    `target_weights` must have dates as index and tickers as columns. Weights are
    forward-filled onto the return index. Positions are executed with a one-period
    lag to reduce look-ahead bias.
    """

    cost_model = cost_model or CostModel()
    returns = price_returns(prices)
    if returns.empty:
        raise ValueError("Not enough price data to calculate returns")

    aligned_weights = target_weights.reindex(returns.index).ffill().fillna(0.0)
    aligned_weights = normalize_weights(aligned_weights, allow_leverage=allow_leverage)
    aligned_weights = aligned_weights.reindex(columns=returns.columns, fill_value=0.0)

    executed_weights = aligned_weights.shift(1).fillna(0.0)
    gross_returns = (executed_weights * returns).sum(axis=1)
    turnover = portfolio_turnover(aligned_weights).reindex(gross_returns.index).fillna(0.0)
    cost_drag = transaction_cost_drag(aligned_weights, cost_model).reindex(gross_returns.index).fillna(0.0)
    net_returns = gross_returns - cost_drag

    summary = summarize_returns(
        net_returns,
        benchmark_returns=benchmark_returns,
        turnover=turnover,
        cost_drag=cost_drag,
        risk_free_rate=risk_free_rate,
        periods_per_year=periods_per_year,
    )

    return BacktestResult(
        strategy_name=strategy_name,
        gross_returns=gross_returns.rename(f"{strategy_name}_gross"),
        net_returns=net_returns.rename(f"{strategy_name}_net"),
        asset_returns=returns,
        weights=aligned_weights,
        executed_weights=executed_weights,
        turnover=turnover.rename("turnover"),
        cost_drag=cost_drag.rename("cost_drag"),
        equity_curve_gross=equity_curve(gross_returns).rename(f"{strategy_name}_gross_equity"),
        equity_curve_net=equity_curve(net_returns).rename(f"{strategy_name}_net_equity"),
        summary=summary.rename(strategy_name),
    )
