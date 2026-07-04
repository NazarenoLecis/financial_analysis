"""Robustness checks for strategy parameter grids."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from itertools import product

import pandas as pd

from strategy_testing.backtest_engine import run_backtest
from strategy_testing.cost_model import CostModel


def parameter_grid(grid: dict[str, Iterable]) -> list[dict]:
    """Expand a dictionary of parameter lists into a list of parameter dictionaries."""

    keys = list(grid)
    values = [list(grid[key]) for key in keys]
    return [dict(zip(keys, combination)) for combination in product(*values)]


def evaluate_parameter_grid(
    prices: pd.DataFrame,
    build_weights: Callable[..., pd.DataFrame],
    grid: dict[str, Iterable],
    *,
    cost_model: CostModel | None = None,
    risk_free_rate: float = 0.0,
    metric: str = "sharpe_ratio",
) -> pd.DataFrame:
    """Backtest every parameter combination and return ranked metrics.

    This does not prove that a strategy is valid. It exposes fragility: if a rule
    works only for one arbitrary parameter choice, the evidence is weak.
    """

    rows = []
    for parameters in parameter_grid(grid):
        try:
            weights = build_weights(prices, **parameters)
            result = run_backtest(
                prices,
                weights,
                strategy_name="grid_strategy",
                cost_model=cost_model,
                risk_free_rate=risk_free_rate,
            )
            row = dict(parameters)
            row.update(result.summary.to_dict())
            rows.append(row)
        except Exception as exc:
            row = dict(parameters)
            row["error"] = str(exc)
            rows.append(row)

    results = pd.DataFrame(rows)
    if metric in results.columns:
        results = results.sort_values(metric, ascending=False)
    return results
