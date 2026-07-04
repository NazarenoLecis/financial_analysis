"""Walk-forward evaluation utilities.

These helpers make the in-sample/out-of-sample split explicit. They are designed
for strategy functions that accept a price window and return target weights for
that same window.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from strategy_testing.backtest_engine import BacktestResult, run_backtest
from strategy_testing.cost_model import CostModel


WeightBuilder = Callable[[pd.DataFrame], pd.DataFrame]


def expanding_walk_forward_backtest(
    prices: pd.DataFrame,
    build_weights: WeightBuilder,
    *,
    first_test_date: str,
    rebalance_frequency: str = "YE",
    strategy_name: str = "walk_forward_strategy",
    cost_model: CostModel | None = None,
    risk_free_rate: float = 0.0,
) -> BacktestResult:
    """Evaluate a strategy through expanding out-of-sample windows.

    The function rebuilds target weights using data available up to each
    rebalancing point, then applies the resulting rule until the next rebalancing
    point. This is a lightweight guardrail against pure in-sample chart fitting.
    """

    clean_prices = prices.sort_index().ffill().dropna(how="all")
    test_dates = clean_prices.loc[first_test_date:].resample(rebalance_frequency).last().index
    if len(test_dates) == 0:
        raise ValueError("No walk-forward test dates found. Check first_test_date and price history.")

    all_weights = []
    previous_date = None

    for test_date in test_dates:
        training_prices = clean_prices.loc[:test_date]
        candidate_weights = build_weights(training_prices)
        if candidate_weights.empty:
            continue

        latest_weights = candidate_weights.iloc[[-1]].copy()
        start_date = test_date if previous_date is None else previous_date
        end_date = test_date
        segment_index = clean_prices.loc[start_date:end_date].index
        repeated = pd.DataFrame(
            [latest_weights.iloc[0].to_dict()] * len(segment_index),
            index=segment_index,
        )
        all_weights.append(repeated)
        previous_date = test_date

    if not all_weights:
        raise ValueError("The weight builder did not return usable weights")

    target_weights = pd.concat(all_weights).sort_index()
    target_weights = target_weights[~target_weights.index.duplicated(keep="last")]
    return run_backtest(
        clean_prices,
        target_weights,
        strategy_name=strategy_name,
        cost_model=cost_model,
        risk_free_rate=risk_free_rate,
    )
