"""Transaction-cost model for strategy backtests.

Costs are expressed in basis points and applied to one-way turnover. A daily
portfolio that changes from 40 percent AAPL to 20 percent AAPL has 20 percent
one-way turnover in AAPL. The total one-way turnover is half of the sum of
absolute weight changes because selling one asset and buying another both appear
in the raw weight differences.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CostModel:
    """Simple one-way trading-cost model.

    Parameters are annualization-independent. They are applied whenever portfolio
    weights change.
    """

    commission_bps: float = 0.0
    spread_bps: float = 0.0
    slippage_bps: float = 0.0
    tax_bps: float = 0.0

    @property
    def total_bps(self) -> float:
        """Total cost in basis points per one-way turnover."""

        return self.commission_bps + self.spread_bps + self.slippage_bps + self.tax_bps

    @property
    def total_rate(self) -> float:
        """Total cost as a decimal rate."""

        return self.total_bps / 10_000.0


def portfolio_turnover(weights: pd.DataFrame) -> pd.Series:
    """Calculate one-way turnover from target portfolio weights."""

    clean_weights = weights.fillna(0.0).astype(float)
    raw_weight_change = clean_weights.diff().abs().sum(axis=1)
    if raw_weight_change.empty:
        return pd.Series(dtype="float64")

    # Opening the initial portfolio is a trade. Subsequent rows use half of gross
    # absolute weight change so a rebalance from one fully invested asset to
    # another is 100 percent one-way turnover, not 200 percent.
    turnover = raw_weight_change / 2.0
    turnover.iloc[0] = clean_weights.iloc[0].abs().sum()
    return turnover


def transaction_cost_drag(weights: pd.DataFrame, cost_model: CostModel) -> pd.Series:
    """Return return drag caused by trading costs."""

    return portfolio_turnover(weights) * cost_model.total_rate
