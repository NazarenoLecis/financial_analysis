"""Run empirical tests for indicator-based rules."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from strategy_testing.backtest_engine import run_backtest
from strategy_testing.benchmark_comparison import equal_weight_benchmark_returns
from strategy_testing.cost_model import CostModel
from strategy_testing.data import fetch_close_prices, resolve_ticker_universe
from technical_analysis.strategy_rules import rsi_reversal_weights, sma_crossover_weights, time_series_momentum_weights
from utils import RichHelpFormatter


def build_weights(prices: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    if args.rule == "sma_crossover":
        return sma_crossover_weights(prices, short_window=args.short_window, long_window=args.long_window)
    if args.rule == "time_series_momentum":
        return time_series_momentum_weights(prices, lookback_days=args.lookback_days, skip_days=args.skip_days)
    if args.rule == "rsi_reversal":
        return rsi_reversal_weights(prices, window=args.rsi_window, lower=args.rsi_lower, upper=args.rsi_upper)
    raise ValueError(f"Unknown rule: {args.rule}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate indicator rules with explicit costs and benchmarks.",
        formatter_class=RichHelpFormatter,
    )
    parser.add_argument("--index", choices=["sp500", "nasdaq100"], default="sp500")
    parser.add_argument("--tickers", nargs="+")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start-date", default="2015-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--rule", choices=["sma_crossover", "time_series_momentum", "rsi_reversal"], default="sma_crossover")
    parser.add_argument("--short-window", type=int, default=50)
    parser.add_argument("--long-window", type=int, default=200)
    parser.add_argument("--lookback-days", type=int, default=252)
    parser.add_argument("--skip-days", type=int, default=21)
    parser.add_argument("--rsi-window", type=int, default=14)
    parser.add_argument("--rsi-lower", type=float, default=30.0)
    parser.add_argument("--rsi-upper", type=float, default=70.0)
    parser.add_argument("--commission-bps", type=float, default=0.0)
    parser.add_argument("--spread-bps", type=float, default=0.0)
    parser.add_argument("--slippage-bps", type=float, default=0.0)
    parser.add_argument("--tax-bps", type=float, default=0.0)
    parser.add_argument("--risk-free-rate", type=float, default=0.02)
    parser.add_argument("--output-csv")
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tickers = resolve_ticker_universe(tickers=args.tickers, index=args.index, limit=args.limit)
    prices = fetch_close_prices(tickers, args.start_date, args.end_date)
    weights = build_weights(prices, args)
    benchmark_returns = equal_weight_benchmark_returns(prices)
    cost_model = CostModel(args.commission_bps, args.spread_bps, args.slippage_bps, args.tax_bps)

    result = run_backtest(
        prices,
        weights,
        strategy_name=args.rule,
        cost_model=cost_model,
        benchmark_returns=benchmark_returns,
        risk_free_rate=args.risk_free_rate,
    )

    print("\nNet performance summary:")
    print(result.summary.to_string(float_format=lambda value: f"{value:.4f}"))

    output = pd.concat(
        [result.gross_returns, result.net_returns, benchmark_returns, result.turnover, result.cost_drag, result.equity_curve_net],
        axis=1,
    )
    if args.output_csv:
        output.to_csv(args.output_csv)
        print(f"\nSaved output to {args.output_csv}")

    if not args.no_plot:
        ax = result.equity_curve_net.plot(figsize=(11, 6), label=f"{args.rule} net")
        (1.0 + benchmark_returns.fillna(0.0)).cumprod().plot(ax=ax, label="equal-weight benchmark")
        ax.set_title(f"{args.rule}: net equity curve vs benchmark")
        ax.set_ylabel("Growth of 1")
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
