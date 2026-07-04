"""Fundamental signals that can be converted into dated strategy weights.

The functions use a reporting lag before a fiscal-period score becomes
investable. This is a practical guardrail against look-ahead bias when exact
filing dates are not available from the basic yfinance statement tables.
"""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from utils import fetch_financial_data, safe_divide, statement_value


def piotroski_score_for_period(data, position: int) -> int:
    """Calculate Piotroski F-score for a specific annual statement position."""

    periods = data.periods
    if position + 2 >= len(periods):
        raise ValueError("Piotroski score requires current, previous, and two-year-ago periods")

    current, previous, two_years_ago = periods[position], periods[position + 1], periods[position + 2]

    net_income = statement_value(data.income_statement, current, "Net Income")
    net_income_previous = statement_value(data.income_statement, previous, "Net Income")
    operating_cash_flow = statement_value(
        data.cash_flow,
        current,
        ["Operating Cash Flow", "Total Cash From Operating Activities"],
    )
    total_assets = statement_value(data.balance_sheet, current, "Total Assets")
    total_assets_previous = statement_value(data.balance_sheet, previous, "Total Assets")

    roa = safe_divide(net_income, total_assets, "current ROA")
    roa_previous = safe_divide(net_income_previous, total_assets_previous, "previous ROA")

    profitability = int(roa > 0)
    profitability += int(operating_cash_flow > 0)
    profitability += int(roa > roa_previous)
    profitability += int(operating_cash_flow > net_income)

    long_term_debt = statement_value(data.balance_sheet, current, "Long Term Debt", required=False, default=0)
    long_term_debt_previous = statement_value(data.balance_sheet, previous, "Long Term Debt", required=False, default=0)
    current_assets = statement_value(data.balance_sheet, current, ["Current Assets", "Total Current Assets"])
    current_liabilities = statement_value(data.balance_sheet, current, ["Current Liabilities", "Total Current Liabilities"])
    current_assets_previous = statement_value(data.balance_sheet, previous, ["Current Assets", "Total Current Assets"])
    current_liabilities_previous = statement_value(data.balance_sheet, previous, ["Current Liabilities", "Total Current Liabilities"])

    current_ratio = safe_divide(current_assets, current_liabilities, "current ratio")
    current_ratio_previous = safe_divide(current_assets_previous, current_liabilities_previous, "previous current ratio")

    shares = statement_value(
        data.balance_sheet,
        current,
        ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
    )
    shares_previous = statement_value(
        data.balance_sheet,
        previous,
        ["Ordinary Shares Number", "Share Issued", "Common Stock Shares Outstanding"],
    )

    leverage_liquidity = int(long_term_debt <= long_term_debt_previous)
    leverage_liquidity += int(current_ratio > current_ratio_previous)
    leverage_liquidity += int(shares <= shares_previous)

    sales = statement_value(data.income_statement, current, ["Total Revenue", "Revenue"])
    sales_previous = statement_value(data.income_statement, previous, ["Total Revenue", "Revenue"])
    cost_of_revenue = statement_value(data.income_statement, current, ["Cost Of Revenue", "Cost Of Goods Sold"])
    cost_of_revenue_previous = statement_value(
        data.income_statement,
        previous,
        ["Cost Of Revenue", "Cost Of Goods Sold"],
    )

    gross_margin = safe_divide(sales - cost_of_revenue, sales, "current gross margin")
    gross_margin_previous = safe_divide(sales_previous - cost_of_revenue_previous, sales_previous, "previous gross margin")

    total_assets_two_years_ago = statement_value(data.balance_sheet, two_years_ago, "Total Assets")
    average_assets = (total_assets + total_assets_previous) / 2
    average_assets_previous = (total_assets_previous + total_assets_two_years_ago) / 2
    asset_turnover = safe_divide(sales, average_assets, "current asset turnover")
    asset_turnover_previous = safe_divide(sales_previous, average_assets_previous, "previous asset turnover")

    operating_efficiency = int(gross_margin > gross_margin_previous)
    operating_efficiency += int(asset_turnover > asset_turnover_previous)

    return profitability + leverage_liquidity + operating_efficiency


def piotroski_score_history(ticker: str, reporting_lag_days: int = 120) -> pd.DataFrame:
    """Return dated Piotroski scores for one ticker."""

    data = fetch_financial_data(ticker)
    rows = []

    for position in range(0, max(len(data.periods) - 2, 0)):
        fiscal_period = pd.to_datetime(data.periods[position])
        try:
            score = piotroski_score_for_period(data, position)
        except Exception as exc:
            rows.append(
                {
                    "ticker": data.yahoo_ticker,
                    "fiscal_period": fiscal_period,
                    "signal_date": fiscal_period + timedelta(days=reporting_lag_days),
                    "piotroski_f_score": pd.NA,
                    "error": str(exc),
                }
            )
            continue

        rows.append(
            {
                "ticker": data.yahoo_ticker,
                "fiscal_period": fiscal_period,
                "signal_date": fiscal_period + timedelta(days=reporting_lag_days),
                "piotroski_f_score": score,
                "error": pd.NA,
            }
        )

    return pd.DataFrame(rows)


def build_piotroski_score_panel(tickers: list[str], reporting_lag_days: int = 120) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a signal-date by ticker score panel plus an error table."""

    frames = []
    errors = []

    for ticker in tickers:
        try:
            history = piotroski_score_history(ticker, reporting_lag_days=reporting_lag_days)
            if history.empty:
                errors.append({"ticker": ticker, "error": "No Piotroski score history"})
                continue
            frames.append(history)
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})

    if not frames:
        return pd.DataFrame(), pd.DataFrame(errors)

    long_scores = pd.concat(frames, ignore_index=True)
    usable = long_scores.dropna(subset=["piotroski_f_score"])
    panel = usable.pivot_table(
        index="signal_date",
        columns="ticker",
        values="piotroski_f_score",
        aggfunc="last",
    ).sort_index()

    error_table = pd.concat(
        [
            pd.DataFrame(errors),
            long_scores.dropna(subset=["error"])[["ticker", "error"]].drop_duplicates(),
        ],
        ignore_index=True,
    )
    return panel, error_table
