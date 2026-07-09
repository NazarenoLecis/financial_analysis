"""Derivative pricing models for plain vanilla equity options.

The script covers the most common educational pricing methods:

1. Black-Scholes for European call and put options;
2. Cox-Ross-Rubinstein binomial trees for European and American options;
3. Monte Carlo simulation for European options under geometric Brownian motion;
4. Black-Scholes implied volatility from an observed market option price.

Inputs can be provided manually, or the script can fetch the latest spot price and
estimate annualized volatility from yfinance historical prices.
"""

import argparse
import math
import sys
from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to import path so direct file execution can import utils.py.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import RichHelpFormatter, fetch_price_history

TRADING_DAYS_PER_YEAR = 252


@dataclass(frozen=True)
class OptionInputs:
    """Inputs shared by the option pricing models."""

    option_type: str
    spot: float
    strike: float
    maturity: float
    risk_free_rate: float
    volatility: float
    dividend_yield: float = 0.0


def validate_option_inputs(inputs: OptionInputs) -> None:
    """Reject inputs that would make the pricing formulas invalid."""

    if inputs.option_type not in {"call", "put"}:
        raise ValueError("option_type must be 'call' or 'put'")
    if inputs.spot <= 0:
        raise ValueError("spot must be positive")
    if inputs.strike <= 0:
        raise ValueError("strike must be positive")
    if inputs.maturity <= 0:
        raise ValueError("maturity must be positive and expressed in years")
    if inputs.volatility <= 0:
        raise ValueError("volatility must be positive and expressed as a decimal")


def normal_cdf(value: float) -> float:
    """Standard normal cumulative distribution function."""

    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def normal_pdf(value: float) -> float:
    """Standard normal probability density function."""

    return math.exp(-0.5 * value * value) / math.sqrt(2.0 * math.pi)


def d1_d2(inputs: OptionInputs) -> tuple[float, float]:
    """Return the Black-Scholes d1 and d2 terms."""

    validate_option_inputs(inputs)
    sigma_sqrt_t = inputs.volatility * math.sqrt(inputs.maturity)
    d1 = (
        math.log(inputs.spot / inputs.strike)
        + (inputs.risk_free_rate - inputs.dividend_yield + 0.5 * inputs.volatility**2) * inputs.maturity
    ) / sigma_sqrt_t
    d2 = d1 - sigma_sqrt_t
    return d1, d2


def black_scholes_price(inputs: OptionInputs) -> dict[str, float]:
    """Price a European option and return the main Greeks."""

    validate_option_inputs(inputs)
    d1, d2 = d1_d2(inputs)
    discounted_spot = inputs.spot * math.exp(-inputs.dividend_yield * inputs.maturity)
    discounted_strike = inputs.strike * math.exp(-inputs.risk_free_rate * inputs.maturity)

    if inputs.option_type == "call":
        price = discounted_spot * normal_cdf(d1) - discounted_strike * normal_cdf(d2)
        delta = math.exp(-inputs.dividend_yield * inputs.maturity) * normal_cdf(d1)
        theta = (
            -(discounted_spot * normal_pdf(d1) * inputs.volatility) / (2 * math.sqrt(inputs.maturity))
            - inputs.risk_free_rate * discounted_strike * normal_cdf(d2)
            + inputs.dividend_yield * discounted_spot * normal_cdf(d1)
        ) / 365
        rho = inputs.strike * inputs.maturity * math.exp(-inputs.risk_free_rate * inputs.maturity) * normal_cdf(d2)
    else:
        price = discounted_strike * normal_cdf(-d2) - discounted_spot * normal_cdf(-d1)
        delta = math.exp(-inputs.dividend_yield * inputs.maturity) * (normal_cdf(d1) - 1)
        theta = (
            -(discounted_spot * normal_pdf(d1) * inputs.volatility) / (2 * math.sqrt(inputs.maturity))
            + inputs.risk_free_rate * discounted_strike * normal_cdf(-d2)
            - inputs.dividend_yield * discounted_spot * normal_cdf(-d1)
        ) / 365
        rho = -inputs.strike * inputs.maturity * math.exp(-inputs.risk_free_rate * inputs.maturity) * normal_cdf(-d2)

    gamma = math.exp(-inputs.dividend_yield * inputs.maturity) * normal_pdf(d1) / (
        inputs.spot * inputs.volatility * math.sqrt(inputs.maturity)
    )
    vega = inputs.spot * math.exp(-inputs.dividend_yield * inputs.maturity) * normal_pdf(d1) * math.sqrt(inputs.maturity)

    return {
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega_per_1pct_vol": vega / 100,
        "theta_per_day": theta,
        "rho_per_1pct_rate": rho / 100,
        "d1": d1,
        "d2": d2,
    }


def binomial_tree_price(inputs: OptionInputs, steps: int = 200, exercise_style: str = "european") -> float:
    """Price an option with the Cox-Ross-Rubinstein binomial tree."""

    validate_option_inputs(inputs)
    if exercise_style not in {"european", "american"}:
        raise ValueError("exercise_style must be 'european' or 'american'")
    if steps <= 0:
        raise ValueError("steps must be positive")

    dt = inputs.maturity / steps
    up = math.exp(inputs.volatility * math.sqrt(dt))
    down = 1 / up
    discount = math.exp(-inputs.risk_free_rate * dt)
    growth = math.exp((inputs.risk_free_rate - inputs.dividend_yield) * dt)
    probability_up = (growth - down) / (up - down)

    if probability_up < 0 or probability_up > 1:
        raise ValueError(
            "Invalid risk-neutral probability. Try more steps or check rates, dividend yield, and volatility."
        )

    up_moves = np.arange(steps + 1)
    terminal_spots = inputs.spot * (up**up_moves) * (down ** (steps - up_moves))
    option_values = intrinsic_value(terminal_spots, inputs.strike, inputs.option_type)

    for step in range(steps - 1, -1, -1):
        option_values = discount * (probability_up * option_values[1:] + (1 - probability_up) * option_values[:-1])

        if exercise_style == "american":
            up_moves = np.arange(step + 1)
            node_spots = inputs.spot * (up**up_moves) * (down ** (step - up_moves))
            option_values = np.maximum(option_values, intrinsic_value(node_spots, inputs.strike, inputs.option_type))

    return float(option_values[0])


def intrinsic_value(spots: np.ndarray, strike: float, option_type: str) -> np.ndarray:
    """Return option payoff at expiry or immediate exercise value."""

    if option_type == "call":
        return np.maximum(spots - strike, 0)
    return np.maximum(strike - spots, 0)


def monte_carlo_price(inputs: OptionInputs, simulations: int = 10_000, seed: int | None = 42) -> dict[str, float]:
    """Price a European option with risk-neutral Monte Carlo simulation."""

    validate_option_inputs(inputs)
    if simulations <= 0:
        raise ValueError("simulations must be positive")

    rng = np.random.default_rng(seed)
    random_draws = rng.standard_normal(simulations)
    terminal_spots = inputs.spot * np.exp(
        (inputs.risk_free_rate - inputs.dividend_yield - 0.5 * inputs.volatility**2) * inputs.maturity
        + inputs.volatility * math.sqrt(inputs.maturity) * random_draws
    )
    payoffs = intrinsic_value(terminal_spots, inputs.strike, inputs.option_type)
    discounted_payoffs = math.exp(-inputs.risk_free_rate * inputs.maturity) * payoffs

    return {
        "price": float(np.mean(discounted_payoffs)),
        "standard_error": float(np.std(discounted_payoffs, ddof=1) / math.sqrt(simulations)),
    }


def implied_volatility(
    inputs: OptionInputs,
    market_price: float,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 100,
    lower_bound: float = 1e-6,
    upper_bound: float = 5.0,
) -> float:
    """Solve Black-Scholes implied volatility with a bisection search."""

    if market_price <= 0:
        raise ValueError("market_price must be positive")

    low_price = black_scholes_price(replace(inputs, volatility=lower_bound))["price"]
    high_price = black_scholes_price(replace(inputs, volatility=upper_bound))["price"]
    if market_price < low_price or market_price > high_price:
        raise ValueError(
            "market_price is outside the range produced by the volatility bounds. "
            "Check the option inputs or widen the volatility search bounds."
        )

    low = lower_bound
    high = upper_bound
    for _ in range(max_iterations):
        mid = (low + high) / 2
        mid_price = black_scholes_price(replace(inputs, volatility=mid))["price"]
        if abs(mid_price - market_price) < tolerance:
            return mid
        if mid_price < market_price:
            low = mid
        else:
            high = mid

    return (low + high) / 2


def latest_spot_price(ticker: str, start_date: str | None = None, end_date: str | None = None) -> float:
    """Fetch the latest available close price for a ticker."""

    data = fetch_price_history(ticker, start_date=start_date, end_date=end_date)
    price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
    prices = data[price_column].dropna()
    if prices.empty:
        raise ValueError(f"No close prices found for {ticker}")
    return float(prices.iloc[-1])


def estimate_annual_volatility(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
    lookback_days: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Estimate annualized volatility from historical daily log returns."""

    if lookback_days <= 1:
        raise ValueError("lookback_days must be greater than 1")

    if start_date is None:
        # Calendar days exceed trading days. The buffer increases the chance of
        # retrieving enough observations around weekends and market holidays.
        start_date = (date.today() - timedelta(days=int(lookback_days * 2.2))).isoformat()

    data = fetch_price_history(ticker, start_date=start_date, end_date=end_date)
    price_column = "Adj Close" if "Adj Close" in data.columns else "Close"
    prices = data[price_column].dropna().tail(lookback_days + 1)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 2:
        raise ValueError(f"Not enough historical prices to estimate volatility for {ticker}")

    return float(log_returns.std(ddof=1) * math.sqrt(TRADING_DAYS_PER_YEAR))


def build_inputs(args: argparse.Namespace) -> tuple[OptionInputs, dict[str, str]]:
    """Combine manual inputs and yfinance-derived market inputs."""

    sources = {}
    spot = args.spot
    volatility = args.volatility

    if spot is None:
        spot = latest_spot_price(args.ticker, start_date=args.start_date, end_date=args.end_date)
        sources["spot"] = f"latest close from yfinance for {args.ticker}"
    else:
        sources["spot"] = "manual input"

    if volatility is None:
        volatility = estimate_annual_volatility(
            args.ticker,
            start_date=args.start_date,
            end_date=args.end_date,
            lookback_days=args.vol_lookback_days,
        )
        sources["volatility"] = f"annualized historical volatility from yfinance for {args.ticker}"
    else:
        sources["volatility"] = "manual input"

    inputs = OptionInputs(
        option_type=args.option_type,
        spot=float(spot),
        strike=args.strike,
        maturity=args.maturity,
        risk_free_rate=args.risk_free_rate,
        volatility=float(volatility),
        dividend_yield=args.dividend_yield,
    )
    validate_option_inputs(inputs)
    return inputs, sources


def run_models(args: argparse.Namespace, inputs: OptionInputs) -> pd.DataFrame:
    """Run the selected pricing models and return comparable results."""

    requested_models = ["black-scholes", "binomial", "monte-carlo"] if args.model == "all" else [args.model]
    rows = []

    for model in requested_models:
        if args.style == "american" and model in {"black-scholes", "monte-carlo"}:
            if args.model == "all":
                continue
            raise ValueError(f"{model} supports European options only. Use --model binomial for American options.")

        if model == "black-scholes":
            result = black_scholes_price(inputs)
            rows.append(
                {
                    "model": "Black-Scholes",
                    "style": "european",
                    "price": result["price"],
                    "standard_error": np.nan,
                    "delta": result["delta"],
                    "gamma": result["gamma"],
                    "vega_per_1pct_vol": result["vega_per_1pct_vol"],
                    "theta_per_day": result["theta_per_day"],
                    "rho_per_1pct_rate": result["rho_per_1pct_rate"],
                }
            )
        elif model == "binomial":
            price = binomial_tree_price(inputs, steps=args.steps, exercise_style=args.style)
            rows.append(
                {
                    "model": f"Binomial tree ({args.steps} steps)",
                    "style": args.style,
                    "price": price,
                    "standard_error": np.nan,
                    "delta": np.nan,
                    "gamma": np.nan,
                    "vega_per_1pct_vol": np.nan,
                    "theta_per_day": np.nan,
                    "rho_per_1pct_rate": np.nan,
                }
            )
        elif model == "monte-carlo":
            result = monte_carlo_price(inputs, simulations=args.simulations, seed=args.seed)
            rows.append(
                {
                    "model": f"Monte Carlo ({args.simulations} simulations)",
                    "style": "european",
                    "price": result["price"],
                    "standard_error": result["standard_error"],
                    "delta": np.nan,
                    "gamma": np.nan,
                    "vega_per_1pct_vol": np.nan,
                    "theta_per_day": np.nan,
                    "rho_per_1pct_rate": np.nan,
                }
            )
        else:
            raise ValueError(f"Unknown model: {model}")

    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Price plain vanilla equity options with Black-Scholes, binomial tree, or Monte Carlo.",
        formatter_class=RichHelpFormatter,
        epilog="""
Inputs:
  Rates and volatility are decimals, not percentages. Use 0.04 for 4 percent.
  Maturity is in years. Use 0.5 for six months and 1.0 for one year.
  If --spot or --volatility are omitted, the script uses yfinance market data.

Models:
  black-scholes prices European call and put options and returns Greeks.
  binomial prices European or American options.
  monte-carlo prices European options and returns a simulation standard error.

Examples:
  python derivative_pricing/option_pricing.py
  python derivative_pricing/option_pricing.py --ticker MSFT --strike 450 --maturity 0.5 --model all
  python derivative_pricing/option_pricing.py --spot 100 --strike 105 --volatility 0.20 --risk-free-rate 0.04 --model black-scholes
  python derivative_pricing/option_pricing.py --spot 100 --strike 105 --volatility 0.20 --style american --model binomial
  python derivative_pricing/option_pricing.py --spot 100 --strike 105 --volatility 0.20 --market-price 7.5
""",
    )

    # Defaults make the script runnable from VS Code without arguments.
    parser.add_argument("--ticker", default="AAPL", help="Yahoo Finance ticker used when spot or volatility are not provided manually.")
    parser.add_argument("--spot", type=float, default=None, help="Current underlying price. If omitted, latest yfinance close is used.")
    parser.add_argument("--strike", type=float, default=220.0, help="Option strike price.")
    parser.add_argument("--maturity", type=float, default=0.5, help="Time to maturity in years.")
    parser.add_argument("--risk-free-rate", type=float, default=0.04, help="Continuously compounded risk-free rate.")
    parser.add_argument("--dividend-yield", type=float, default=0.0, help="Continuous dividend yield.")
    parser.add_argument("--volatility", type=float, default=None, help="Annualized volatility. If omitted, historical volatility is estimated.")
    parser.add_argument("--vol-lookback-days", type=int, default=TRADING_DAYS_PER_YEAR, help="Trading-day window used to estimate volatility.")
    parser.add_argument("--start-date", default=None, help="Optional start date for yfinance price history, YYYY-MM-DD.")
    parser.add_argument("--end-date", default=None, help="Optional end date for yfinance price history, YYYY-MM-DD.")
    parser.add_argument("--option-type", choices=["call", "put"], default="call", help="Option payoff type.")
    parser.add_argument("--style", choices=["european", "american"], default="european", help="Exercise style.")
    parser.add_argument(
        "--model",
        choices=["black-scholes", "binomial", "monte-carlo", "all"],
        default="all",
        help="Pricing model to run.",
    )
    parser.add_argument("--steps", type=int, default=200, help="Number of binomial tree steps.")
    parser.add_argument("--simulations", type=int, default=10_000, help="Number of Monte Carlo simulations.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used by Monte Carlo simulation.")
    parser.add_argument("--market-price", type=float, default=None, help="Observed market option price used to calculate implied volatility.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    inputs, sources = build_inputs(args)
    results = run_models(args, inputs)

    assumptions = pd.DataFrame(
        [
            {"input": "option_type", "value": inputs.option_type, "source": "manual/default"},
            {"input": "spot", "value": inputs.spot, "source": sources["spot"]},
            {"input": "strike", "value": inputs.strike, "source": "manual/default"},
            {"input": "maturity_years", "value": inputs.maturity, "source": "manual/default"},
            {"input": "risk_free_rate", "value": inputs.risk_free_rate, "source": "manual/default"},
            {"input": "dividend_yield", "value": inputs.dividend_yield, "source": "manual/default"},
            {"input": "volatility", "value": inputs.volatility, "source": sources["volatility"]},
        ]
    )

    print("Assumptions:")
    print(assumptions.to_string(index=False))
    print("\nPricing results:")
    print(results.to_string(index=False, float_format=lambda value: f"{value:,.6f}"))

    if args.market_price is not None:
        if args.style != "european":
            print("\nImplied volatility is shown only for European Black-Scholes inputs.")
        else:
            implied_vol = implied_volatility(inputs, args.market_price)
            print(f"\nBlack-Scholes implied volatility from market price {args.market_price:,.4f}: {implied_vol:,.6f}")

    print("\nNote: outputs are model estimates based on simplified assumptions, not trading advice.")


if __name__ == "__main__":
    main()
