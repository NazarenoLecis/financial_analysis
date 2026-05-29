"""Reusable technical indicators.

The scripts in this folder import these functions instead of repeating the same
math in several places. Each function receives a pandas Series of prices and
returns either a Series or DataFrame that can be printed, analyzed, or plotted.
"""

import pandas as pd


def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Calculate a simple moving average.

    A simple moving average smooths price data by taking the average of the last
    `window` closing prices. For example, a 20-day SMA is the average close over
    the most recent 20 trading days.
    """

    return prices.rolling(window=window).mean()


def exponential_moving_average(prices: pd.Series, span: int) -> pd.Series:
    """Calculate an exponential moving average.

    An EMA gives more weight to recent prices, so it reacts faster to new price
    movements than a simple moving average.
    """

    return prices.ewm(span=span, adjust=False).mean()


def relative_strength_index(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using average gains and losses.

    RSI is a momentum indicator on a 0 to 100 scale. Readings above 70 are often
    treated as overbought, while readings below 30 are often treated as oversold.
    """

    # Price changes from one day to the next.
    delta = prices.diff()

    # Separate positive moves from negative moves. Losses are converted to
    # positive numbers so average gains and average losses can be compared.
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Use rolling averages over the selected window, usually 14 days.
    average_gain = gains.rolling(window=window).mean()
    average_loss = losses.rolling(window=window).mean()
    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD, signal line, and histogram.

    MACD compares a fast EMA with a slow EMA. When the MACD line is above the
    signal line, traders often read it as positive short-term momentum.
    """

    # Standard MACD uses 12-day and 26-day EMAs, then a 9-day EMA signal line.
    fast_ema = exponential_moving_average(prices, fast)
    slow_ema = exponential_moving_average(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = exponential_moving_average(macd_line, signal)

    # The histogram shows the distance between MACD and its signal line.
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger middle, upper, and lower bands.

    The middle band is usually a 20-day SMA. The upper and lower bands are set a
    chosen number of standard deviations away from that middle band.
    """

    middle = simple_moving_average(prices, window)
    rolling_std = prices.rolling(window=window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})
