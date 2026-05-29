import pandas as pd


def simple_moving_average(prices: pd.Series, window: int) -> pd.Series:
    """Calculate a simple moving average."""

    return prices.rolling(window=window).mean()


def exponential_moving_average(prices: pd.Series, span: int) -> pd.Series:
    """Calculate an exponential moving average."""

    return prices.ewm(span=span, adjust=False).mean()


def relative_strength_index(prices: pd.Series, window: int = 14) -> pd.Series:
    """Calculate RSI using average gains and losses."""

    delta = prices.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = gains.rolling(window=window).mean()
    average_loss = losses.rolling(window=window).mean()
    relative_strength = average_gain / average_loss
    return 100 - (100 / (1 + relative_strength))


def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Calculate MACD, signal line, and histogram."""

    fast_ema = exponential_moving_average(prices, fast)
    slow_ema = exponential_moving_average(prices, slow)
    macd_line = fast_ema - slow_ema
    signal_line = exponential_moving_average(macd_line, signal)
    histogram = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "signal": signal_line, "histogram": histogram})


def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> pd.DataFrame:
    """Calculate Bollinger middle, upper, and lower bands."""

    middle = simple_moving_average(prices, window)
    rolling_std = prices.rolling(window=window).std()
    upper = middle + num_std * rolling_std
    lower = middle - num_std * rolling_std
    return pd.DataFrame({"middle": middle, "upper": upper, "lower": lower})
