import numpy as np
import pandas as pd
import pandas_ta as ta


def label_by_fixed_horizon(df: pd.DataFrame, horizon: int, threshold: float, price_col: str = "close") -> pd.Series:
    """
    Labels time series data based on fixed-time horizon returns.

    Args:
        df (pd.DataFrame): DataFrame containing price data.
        horizon (int): Number of future bars to calculate returns.
        threshold (float): Threshold for classifying returns as positive or negative.
        price_col (str): Column name of the price data.

    Returns:
        pd.Series: A Series of labels (-1, 0, or 1) indexed to the original DataFrame.
                   -1: Negative return below the threshold.
                    0: Return within the threshold range.
                    1: Positive return above the threshold.
    """
    prices = df[price_col]
    future_returns = (prices.shift(-horizon) - prices) / prices  # % return

    labels = np.where(future_returns > threshold, 1, np.where(future_returns < -threshold, -1, 0))

    return pd.Series(labels, index=df.index, name="label")


def label_by_absolute_threshold(
    df: pd.DataFrame, threshold: float, horizon: int, price_col: str = "Close"
) -> pd.DataFrame:
    """
    Labels data based on absolute price movement over a fixed horizon.

    Args:
        df (pd.DataFrame): DataFrame containing price data.
        threshold (float): Absolute price movement threshold.
        horizon (int): Number of future bars to evaluate price movement.
        price_col (str): Column name of the price data.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Label' column:
                      -1: Price dropped below the lower threshold.
                       0: Price stayed within the thresholds.
                       1: Price exceeded the upper threshold.
    """
    df = df.copy()
    df["Label"] = 0

    upper_threshold = df[price_col] + threshold
    lower_threshold = df[price_col] - threshold

    for i in range(len(df) - horizon):
        future_prices = df.loc[i + 1 : i + horizon, price_col]
        if any(future_prices > upper_threshold[i]):
            df.loc[i, "Label"] = 1
        elif any(future_prices < lower_threshold[i]):
            df.loc[i, "Label"] = -1

    return df


def label_by_atr_threshold(df: pd.DataFrame, horizon: int, atr_period: int = 14, atr_mult: float = 2.0) -> pd.DataFrame:
    """
    Labels data based on ATR-based thresholds over a fixed horizon.

    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        horizon (int): Number of future bars to evaluate price movement.
        atr_period (int): Period for ATR calculation.
        atr_mult (float): Multiplier for ATR to set the threshold range.

    Returns:
        pd.DataFrame: DataFrame with an additional 'Label_ATR' column:
                      -1: Price dropped below the ATR-based lower threshold.
                       0: Price stayed within the ATR-based thresholds.
                       1: Price exceeded the ATR-based upper threshold.
    """
    df = df.copy()

    # Compute ATR
    df["ATR"] = ta.atr(high=df["High"], low=df["Low"], close=df["Close"], length=atr_period)

    # Define dynamic thresholds based on ATR
    upper_threshold = df["Close"] + atr_mult * df["ATR"]
    lower_threshold = df["Close"] - atr_mult * df["ATR"]

    df["Label_ATR"] = 0

    # Label based on future movement against ATR thresholds
    for i in range(len(df) - horizon):
        future_prices = df.loc[i + 1 : i + horizon, "Close"]
        if any(future_prices > upper_threshold[i]):
            df.loc[i, "Label_ATR"] = 1
        elif any(future_prices < lower_threshold[i]):
            df.loc[i, "Label_ATR"] = -1

    return df


def label_by_atr_long_short(
    df: pd.DataFrame, horizon: int, atr_period: int = 14, tp_mult: float = 1.5, sl_mult: float = 1.0
) -> pd.DataFrame:
    """
    Labels data for both long and short positions using ATR-based thresholds.

    Args:
        df (pd.DataFrame): DataFrame containing 'High', 'Low', and 'Close' columns.
        horizon (int): Number of future bars to evaluate price movement.
        atr_period (int): Period for ATR calculation.
        tp_mult (float): Take profit multiplier for ATR.
        sl_mult (float): Stop loss multiplier for ATR.

    Returns:
        pd.DataFrame: DataFrame with 'Label_Long' and 'Label_Short' columns:
                      Label_Long:
                        1: Price exceeded the take profit threshold without hitting stop loss.
                        0: No significant movement.
                      Label_Short:
                       -1: Price dropped below the stop loss threshold without hitting take profit.
                        0: No significant movement.
    """
    df = df.copy()
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=atr_period)

    # Define long thresholds
    lower_threshold_long = df["Close"] - df["ATR"] * sl_mult
    upper_threshold_long = df["Close"] + df["ATR"] * tp_mult

    # Define short thresholds
    lower_threshold_short = df["Close"] - df["ATR"] * tp_mult
    upper_threshold_short = df["Close"] + df["ATR"] * sl_mult

    df["Label_Long"] = 0
    df["Label_Short"] = 0

    for i in range(len(df) - horizon):
        future_prices = df["Close"].iloc[i + 1 : i + 1 + horizon]

        # SHORT condition: price drops below lower without hitting upper
        cond_short_1 = (future_prices < lower_threshold_short.iloc[i]).any()
        cond_short_2 = (future_prices < upper_threshold_short.iloc[i]).all()
        if cond_short_1 and cond_short_2:
            df.loc[i, "Label_Short"] = -1

        # LONG condition: price rises above upper without hitting lower
        cond_long_1 = (future_prices > upper_threshold_long.iloc[i]).any()
        cond_long_2 = (future_prices > lower_threshold_long.iloc[i]).all()
        if cond_long_1 and cond_long_2:
            df.loc[i, "Label_Long"] = 1

    return df


def label_by_trade_outcomes(df: pd.DataFrame, horizon: int, tp: float, sl: float) -> pd.Series:
    """
    Labels trading data based on simulated trade outcomes.

    Args:
        df (pd.DataFrame): DataFrame containing price data with a 'Close' column.
        horizon (int): Number of future bars to simulate the trade.
        tp (float): Take-profit threshold as a percentage (e.g., 0.02 for 2%).
        sl (float): Stop-loss threshold as a percentage (e.g., 0.01 for 1%).

    Returns:
        pd.Series: A Series of labels (-1, 0, or 1) indexed to the original DataFrame:
                   -1: Stop-loss hit within the horizon.
                    0: Neither take-profit nor stop-loss hit within the horizon.
                    1: Take-profit hit within the horizon.
    """
    close_prices = df["Close"].values
    labels = []

    for i in range(len(df)):
        entry_price = close_prices[i]
        tp_price = entry_price * (1 + tp)
        sl_price = entry_price * (1 - sl)
        label = 0

        for j in range(1, horizon + 1):
            if i + j >= len(df):
                break
            future_price = close_prices[i + j]
            if future_price >= tp_price:
                label = 1
                break
            elif future_price <= sl_price:
                label = -1
                break

        labels.append(label)

    return pd.Series(labels, index=df.index, name="Label")
