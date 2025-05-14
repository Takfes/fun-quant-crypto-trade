import numpy as np
import pandas as pd
import talib
from sklearn.base import BaseEstimator, TransformerMixin


def target_transformer_wrapper(transform_function):
    """
    A wrapper to convert a target transformation function into a scikit-learn compatible Transformer.

    Args:
        transform_function (callable): A function that transforms the target variable.

    Returns:
        A scikit-learn compatible transformer for the target variable.
    """

    class WrappedTargetTransformer(BaseEstimator, TransformerMixin):
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, y, X=None):
            # No fitting necessary for this transformer
            return self

        def transform(self, y):
            # Ensure input is a pandas Series
            if not isinstance(y, pd.Series):
                raise ValueError("Input must be a pandas Series")

            y_ = y.copy()
            return transform_function(y_, **self.kwargs)

    return WrappedTargetTransformer


def fixed_horizon_labeler_func(
    df: pd.DataFrame,
    horizon: int,
    up_threshold: float,
    down_threshold: float,
    close_col: str = "close",
) -> pd.Series:
    """
    Labels time series data based on fixed-time horizon returns with asymmetrical thresholds.

    Args:
        df (pd.DataFrame): DataFrame containing price data.
        horizon (int): Number of future bars to calculate returns.
        up_threshold (float): Threshold for classifying positive returns.
        down_threshold (float): Threshold for classifying negative returns.
        close_col (str): Column name of the price data.

    Returns:
        pd.Series: A Series of labels (-1, 0, or 1) indexed to the original DataFrame.
                   -1: Negative return below the down_threshold.
                    0: Return within the threshold range.
                    1: Positive return above the up_threshold.
    """
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in DataFrame.")

    prices = df[close_col]
    future_returns = (prices.shift(-horizon) - prices) / prices  # % return

    labels = np.where(
        future_returns > up_threshold,
        1,
        np.where(future_returns < -down_threshold, -1, 0),
    )

    return pd.Series(labels, index=df.index, name=f"target_h{horizon}up{up_threshold}down{down_threshold}")


FixedHorizonTargetTransformer = target_transformer_wrapper(fixed_horizon_labeler_func)


def trade_outcome_labeler_func(
    df: pd.DataFrame, horizon: int, tp: float, sl: float, close_col: str = "close"
) -> pd.Series:
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
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in DataFrame.")

    close_prices = df[close_col].values
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

    return pd.Series(labels, index=df.index, name=f"target_h{horizon}tp{tp}sl{sl}")


TradeOutcomeTargetTransformer = target_transformer_wrapper(trade_outcome_labeler_func)


def atr_threshold_labeler_func(
    df: pd.DataFrame,
    horizon: int,
    atr_period: int = 14,
    atr_mult_long: float = 2.0,
    atr_mult_short: float = 2.0,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
) -> pd.DataFrame:
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
    if close_col not in df.columns:
        raise ValueError(f"Column '{close_col}' not found in DataFrame.")

    if high_col not in df.columns:
        raise ValueError(f"Column '{high_col}' not found in DataFrame.")

    if low_col not in df.columns:
        raise ValueError(f"Column '{low_col}' not found in DataFrame.")

    df = df.copy()

    # Compute ATR
    df["ATR"] = talib.ATR(df[high_col], df[low_col], df[close_col], timeperiod=atr_period)

    # Define dynamic thresholds based on ATR
    upper_threshold = df[close_col] + atr_mult_long * df["ATR"]
    lower_threshold = df[close_col] - atr_mult_short * df["ATR"]

    df["Label_ATR"] = 0

    # Label based on future movement against ATR thresholds
    for i in range(len(df) - horizon):
        future_prices = df.loc[i + 1 : i + horizon, close_col]
        if any(future_prices > upper_threshold[i]):
            df.loc[i, "Label_ATR"] = 1
        elif any(future_prices < lower_threshold[i]):
            df.loc[i, "Label_ATR"] = -1

    return pd.Series(
        df["Label_ATR"].values,
        index=df.index,
        name=f"target_h{horizon}atrp{atr_period}atrl{atr_mult_long}atrs{atr_mult_short}",
    )


ATRThresholdTargetTransformer = target_transformer_wrapper(atr_threshold_labeler_func)
