import numpy as np
import pandas as pd
from scipy.signal import argrelextrema

from pytrade.features.wrappers import transformer_wrapper


def local_maxima_transformer_func(df, column, order=20):
    """
    Calculate local maxima for the specified column in the DataFrame.

    Parameters:
    - df: DataFrame with the data.
    - column: The column name to calculate local maxima for.
    - order: Number of points to consider for local extrema detection.

    Returns:
    - df: DataFrame with added column for local maxima.
    """
    dfc = df.copy()

    # Identify local maxima in the specified column
    dfc[f"{column}_local_maxima"] = dfc[column].iloc[
        argrelextrema(dfc[column].values, np.greater_equal, order=order)[0]
    ]

    return dfc


LocalMaximaTransformer = transformer_wrapper(local_maxima_transformer_func)


def local_minima_transformer_func(df, column, order=20):
    """
    Calculate local maxima and minima for the 'close' column in the DataFrame.

    Parameters:
    - df: DataFrame with a 'close' column.
    - column: The column name to calculate local minima for.
    - order: Number of points to consider for local extrema detection.

    Returns:
    - df: DataFrame with added columns for local maxima and minima.
    """
    dfc = df.copy()

    # Identify local maxima and minima in price
    dfc[f"{column}_local_minima"] = dfc[column].iloc[argrelextrema(dfc[column].values, np.less_equal, order=order)[0]]

    return dfc


LocalMinimaTransformer = transformer_wrapper(local_minima_transformer_func)


def directional_change_transformer_func(
    df: pd.DataFrame, sigma: float = 0.01, close: str = "close", high: str = "high", low: str = "low"
) -> tuple[list[list[int | float]], list[list[int | float]]]:
    """
    Detects directional changes in a time series based on a sigma threshold.
    References:
    - https://www.youtube.com/watch?v=X31hyMhB-3s

    This implementation tracks significant turning points (tops and bottoms) by identifying
    when price movements exceed a specified percentage threshold (sigma).

    Args:
        close (pd.Series): Series of close prices.
        high (pd.Series): Series of high prices.
        low (pd.Series): Series of low prices.
        sigma (float): Threshold (as a fraction, e.g., 0.01 for 1%) to confirm a directional change.

    Returns:
        tuple[list[list[int | float]], list[list[int | float]]]:
            A tuple containing:
            - tops: list of [confirmation_index, peak_index, peak_price]
            - bottoms: list of [confirmation_index, bottom_index, bottom_price]
    """
    # Copy the DataFrame to avoid modifying the original
    dfc = df.copy()

    if close not in dfc.columns:
        raise ValueError(f"Column '{close}' not found in DataFrame.")

    if high not in dfc.columns:
        raise ValueError(f"Column '{high}' not found in DataFrame.")

    if low not in dfc.columns:
        raise ValueError(f"Column '{low}' not found in DataFrame.")

    close = dfc[close]
    high = dfc[high]
    low = dfc[low]

    up_zig = True  # Indicates the last confirmed extreme was a bottom, so we expect a top next
    tmp_max = high.iloc[0]
    tmp_min = low.iloc[0]
    tmp_max_i = high.index[0]
    tmp_min_i = low.index[0]

    tops: list[list[int | float]] = []
    bottoms: list[list[int | float]] = []

    for i in range(len(close)):
        if up_zig:
            # Currently expecting a top
            if high.iloc[i] > tmp_max:
                # New local high
                tmp_max = high.iloc[i]
                tmp_max_i = high.index[i]
            elif close.iloc[i] < tmp_max - tmp_max * sigma:
                # Price has retraced from high enough to confirm a top
                # top = [i, tmp_max_i, tmp_max]
                top = [tmp_max_i, tmp_max]
                tops.append(top)

                # Prepare for detecting a bottom
                up_zig = False
                tmp_min = low.iloc[i]
                tmp_min_i = low.index[i]
        else:
            # Currently expecting a bottom
            if low.iloc[i] < tmp_min:
                # New local low
                tmp_min = low.iloc[i]
                tmp_min_i = low.index[i]
            elif close.iloc[i] > tmp_min + tmp_min * sigma:
                # Price has bounced enough to confirm a bottom
                # bottom = [i, tmp_min_i, tmp_min]
                bottom = [tmp_min_i, tmp_min]
                bottoms.append(bottom)

                # Prepare for detecting a top
                up_zig = True
                tmp_max = high.iloc[i]
                tmp_max_i = high.index[i]

    # Update dataframe with the detected tops and bottoms
    dfc["directional_change_tops"] = pd.Series(
        np.array(tops)[:, 1], index=np.array(tops)[:, 0], name="directional_change_tops"
    )
    dfc["directional_change_bottoms"] = pd.Series(
        np.array(bottoms)[:, 1], index=np.array(bottoms)[:, 0], name="directional_change_bottoms"
    )

    return dfc


DirectionalChangeTransformer = transformer_wrapper(directional_change_transformer_func)
