from calendar import c

import numpy as np
import pandas as pd
from regex import P
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


def pips_transformer_func(
    df: pd.DataFrame, n_pips: int, dist_measure: int = 1, close: str = "close"
) -> tuple[list[int], list[float]]:
    """
    Extracts PIPs (Perceptually Important Points) from a time series using one of three distance measures.

    This function iteratively finds the most perceptually important points in the data,
    starting from endpoints, based on how far a point deviates from its neighbors.

    Args:
        data (np.ndarray): 1D array of time series data.
        n_pips (int): Number of PIPs to extract.
        dist_measure (int): Distance measure to use.
            1 = Euclidean distance
            2 = Perpendicular distance
            3 = Vertical distance

    Returns:
        tuple[list[int], list[float]]:
            - pips_x: Indices of selected PIPs.
            - pips_y: Corresponding values from `data` at those indices.
    """
    # Copy the DataFrame to avoid modifying the original
    dfc = df.copy()

    # Validate input parameters
    if close not in dfc.columns:
        raise ValueError(f"Column '{close}' not found in DataFrame.")

    # Extract the data from the specified column
    data = dfc[close].values

    # Instantiate the output lists
    pips_x = [0, len(data) - 1]  # Indices of first and last points
    pips_y = [data[0], data[-1]]  # Corresponding values

    for curr_point in range(2, n_pips):
        max_dist = 0.0
        max_dist_idx = -1
        insert_index = -1

        for k in range(0, curr_point - 1):
            left_adj = k
            right_adj = k + 1

            time_diff = pips_x[right_adj] - pips_x[left_adj]
            price_diff = pips_y[right_adj] - pips_y[left_adj]
            slope = price_diff / time_diff
            intercept = pips_y[left_adj] - pips_x[left_adj] * slope

            for i in range(pips_x[left_adj] + 1, pips_x[right_adj]):
                # Compute distance based on selected method
                d = 0.0
                if dist_measure == 1:
                    # Euclidean distance
                    d += ((pips_x[left_adj] - i) ** 2 + (pips_y[left_adj] - data[i]) ** 2) ** 0.5
                    d += ((pips_x[right_adj] - i) ** 2 + (pips_y[right_adj] - data[i]) ** 2) ** 0.5
                elif dist_measure == 2:
                    # Perpendicular distance
                    d = abs(slope * i + intercept - data[i]) / (slope**2 + 1) ** 0.5
                else:
                    # Vertical distance
                    d = abs(slope * i + intercept - data[i])

                # Track maximum distance point
                if d > max_dist:
                    max_dist = d
                    max_dist_idx = i
                    insert_index = right_adj

        # Insert the new PIP into the lists at the correct position
        pips_x.insert(insert_index, max_dist_idx)
        pips_y.insert(insert_index, data[max_dist_idx])

    # Convert output to dataframe format
    dfc["preceptually_important_points"] = pd.Series(
        pips_y, index=[dfc.index[i] for i in pips_x], name="preceptually_important_points"
    )

    return dfc


PipsTransformer = transformer_wrapper(pips_transformer_func)
