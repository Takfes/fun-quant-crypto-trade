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
