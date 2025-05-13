import numpy as np
import pandas as pd
import talib

from pytrade.features.structure import local_maxima_transformer_func, local_minima_transformer_func
from pytrade.features.wrappers import transformer_wrapper

# https://www.investopedia.com/terms/d/divergence.asp#:~:text=RSI%3A%20Divergence%20appears%20when%20the,levels%2C%20it%20signals%20bearish%20divergence.


def rsi_divergence_transformer_func(df, rsi_period=20, extrema_order=20):
    """
    Transforms a given DataFrame by calculating RSI-based divergence features.
    This function computes the Relative Strength Index (RSI) for a given price
    series and identifies divergences between the price and the RSI. It uses
    local extrema (maxima and minima) to determine trends and calculates
    divergence features for both maxima and minima.
    Args:
        df (pd.DataFrame): Input DataFrame containing at least a "close" column
            for price data, as well as "high" and "low" columns for extrema
            calculations.
        rsi_period (int, optional): The time period for calculating the RSI.
            Defaults to 20.
        extrema_order (int, optional): The order parameter for identifying local
            extrema (maxima and minima). Higher values result in fewer extrema
            points. Defaults to 20.
    Returns:
        pd.DataFrame: A transformed DataFrame with additional columns for RSI,
        local extrema, trends, divergences, and divergence signs:
            - "rsi": The calculated RSI values.
            - "{column}_local_maxima" and "{column}_local_minima": Local extrema
              for the specified column.
            - "{column}_trend": Trend of the local extrema for the specified
              column.
            - "divergence_maxima_high_rsi" and "divergence_minima_low_rsi":
              Divergence values for maxima and minima between price and RSI.
            - "divergence_maxima_high_rsi_sign" and
              "divergence_minima_low_rsi_sign": Binary indicators for divergence
              signs (1 for divergence, 0 otherwise).
    Notes:
        - This function requires the `talib` library for RSI calculation and
          assumes the presence of helper functions `local_maxima_func` and
          `local_minima_func` for identifying local extrema.
        - The input DataFrame should have columns "close", "high", and "low"
          for proper functionality.
    Example:
        >>> transformed_df = rsi_divergence_transformer_func(df, rsi_period=14, extrema_order=10)
    """

    # copy dataframe
    dfc = df.copy()

    # secondary series name
    ssname = "rsi"

    # calculate higher high and lower low
    def local_extrema_trend(df, column):
        """
        Calculate the trend of local extrema for a specified column in a DataFrame.
        This function computes the percentage change of the specified column,
        fills missing values forward and backward, and replaces zeros with NaN
        to avoid flat trends. It then forward-fills NaN values and replaces any
        remaining NaN values with 0. The result is stored in a new column
        named '{column}_trend'.
        Parameters:
            df (pd.DataFrame): The input DataFrame containing the data.
            column (str): The name of the column for which to calculate the trend.
        Returns:
            pd.DataFrame: A copy of the input DataFrame with an additional column
                          '{column}_trend' containing the calculated trend values.
        """

        dfc = df.copy()
        dfc[f"{column}_trend"] = dfc[column].ffill().bfill().pct_change().fillna(0).replace(0, np.nan).ffill().fillna(0)
        return dfc

    # calculate seconday series
    dfc[ssname] = talib.RSI(dfc["close"], timeperiod=rsi_period)

    # create local extrema for price
    dfc = local_maxima_transformer_func(dfc, column="high", order=extrema_order)
    dfc = local_minima_transformer_func(dfc, column="low", order=extrema_order)

    # create local extrema for secondary series
    dfc = local_maxima_transformer_func(dfc, column=ssname, order=extrema_order)
    dfc = local_minima_transformer_func(dfc, column=ssname, order=extrema_order)

    # calculate local extrema trend for price
    dfc = local_extrema_trend(dfc, column="high_local_maxima")
    dfc = local_extrema_trend(dfc, column="low_local_minima")

    # calculate local extrema trend for secondary series
    dfc = local_extrema_trend(dfc, column=f"{ssname}_local_maxima")
    dfc = local_extrema_trend(dfc, column=f"{ssname}_local_minima")

    # calculate divergence for local maxima between price and secondary series
    dfc[f"divergence_maxima_high_{ssname}"] = (
        dfc["high_local_maxima_trend"].mul(dfc[f"{ssname}_local_maxima_trend"]).clip(upper=0)
    )

    # calculate divergence for local minima between price and secondary series
    dfc[f"divergence_minima_low_{ssname}"] = (
        dfc["low_local_minima_trend"].mul(dfc[f"{ssname}_local_minima_trend"]).clip(upper=0)
    )

    # calculate divergence's sign for local maxima between price and secondary series
    dfc[f"divergence_maxima_high_{ssname}_sign"] = np.where(dfc[f"divergence_maxima_high_{ssname}"] < 0, 1, 0)

    # calculate divergence's sign for local minima between price and secondary series
    dfc[f"divergence_minima_low_{ssname}_sign"] = np.where(dfc[f"divergence_minima_low_{ssname}"] < 0, 1, 0)

    return dfc


RSIDivergenceTransformer = transformer_wrapper(rsi_divergence_transformer_func)
