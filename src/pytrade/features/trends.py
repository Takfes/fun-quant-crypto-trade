import pandas as pd
import talib

from pytrade.features.wrappers import transformer_wrapper


def triple_sma_transformer_func(df, column="close", fast_period=10, medium_period=30, slow_period=100):
    """
    Applies a triple simple moving average (SMA) transformation to a DataFrame and generates
    additional features based on the relationship between the specified column and the SMAs.

    Parameters:
        df (pd.DataFrame): The input DataFrame containing the data.
        column (str): The name of the column to calculate SMAs for. Default is "close".
        fast_period (int): The time period for the fast SMA. Default is 10.
        medium_period (int): The time period for the medium SMA. Default is 30.
        slow_period (int): The time period for the slow SMA. Default is 100.

    Returns:
        pd.DataFrame: A copy of the input DataFrame with the following additional columns:
            - "sma_fast": The fast SMA of the specified column.
            - "sma_medium": The medium SMA of the specified column.
            - "sma_slow": The slow SMA of the specified column.
            - "{column}_vs_sma_fast": The difference between the specified column and the fast SMA.
            - "{column}_vs_sma_medium": The difference between the specified column and the medium SMA.
            - "{column}_vs_sma_slow": The difference between the specified column and the slow SMA.
            - "is_{column}_above_all_sma": A binary column indicating if the specified column is above all SMAs.
            - "is_{column}_below_all_sma": A binary column indicating if the specified column is below all SMAs.
    """

    dfc = df.copy()
    dfc["sma_fast"] = talib.SMA(dfc[column], timeperiod=fast_period)
    dfc["sma_medium"] = talib.SMA(dfc[column], timeperiod=medium_period)
    dfc["sma_slow"] = talib.SMA(dfc[column], timeperiod=slow_period)

    dfc[f"{column}_vs_sma_fast"] = dfc[column] - dfc["sma_fast"]
    dfc[f"{column}_vs_sma_medium"] = dfc[column] - dfc["sma_medium"]
    dfc[f"{column}_vs_sma_slow"] = dfc[column] - dfc["sma_slow"]

    dfc["is_{column}_above_all_sma"] = (
        (dfc[column] > dfc["sma_fast"])
        & (dfc[column] > dfc["sma_medium"])
        & (dfc[column] > dfc["sma_slow"])
        & dfc["sma_fast"].notna()
        & dfc["sma_medium"].notna()
        & dfc["sma_slow"].notna()
    ).astype(int)

    dfc["is_{column}_below_all_sma"] = (
        (dfc[column] < dfc["sma_fast"])
        & (dfc[column] < dfc["sma_medium"])
        & (dfc[column] < dfc["sma_slow"])
        & dfc["sma_fast"].notna()
        & dfc["sma_medium"].notna()
        & dfc["sma_slow"].notna()
    ).astype(int)

    return dfc


TripleSMATransformer = transformer_wrapper(triple_sma_transformer_func)
