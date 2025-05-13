import talib

from pytrade.features.wrappers import transformer_wrapper


def candle_pattern_transformer_func(df):
    dfc = df.copy()
    candle_names = talib.get_function_groups()["Pattern Recognition"]
    for candle in candle_names:
        dfc[f"{candle}_pattern"] = getattr(talib, candle)(
            dfc["open"],
            dfc["high"],
            dfc["low"],
            dfc["close"],
        )
    return dfc


CandlePatternTransformer = transformer_wrapper(candle_pattern_transformer_func)
