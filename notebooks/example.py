import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
from ta import add_all_ta_features
from talib.abstract import Function

from pytrade.loaders import fetch_crypto_ohlcv_data

raw = fetch_crypto_ohlcv_data(symbol="BTC/USDT", timeframe="1d", lookback_period="3m")

# # OUT-OF-THE-BOX
# xxx = add_all_ta_features(raw, open="open", high="high", low="low", close="close", volume="volume")

# =========================== TESTING FUNCTIONALITY
# ==================================================

qq = generate_talib_indicators(dfc)
qq.columns

[x for x in dfc.columns if x not in qq.columns]

qq.equals(dfc)

# =========================== PRETEND FUNCTION START
# ==================================================

dfc = raw.copy()
dfc.columns
# group_names = "Momentum Indicators"
# group_names = "LOL"
# group_names = 12
# group_names = ["Cycle Indicators"]
# group_names = ["Momentum Indicators"]
# group_names = ["Overlap Studies"] # MAVP missing period
# group_names = ["Pattern Recognition"]
# group_names = ["Statistic Functions"]
# group_names = ["Volatility Indicators"]
# group_names = ["Volume Indicators"]  # issue with OBV - price and prices
# group_names = ["Math Operators"]  # exclude this
# group_names = ["Math Transform"]  # exclude this
# group_names = ["Price Transform"] # exclude this
group_names = None

inidcators = [
    {"name": "APO", "params": {"timeperiod": 30}},
    {"name": "AROON", "params": {"fastperiod": 12, "slowperiod": 26, "matype": 0}},
    {"name": "BOP"},
]

exclude_groups = ["Math Operators", "Math Transform", "Price Transform"]


# Get the list of all function groups
all_function_groups = list(talib.get_function_groups().keys())

# Handle group_names input
# If string, convert to list
if isinstance(group_names, str):
    group_names = [group_names]

# If list, check if all groups are valid
if isinstance(group_names, list):
    for gn in group_names:
        if gn not in all_function_groups:
            raise ValueError(
                f"Group '{gn}' not found in TA-Lib function groups. Check `talib.get_function_groups()` for all available options."
            )
# If None, use all groups
elif group_names is None:
    group_names = [x for x in all_function_groups if x not in exclude_groups]
else:
    raise ValueError("group_names must be a string or a list of strings.")

# START LOOP
# Iterate over the group names
for group_name in group_names:
    # Get all the functions for the group name
    functions = talib.get_function_groups()[group_name]

    # Iterate over the function names
    for func_name in functions:
        # Grab the function based on its name
        func = Function(func_name)

        # Determine the required input parameters
        temp_ri = list(func.info["input_names"].values())
        required_inputs_ = temp_ri[0] if isinstance(temp_ri[0], list) else temp_ri
        # Flatten the required_inputs list - issue with OBV : ['close', ['volume']]
        required_inputs = [
            item for sublist in required_inputs_ for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

        # Determine the expected output names
        output_names = func.info.get("output_names", [])

        # Extract the required inputs from the DataFrame
        inputs = {key: dfc[key] for key in required_inputs if key in dfc}

        # Check if all required inputs are present in the DataFrame
        if len(inputs) < len(required_inputs):
            missing_inputs = set(required_inputs) - set(inputs.keys())
            print(f"Skipping function '{func_name}' due to missing inputs: {missing_inputs}")
            continue

        # Execute the function
        results = func(inputs)

        # Fomrat the function group name
        group_name_slug = group_name.lower().replace(" ", "_")
        # Format the function name
        func_name_slug = func_name.replace(" ", "_")
        # Format the output names
        output_names_slug = [x.lower().replace(" ", "_") for x in output_names]

        # Collect the results in a dictionary for batch insertion
        new_columns = {}

        if len(output_names) == 1:
            new_columns[f"{group_name_slug}_{func_name_slug}"] = results
        else:
            for output_name_slug, result in zip(output_names_slug, results):
                new_columns[f"{group_name_slug}_{func_name_slug}_{output_name_slug}"] = result

        # Add the new columns to the DataFrame in a single operation
        dfc = pd.concat([dfc, pd.DataFrame(new_columns, index=dfc.index)], axis=1)
