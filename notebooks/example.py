import numpy as np
import pandas as pd
import plotly.graph_objects as go
import talib
from attr import validate
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


# =========================== DEFINE FUNCTION
# ==================================================

indicators = [
    {"name": "APO", "params": {"timeperiod": 30}},
    {"name": "AROON", "input_names": ["high", "low"], "params": {"fastperiod": 12, "slowperiod": 26, "matype": 0}},
    {"name": "BOP"},
    {"name": "SMA", "input_names": ["price", "close"], "params": {"timeperiod": 30}},
]

"""
{'name': 'SMA',
 'group': 'Overlap Studies',
 'display_name': 'Simple Moving Average',
 'function_flags': ['Output scale same as input'],
 'input_names': OrderedDict([('price', 'close')]),
 'parameters': OrderedDict([('timeperiod', 30)]),
 'output_flags': OrderedDict([('real', ['Line'])]),
 'output_names': ['real']}

{'name': 'AROON',
 'group': 'Momentum Indicators',
 'display_name': 'Aroon',
 'function_flags': None,
 'input_names': OrderedDict([('prices', ['high', 'low'])]),
 'parameters': OrderedDict([('timeperiod', 14)]),
 'output_flags': OrderedDict([('aroondown', ['Dashed Line']),
              ('aroonup', ['Line'])]),
 'output_names': ['aroondown', 'aroonup']}
"""

Function("AROON").info

ls = [f"{k[:3]}{v}".replace(".", "&") for k, v in Function("BBANDS").info["parameters"].items()]

"_".join(ls)


func_id_1 = {
    "name": "AROON",
    "input_names": ["high", "low"],
    "params": {"timeperiod": 14},
}

func_id_2 = {"name": "AROON", "params": {"fastperiod": 12, "slowperiod": 26, "matype": 0}}

func_id = {"name": "SMA", "input_names": ["close"], "params": {"timeperiod": 30}}

g1 = apply_talib_function(dfc, func_id_1)
g2 = apply_talib_function(dfc, func_id_2)
g3 = apply_talib_function(dfc, "AROON")

g1.equals(g2)
g1.equals(g3)

apply_talib_function(dfc, "MAVP")


def apply_talib_function(df, func_id):
    def _get_default_input_names(func):
        return [
            item
            for sublist in func.info["input_names"].values()
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]

    # Create a copy of the DataFrame to avoid modifying the original
    dfc = df.copy()

    # Determine the function name and parameters based on func_id
    if isinstance(func_id, str):
        func_name = func_id
        params = {}
    elif isinstance(func_id, dict):
        func_name = func_id.get("name")
        if not func_name:
            raise ValueError("Dictionary func_id must contain a 'name' key with the function name.")
        params = func_id.get("params", {})
    else:
        raise ValueError("func_id must be either a string or a dictionary.")

    # Grab the function based on its name
    func = Function(func_name)

    # Validate user-supplied params are a subset of the default parameters
    default_params = set(func.info["parameters"].keys())
    if params:
        user_params = set(params.keys())
        if not user_params.issubset(default_params):
            raise ValueError(
                f"Function '{func_name}' received invalid parameter(s): {user_params - default_params}. "
                f"Valid parameters are: {default_params}"
            )

    try:
        # Determine the required input parameters
        if isinstance(func_id, dict) and "input_names" in func_id:
            input_names = func_id["input_names"]
            if not isinstance(input_names, list):
                raise ValueError("input_names must be a list of strings.")
            if not input_names:
                raise ValueError("input_names list cannot be empty.")
        else:
            input_names = _get_default_input_names(func)

        # Check if all required inputs are present in the DataFrame
        if not all(x in dfc.columns for x in input_names):
            raise ValueError(
                f"Function '{func_name}' requires inputs {input_names}, but not all are present in the DataFrame."
            )
        # Extract the required inputs from the DataFrame
        inputs = {key: dfc[key] for key in input_names if key in dfc}

        # Execute the function
        try:
            results = func(inputs, **params)
        except Exception as e:
            raise RuntimeError(f"Failed to apply function '{func_name}': {e}") from e

        # Determine the expected output names
        output_names = func.info.get("output_names", [])

        # Prepare the group slug
        group_slug = func.info.get("group").lower().replace(" ", "_")
        # Prepare the function slug
        func_slug = func_name.replace(" ", "_")
        # Prepare the output slug
        output_slug = [x.lower().replace(" ", "_") for x in output_names]

        # Prepare the params slug
        if params:
            param_slug = "_".join([f"{k[:3]}{v}".replace(".", "&") for k, v in params.items()])
        else:
            param_slug = "_".join([f"{k[:3]}{v}".replace(".", "&") for k, v in func.info["parameters"].items()])

        # Collect the results in a dictionary for batch insertion
        new_columns = (
            {
                f"{group_slug}_{func_slug}_{output}{'_' + param_slug if param_slug else ''}": result
                for output, result in zip(output_slug, results)
            }
            if len(output_names) > 1
            else {f"{group_slug}_{func_slug}{'_' + param_slug if param_slug else ''}": results}
        )

        return pd.concat([dfc, pd.DataFrame(new_columns, index=dfc.index)], axis=1)
    except Exception as e:
        print(f"Failed to apply function '{func_name}': {e}")
        return dfc


# =========================== TESTING FUNCTIONALITY
# ==================================================


generate_talib_indicators(dfc, group_names="Momentum Indicators")
generate_talib_indicators(dfc, group_names=["Pattern Recognition", "Volume Indicators"])
generate_talib_indicators(dfc)

indicators = [
    {"name": "BOP"},
    {"name": "BBANDS"},
    {"name": "SAREXT"},
    {"name": "T3"},
    {"name": "AROON", "input_names": ["high", "low"], "params": {"timeperiod": 14}},
    {"name": "SMA", "input_names": ["close"], "params": {"timeperiod": 30}},
    {"name": "SMA", "input_names": ["close"], "params": {"timeperiod": 60}},
]

generate_talib_indicators(dfc, indicators=indicators)


def generate_talib_indicators(
    df,
    indicators=None,
    group_names=None,
    exclude_groups=exclude_groups,
):
    """
    Wrapper function to generate TA-Lib indicators.

    Parameters
    ----------
    df : pd.DataFrame
        Input OHLCV DataFrame.
    indicators : list of dict, optional
        List of indicator specifications (dicts with 'name', optional 'input_names', 'params').
    group_names : list of str or str or None, optional
        TA-Lib function group(s) to use. If None, uses all groups except excluded ones.
    exclude_groups : list of str, optional
        Groups to exclude if group_names is None.

    Returns
    -------
    pd.DataFrame
        DataFrame with added indicator columns.
    """

    dfc = df.copy()
    if exclude_groups is None:
        exclude_groups = ["Math Operators", "Math Transform", "Price Transform"]

    if indicators is not None:
        # Use explicit indicator list
        for ind in indicators:
            dfc = apply_talib_function(dfc, ind)
        return dfc

    # Get all function groups
    all_function_groups = list(talib.get_function_groups().keys())

    # Handle group_names input
    if isinstance(group_names, str):
        group_names = [group_names]

    if isinstance(group_names, list):
        for gn in group_names:
            if gn not in all_function_groups:
                raise ValueError(
                    f"Group '{gn}' not found in TA-Lib function groups. Check `talib.get_function_groups()` for all available options."
                )
    elif group_names is None:
        group_names = [x for x in all_function_groups if x not in exclude_groups]
    else:
        raise ValueError("group_names must be a string or a list of strings.")

    # Iterate over the group names and functions
    for group_name in group_names:
        functions = talib.get_function_groups()[group_name]
        for func_name in functions:
            dfc = apply_talib_function(dfc, func_name)
    return dfc
