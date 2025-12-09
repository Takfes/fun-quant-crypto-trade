import pandas as pd
import talib
from talib.abstract import Function

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


def talib_transformer_func(
    df,
    indicators=None,
    group_names=None,
    exclude_groups=None,
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


TalibTransformer = transformer_wrapper(talib_transformer_func)
