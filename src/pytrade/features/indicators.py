import pandas as pd
import talib
from talib.abstract import Function


def validate_group_names(group_names, all_function_groups, exclude_groups=None):
    if isinstance(group_names, str):
        group_names = [group_names]
    if isinstance(group_names, list):
        for gn in group_names:
            if gn not in all_function_groups:
                raise ValueError(
                    f"Group '{gn}' not found in TA-Lib function groups. Check `talib.get_function_groups()` for all available options."
                )
    elif group_names is None:
        group_names = [x for x in all_function_groups if x not in (exclude_groups or [])]
    else:
        raise ValueError("group_names must be a string, a list of strings, or None.")
    return group_names


def process_function(func_name, dfc, group_name=None, params=None):
    try:
        func = Function(func_name)
        required_inputs = [
            item
            for sublist in func.info["input_names"].values()
            for item in (sublist if isinstance(sublist, list) else [sublist])
        ]
        inputs = {key: dfc[key] for key in required_inputs if key in dfc}
        if len(inputs) < len(required_inputs):
            print(f"Skipping function '{func_name}' due to missing inputs: {set(required_inputs) - set(inputs.keys())}")
            return dfc
        results = func(inputs, **(params or {}))
        output_names = func.info.get("output_names", [])
        group_slug = group_name.lower().replace(" ", "_") if group_name else "custom"
        func_slug = func_name.replace(" ", "_")
        new_columns = (
            {
                f"{group_slug}_{func_slug}_{output.lower().replace(' ', '_')}": result
                for output, result in zip(output_names, results)
            }
            if len(output_names) > 1
            else {f"{group_slug}_{func_slug}": results}
        )
        return pd.concat([dfc, pd.DataFrame(new_columns, index=dfc.index)], axis=1)
    except Exception as e:
        print(f"Failed to apply function '{func_name}': {e}")
        return dfc


def generate_talib_indicators(
    df, group_names=None, indicators=None, exclude_groups=["Math Operators", "Math Transform", "Price Transform"]
):
    """
    Generate TA-Lib indicators based on group names or explicit indicator specifications.

    Parameters:
        df (pd.DataFrame): Input DataFrame with required columns (e.g., open, high, low, close, volume).
        group_names (list or str, optional): List of TA-Lib function group names or a single group name.
            If None, all groups are used except those in exclude_groups.
        indicators (list of dict, optional): Explicit list of indicators to generate with their parameters.
            Each dict should have 'name' (indicator name) and 'params' (dict of parameters).
        exclude_groups (list, optional): List of group names to exclude when group_names is None.

    Returns:
        pd.DataFrame: DataFrame with the generated indicators as new columns.
    """
    dfc = df.copy()

    # If indicators list is provided, prioritize it and generate only those features
    if indicators:
        for indicator in indicators:
            dfc = process_function(indicator["name"], dfc, params=indicator.get("params", {}))
        return dfc

    # Otherwise, follow the group-based logic
    all_function_groups = list(talib.get_function_groups().keys())
    group_names = validate_group_names(group_names, all_function_groups, exclude_groups)

    for group_name in group_names:
        for func_name in talib.get_function_groups()[group_name]:
            dfc = process_function(func_name, dfc, group_name)

    return dfc
