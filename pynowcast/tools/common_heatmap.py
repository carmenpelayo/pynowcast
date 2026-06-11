import numpy as np
import pandas as pd
from types import SimpleNamespace


def common_heatmap(xest: pd.DataFrame,
                   Par,
                   groups: np.ndarray,
                   groups_name: list,
                   fullnames: list) -> SimpleNamespace:
    """
    Compute standardized z-scores for each input series and aggregated group-level z-scores.

    Args:
        xest: DataFrame of input series (index = datetime, columns = series values).
        Par: namespace of model parameters (not modified here).
        groups: array of group IDs for each series (length = number of columns in xest).
        groups_name: list of group names corresponding to each unique group ID.
        fullnames: list of full descriptive names for each series (length = n_series).

    Returns:
        SimpleNamespace with attributes:
            names: list of series names (fullnames).
            zscores: ndarray of shape (T, N) of standardized series values.
            names_agg: list of group names.
            zscores_agg: ndarray of shape (T, G) of aggregated group z-scores.
    """
    # Ensure DataFrame columns match groups and fullnames length
    if xest.shape[1] != len(groups) or xest.shape[1] != len(fullnames):
        raise ValueError("Length of 'groups' and 'fullnames' must match number of columns in xest")

    # Compute z-scores for each series
    zscores_df = (xest - xest.mean()) / xest.std(ddof=0)

    # Prepare output names
    names = list(fullnames)

    # Aggregate z-scores by group
    unique_groups = np.unique(groups)
    G = len(unique_groups)
    # Map unique_groups to group_name in the same order as groups_name
    # Assume groups_name is ordered by unique_groups
    names_agg = list(groups_name)
    # Initialize DataFrame for aggregated z-scores
    zscores_agg_df = pd.DataFrame(index=xest.index)
    for idx, g in enumerate(unique_groups):
        cols = [i for i, grp in enumerate(groups) if grp == g]
        if cols:
            zscores_agg_df[names_agg[idx]] = zscores_df.iloc[:, cols].mean(axis=1)
        else:
            zscores_agg_df[names_agg[idx]] = np.nan

    # Build SimpleNamespace
    heatmap = SimpleNamespace(
        names=names,
        zscores=zscores_df.values,
        names_agg=names_agg,
        zscores_agg=zscores_agg_df.values
    )
    return heatmap
