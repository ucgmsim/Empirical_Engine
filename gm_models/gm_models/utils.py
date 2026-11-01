"""Helpers shared across the ground motion models."""

import pandas as pd


def validate_columns(rupture_df: pd.DataFrame, required: set[str]) -> None:
    """Raise ``ValueError`` if `rupture_df` is missing required columns.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        Input rupture/site dataframe to validate.
    required : set[str]
        Column names the model needs.
    """
    missing = required - set(rupture_df.columns)
    if missing:
        raise ValueError(f"rupture_df is missing required columns: {sorted(missing)}")
