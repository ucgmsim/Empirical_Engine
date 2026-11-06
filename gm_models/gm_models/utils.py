"""Helpers shared across the ground motion models."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core


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


def evaluate_ims(
    compute: Callable[[_core.IM], _core.GMMResult],
    index: pd.Index,
    im: str,
    periods: Sequence[float] | None = None,
    attrs: dict[str, str | int | float] | None = None,
) -> xr.Dataset:
    """Evaluate a model for one IM type and collect the results in a Dataset.

    Parameters
    ----------
    compute : Callable[[_core.IM], _core.GMMResult]
        Evaluates the model for one `_core.IM` over all rupture-site pairs.
    index : pd.Index
        Rupture-site index, in the order `compute` returns values. Its name
        (``"index"`` if unnamed) becomes the site dimension.
    im : str
        Intensity measure: one of ``"PGA"``, ``"PGV"``, ``"pSA"``.
    periods : Sequence[float], optional
        Periods to compute, required if `im` is ``"pSA"``. Ignored otherwise.
    attrs : dict[str, str | int | float], optional
        Extra Dataset attributes, e.g. the model name and its settings.

    Returns
    -------
    xr.Dataset
        Variables ``ln_mean`` (natural-log space), ``std_total``, ``std_inter``
        and ``std_intra``, with dimensions ``(site, period)`` for ``pSA`` and
        ``(site,)`` for PGA/PGV.

    Raises
    ------
    ValueError
        If `im` is unsupported, or `im` is ``"pSA"`` without `periods`.
    """
    variables = ("ln_mean", "std_total", "std_inter", "std_intra")
    site_dim = index.name or "index"
    if im == "pSA":
        if not periods:
            raise ValueError("periods must be provided when im='pSA'")
        # Filled one period at a time, so only one period's results are held
        # alongside the output. Fortran order keeps each period's column contiguous.
        arrays = {
            var: np.empty((len(index), len(periods)), order="F") for var in variables
        }
        for j, period in enumerate(periods):
            result = compute(_core.IM.SA(period=float(period)))
            for var, arr in arrays.items():
                arr[:, j] = getattr(result, var)
        data = {var: ((site_dim, "period"), arr) for var, arr in arrays.items()}
        coords = {
            site_dim: index,
            "period": ("period", np.asarray(periods, dtype=np.float64), {"units": "s"}),
        }
    elif im in ("PGA", "PGV"):
        result = compute(_core.IM.PGA() if im == "PGA" else _core.IM.PGV())
        data = {var: ((site_dim,), getattr(result, var)) for var in variables}
        coords = {site_dim: index}
    else:
        raise ValueError(f"Unsupported im '{im}': expected one of 'PGA', 'PGV', 'pSA'")

    ds = xr.Dataset(data, coords=coords, attrs={"im": im, **(attrs or {})})
    ds["ln_mean"].attrs["units"] = "ln(cm/s)" if im == "PGV" else "ln(g)"
    return ds
