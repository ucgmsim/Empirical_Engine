"""Helpers shared across the ground motion models."""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _core


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


# Scalar IMs, and the ln_mean units of every IM.
_SCALAR_IMS = {
    "PGA": _core.IM.PGA,
    "PGV": _core.IM.PGV,
    "CAV": _core.IM.CAV,
    "Ds575": _core.IM.Ds575,
    "Ds595": _core.IM.Ds595,
    "Ds2080": _core.IM.Ds2080,
}
_UNITS = {
    "PGA": "ln(g)",
    "pSA": "ln(g)",
    "PGV": "ln(cm/s)",
    "CAV": "ln(g-s)",
    "EAS": "ln(g-s)",
    "Ds575": "ln(s)",
    "Ds595": "ln(s)",
    "Ds2080": "ln(s)",
}


def evaluate_ims(
    compute: Callable[[_core.IM], _core.GMMResult],
    index: pd.Index,
    im: str,
    periods: Sequence[float] | None = None,
    attrs: dict[str, str | int | float] | None = None,
    frequencies: Sequence[float] | None = None,
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
        Intensity measure: ``"pSA"``, ``"EAS"``, or one of ``"PGA"``,
        ``"PGV"``, ``"CAV"``, ``"Ds575"``, ``"Ds595"``, ``"Ds2080"``.
    periods : Sequence[float], optional
        Periods to compute, required if `im` is ``"pSA"``. Ignored otherwise.
    attrs : dict[str, str | int | float], optional
        Extra Dataset attributes, e.g. the model name and its settings.
    frequencies : Sequence[float], optional
        Frequencies (Hz) to compute, required if `im` is ``"EAS"``. Ignored
        otherwise.

    Returns
    -------
    xr.Dataset
        Variables ``ln_mean`` (natural-log space), ``std_total``, ``std_inter``
        and ``std_intra``, with dimensions ``(site, period)`` for ``pSA``,
        ``(site, frequency)`` for ``EAS``, and ``(site,)`` otherwise.

    Raises
    ------
    ValueError
        If `im` is unsupported, or `im` is ``"pSA"`` without `periods` or
        ``"EAS"`` without `frequencies`.
    """
    variables = ("ln_mean", "std_total", "std_inter", "std_intra")
    site_dim = index.name or "index"
    if im in ("pSA", "EAS"):
        dim, values, unit, make_im = (
            ("period", periods, "s", lambda v: _core.IM.SA(period=v))
            if im == "pSA"
            else ("frequency", frequencies, "Hz", lambda v: _core.IM.EAS(frequency=v))
        )
        if values is None or len(values) == 0:
            raise ValueError(f"{dim} values must be provided when im='{im}'")
        # Filled one column at a time, so only one column's results are held
        # alongside the output. Fortran order keeps each column contiguous.
        arrays = {
            var: np.empty((len(index), len(values)), order="F") for var in variables
        }
        for j, value in enumerate(values):
            result = compute(make_im(float(value)))
            for var, arr in arrays.items():
                arr[:, j] = getattr(result, var)
        data = {var: ((site_dim, dim), arr) for var, arr in arrays.items()}
        coords = {
            site_dim: index,
            dim: (dim, np.asarray(values, dtype=np.float64), {"units": unit}),
        }
    elif im in _SCALAR_IMS:
        result = compute(_SCALAR_IMS[im]())
        data = {var: ((site_dim,), getattr(result, var)) for var in variables}
        coords = {site_dim: index}
    else:
        raise ValueError(
            f"Unsupported im '{im}': expected 'pSA', 'EAS' or one of {list(_SCALAR_IMS)}"
        )

    ds = xr.Dataset(data, coords=coords, attrs={"im": im, **(attrs or {})})
    ds["ln_mean"].attrs["units"] = _UNITS[im]
    return ds
