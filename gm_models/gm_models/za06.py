"""Python scaffolding around the Rust ZA_06 (Zhao et al. 2006) implementation.

Column/input validation lives here; the GMPE math lives in the compiled
`gm_models._core` Rust extension (see `src-rust/za06.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _core, constants, utils

_REQUIRED_COLUMNS = {
    constants.TectType.ACTIVE_SHALLOW: {"mag", "rrup", "vs30", "hypo_depth", "rake"},
    constants.TectType.SUBDUCTION_INTERFACE: {"mag", "rrup", "vs30", "hypo_depth"},
    constants.TectType.SUBDUCTION_SLAB: {"mag", "rrup", "vs30", "hypo_depth"},
}


def run_za06(
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    extrapolate: bool = False,
) -> xr.Dataset:
    """Run the ZA_06 (Zhao et al. 2006) model.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (active shallow, subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30`` and
        ``hypo_depth`` (plus ``rake`` for `TectType.ACTIVE_SHALLOW`).
    im : str
        Intensity measure: one of ``"PGA"``, ``"pSA"``.
    periods : Sequence[float], optional
        Periods to compute, required if `im` is ``"pSA"``. Ignored otherwise.
    extrapolate : bool
        Whether to extrapolate pSA above the 5 s limit of the coefficient table
        as `oq_wrapper` does (constant spectral displacement: ln mean
        + 2 ln(5 / T)). Off by default, in which case such periods raise.

    Returns
    -------
    xr.Dataset
        Variables ``ln_mean`` (natural-log space), ``std_total``, ``std_inter``
        and ``std_intra`` over `rupture_df`'s index (plus ``period`` for
        ``pSA``), with the model settings as attributes. See
        `gm_models.utils.evaluate_ims`.

    Raises
    ------
    ValueError
        If `tect_type` is unsupported, a required column is missing, or a pSA
        period is above 5 s and `extrapolate` is False.
    """
    if tect_type not in _REQUIRED_COLUMNS:
        raise ValueError(f"ZA_06 does not support tectonic type {tect_type}")
    utils.validate_columns(rupture_df, _REQUIRED_COLUMNS[tect_type])

    mag, rrup, vs30, hypo_depth = (
        rupture_df[col].to_numpy(dtype=np.float64)
        for col in ("mag", "rrup", "vs30", "hypo_depth")
    )
    if tect_type is constants.TectType.ACTIVE_SHALLOW:
        inputs = _core.Za06Inputs.asc(
            mag, rrup, vs30, hypo_depth, rupture_df["rake"].to_numpy(dtype=np.float64)
        )
    elif tect_type is constants.TectType.SUBDUCTION_INTERFACE:
        inputs = _core.Za06Inputs.interface(mag, rrup, vs30, hypo_depth)
    else:
        inputs = _core.Za06Inputs.slab(mag, rrup, vs30, hypo_depth)
    return utils.evaluate_ims(
        lambda core_im: _core.za06(inputs, core_im, extrapolate),
        rupture_df.index,
        im,
        periods,
        # extrapolate as int: netCDF attributes can't be bool.
        attrs={
            "model": "ZA_06",
            "tect_type": str(tect_type),
            "extrapolate": int(extrapolate),
        },
    )
