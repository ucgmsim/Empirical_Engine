"""Python scaffolding around the Rust CB_10 (Campbell and Bozorgnia, 2008/2010) implementation.

Column validation lives here; the actual GMPE math lives in the compiled
`gm_models._core` Rust extension (see `src-rust/cb10.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _core, constants, utils

_COLUMNS = ("mag", "dip", "rake", "ztor", "rrup", "rjb", "vs30", "z2pt5")


def run_cb10(
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
) -> xr.Dataset:
    """Run the CB_10 model: Campbell and Bozorgnia (2008) NGA with the 2010 CAV model.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type, only active shallow is supported.
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``dip``, ``rake``,
        ``ztor``, ``rrup``, ``rjb``, ``vs30`` and ``z2pt5``.
    im : str
        Intensity measure: one of ``"PGA"``, ``"PGV"``, ``"CAV"``, ``"pSA"``.
    periods : Sequence[float], optional
        Periods to compute, required if `im` is ``"pSA"``. Ignored otherwise.

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
        If `tect_type` is not active shallow, or `rupture_df` is missing
        columns.
    """
    if tect_type is not constants.TectType.ACTIVE_SHALLOW:
        raise ValueError(f"CB_10 does not support tectonic type {tect_type}")
    utils.validate_columns(rupture_df, set(_COLUMNS))

    inputs = _core.CB10Inputs(
        *(rupture_df[col].to_numpy(dtype=np.float64) for col in _COLUMNS)
    )
    return utils.evaluate_ims(
        lambda core_im: _core.cb10(inputs, core_im),
        rupture_df.index,
        im,
        periods,
        attrs={"model": "CB_10", "tect_type": str(tect_type)},
    )
