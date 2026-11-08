"""Python scaffolding around the Rust P_20 (Parker et al. 2020) implementation.

Column/input validation lives here; the actual GMPE math lives in the compiled
`gm_models._core` Rust extension (see `src-rust/p20.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core
from gm_models.constants import TectType
from gm_models.utils import evaluate_ims, validate_columns

_REQUIRED_COLUMNS = {
    TectType.SUBDUCTION_INTERFACE: {"mag", "rrup", "vs30"},
    TectType.SUBDUCTION_SLAB: {"mag", "rrup", "vs30", "hypo_depth"},
}


def run_p20(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
) -> xr.Dataset:
    """Run the P_20 (Parker et al. 2020) model, global version.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30``, and
        (for `TectType.SUBDUCTION_SLAB`) ``hypo_depth``.
    im : str
        Intensity measure: one of ``"PGA"``, ``"PGV"``, ``"pSA"``.
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
        If `tect_type` is unsupported, or `rupture_df` is missing columns.
    """
    if tect_type not in _REQUIRED_COLUMNS:
        raise ValueError(f"P_20 does not support tectonic type {tect_type}")
    validate_columns(rupture_df, _REQUIRED_COLUMNS[tect_type])

    mag, rrup, vs30 = (
        rupture_df[col].to_numpy(dtype=np.float64) for col in ("mag", "rrup", "vs30")
    )
    inputs = (
        _core.P20Inputs.interface(mag, rrup, vs30)
        if tect_type is TectType.SUBDUCTION_INTERFACE
        else _core.P20Inputs.slab(
            mag, rrup, vs30, rupture_df["hypo_depth"].to_numpy(dtype=np.float64)
        )
    )
    return evaluate_ims(
        lambda core_im: _core.p20(inputs, core_im),
        rupture_df.index,
        im,
        periods,
        attrs={"model": "P_20", "tect_type": str(tect_type)},
    )
