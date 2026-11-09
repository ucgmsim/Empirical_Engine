"""Python scaffolding around the Rust BCH_16 (BC Hydro, ESHM20 theta6) implementation.

Column/input validation and `oq_wrapper`'s ``xvf`` default live here; the GMPE
math lives in the compiled `gm_models._core` Rust extension (see
`src-rust/bch16.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _core, constants, utils

_REQUIRED_COLUMNS = {
    constants.TectType.SUBDUCTION_INTERFACE: {"mag", "rrup", "vs30"},
    constants.TectType.SUBDUCTION_SLAB: {"mag", "rhypo", "vs30", "hypo_depth"},
}


def run_bch16(
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
) -> xr.Dataset:
    """Run the BCH_16 model (OpenQuake's ESHM20 BC Hydro, "Step" forearc/backarc taper).

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``vs30`` and ``rrup``
        (interface) or ``rhypo`` and ``hypo_depth`` (slab). The optional
        ``xvf`` column (signed distance to the volcanic front in km, positive
        on the forearc side, negative on the backarc side) defaults to 0, as in
        `oq_wrapper`.
    im : str
        Intensity measure: one of ``"PGA"``, ``"pSA"``.
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
        If `tect_type` is unsupported or a required column is missing.
    """
    if tect_type not in _REQUIRED_COLUMNS:
        raise ValueError(f"BCH_16 does not support tectonic type {tect_type}")
    utils.validate_columns(rupture_df, _REQUIRED_COLUMNS[tect_type])

    mag, vs30 = (rupture_df[col].to_numpy(dtype=np.float64) for col in ("mag", "vs30"))
    xvf = (
        rupture_df["xvf"].to_numpy(dtype=np.float64)
        if "xvf" in rupture_df
        else np.zeros(len(rupture_df))
    )
    inputs = (
        _core.Bch16Inputs.interface(
            mag, rupture_df["rrup"].to_numpy(dtype=np.float64), vs30, xvf
        )
        if tect_type is constants.TectType.SUBDUCTION_INTERFACE
        else _core.Bch16Inputs.slab(
            mag,
            rupture_df["rhypo"].to_numpy(dtype=np.float64),
            vs30,
            xvf,
            rupture_df["hypo_depth"].to_numpy(dtype=np.float64),
        )
    )
    return utils.evaluate_ims(
        lambda core_im: _core.bch16(inputs, core_im),
        rupture_df.index,
        im,
        periods,
        attrs={"model": "BCH_16", "tect_type": str(tect_type)},
    )
