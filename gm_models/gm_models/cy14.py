"""Python scaffolding around the Rust CY_14 (Chiou and Youngs, 2014) implementation.

Column/input validation and epistemic-branch handling live here; the actual
GMPE math lives in the compiled `gm_models._core` Rust extension (see
`src-rust/cy14.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _core, constants, utils

_REQUIRED_COLUMNS = {
    "mag",
    "dip",
    "rake",
    "ztor",
    "rrup",
    "rjb",
    "rx",
    "vs30",
    "z1pt0",
}


def run_cy14(
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: constants.EpistemicBranch = constants.EpistemicBranch.CENTRAL,
) -> xr.Dataset:
    """Run the CY_14 (Chiou and Youngs, 2014) model with OpenQuake's default arguments.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (active shallow or volcanic, which share the model).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``dip``, ``rake``,
        ``ztor``, ``rrup``, ``rjb``, ``rx``, ``vs30`` and ``z1pt0`` (in km, as
        for `oq_wrapper.run_gmm`). ``vs30measured`` is optional and defaults to
        False.
    im : str
        Intensity measure: one of ``"PGA"``, ``"PGV"``, ``"pSA"``.
    periods : Sequence[float], optional
        Periods to compute, required if `im` is ``"pSA"``. Ignored otherwise.
    epistemic_branch : EpistemicBranch
        Epistemic uncertainty branch. Defaults to `EpistemicBranch.CENTRAL`.

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
        If `tect_type` is not active shallow or volcanic, or required columns
        are missing.
    """
    if tect_type not in (
        constants.TectType.ACTIVE_SHALLOW,
        constants.TectType.VOLCANIC,
    ):
        raise ValueError(f"CY_14 does not support tectonic type {tect_type}")
    utils.validate_columns(rupture_df, _REQUIRED_COLUMNS)
    if (rupture_df["z1pt0"] >= 10).any():
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )

    mag, dip, rake, ztor, rrup, rjb, rx, vs30 = (
        rupture_df[col].to_numpy(dtype=np.float64)
        for col in ("mag", "dip", "rake", "ztor", "rrup", "rjb", "rx", "vs30")
    )
    inputs = _core.CY14Inputs(
        mag,
        dip,
        rake,
        ztor,
        rrup,
        rjb,
        rx,
        vs30,
        rupture_df.get("vs30measured", pd.Series(False, index=rupture_df.index))
        .to_numpy()
        .astype(np.bool_),
        # km -> m, as oq_wrapper does for OpenQuake.
        rupture_df["z1pt0"].to_numpy(dtype=np.float64) * 1000,
    )
    return utils.evaluate_ims(
        lambda core_im: _core.cy14(
            inputs, core_im, constants.SIGMA_MU_EPSILON[epistemic_branch]
        ),
        rupture_df.index,
        im,
        periods,
        attrs={
            "model": "CY_14",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
        },
    )
