"""Python scaffolding around the Rust P_21 (NZ NSHM2022 Parker et al. 2020) implementation.

Column/input validation and epistemic-branch handling live here; the actual
GMPE math lives in the compiled `gm_models._core` Rust
extension (see `src-rust/p21.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _core, constants, utils

_REQUIRED_COLUMNS = {
    constants.TectType.SUBDUCTION_INTERFACE: {"mag", "rrup", "vs30"},
    constants.TectType.SUBDUCTION_SLAB: {
        "mag",
        "rrup",
        "vs30",
        "hypo_depth",
        "backarc",
    },
}


def run_p21(
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: constants.EpistemicBranch = constants.EpistemicBranch.CENTRAL,
    modified_sigma: bool = True,
) -> xr.Dataset:
    """Run the P_21 (NZ NSHM2022 Parker et al. 2020) model, global (GLO) region.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30``, and
        (for `TectType.SUBDUCTION_SLAB`) ``hypo_depth`` and ``backarc``.
    im : str
        Intensity measure: one of ``"PGA"``, ``"PGV"``, ``"pSA"``.
    periods : Sequence[float], optional
        Periods to compute, required if `im` is ``"pSA"``. Ignored otherwise.
    epistemic_branch : EpistemicBranch
        Epistemic uncertainty branch. Defaults to `EpistemicBranch.CENTRAL`.
    modified_sigma : bool
        Whether to use the NZ "modified sigma" nonlinear standard-deviation
        model instead of the original (linear) one. Defaults to True, as in
        the NSHM 2022 GMCM logic tree (Bradley et al. 2024) and `oq_wrapper`.

    Returns
    -------
    xr.Dataset
        Variables ``ln_mean`` (natural-log space), ``std_total``, ``std_inter``
        and ``std_intra`` over `rupture_df`'s index (plus ``period`` for
        ``pSA``), with the model settings as attributes. See
        `gm_models.utils.evaluate_ims`.
    """
    utils.validate_columns(rupture_df, _REQUIRED_COLUMNS[tect_type])

    mag, rrup, vs30 = (
        rupture_df[col].to_numpy(dtype=np.float64) for col in ("mag", "rrup", "vs30")
    )
    inputs = (
        _core.P21Inputs.interface(mag, rrup, vs30)
        if tect_type is constants.TectType.SUBDUCTION_INTERFACE
        else _core.P21Inputs.slab(
            mag,
            rrup,
            vs30,
            rupture_df["hypo_depth"].to_numpy(dtype=np.float64),
            rupture_df["backarc"].to_numpy(dtype=np.bool_),
        )
    )
    return utils.evaluate_ims(
        lambda core_im: _core.p21(
            inputs,
            core_im,
            constants.SIGMA_MU_EPSILON[epistemic_branch],
            modified_sigma,
        ),
        rupture_df.index,
        im,
        periods,
        # modified_sigma as int: netCDF attributes can't be bool.
        attrs={
            "model": "P_21",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
            "modified_sigma": int(modified_sigma),
        },
    )
