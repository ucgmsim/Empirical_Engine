"""Python scaffolding around the Rust A_22 (Atkinson 2022 NZ NSHM backbone) implementation.

Column/input validation and epistemic-branch handling live here; the actual
GMPE math lives in the compiled `gm_models._core` Rust extension (see
`src-rust/a22.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core
from gm_models.constants import EpistemicBranch, TectType
from gm_models.utils import evaluate_ims, validate_columns

_BRANCHES = {
    EpistemicBranch.LOWER: _core.Branch.Lower,
    EpistemicBranch.CENTRAL: _core.Branch.Central,
    EpistemicBranch.UPPER: _core.Branch.Upper,
}


def run_a22(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: EpistemicBranch = EpistemicBranch.CENTRAL,
    modified_sigma: bool = True,
) -> xr.Dataset:
    """Run the A_22 (Atkinson 2022 NZ NSHM backbone) model.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (active shallow, volcanic, subduction interface or slab).
        Volcanic uses the crustal model, as in `oq_wrapper`.
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30``, and
        (for `TectType.SUBDUCTION_SLAB`) ``backarc``.
    im : str
        Intensity measure: ``"PGA"`` or ``"pSA"``.
    periods : Sequence[float], optional
        Periods to compute, required if `im` is ``"pSA"``. Ignored otherwise.
    epistemic_branch : EpistemicBranch
        Epistemic uncertainty branch. Defaults to `EpistemicBranch.CENTRAL`.
    modified_sigma : bool
        Whether to use the NZ "modified sigma" nonlinear standard-deviation
        model instead of the tabulated one. Defaults to True, as in the NSHM
        2022 GMCM logic tree (Bradley et al. 2024) and `oq_wrapper`.

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
        If `tect_type` is unsupported, `rupture_df` lacks required columns, or
        `im` is not defined by the model.
    """
    if tect_type not in TectType:
        raise ValueError(f"Unsupported tect_type {tect_type!r}")
    required = {"mag", "rrup", "vs30"}
    if tect_type is TectType.SUBDUCTION_SLAB:
        required.add("backarc")
    validate_columns(rupture_df, required)

    mag, rrup, vs30 = (
        rupture_df[col].to_numpy(dtype=np.float64) for col in ("mag", "rrup", "vs30")
    )
    if tect_type is TectType.SUBDUCTION_SLAB:
        inputs = _core.A22Inputs.slab(
            mag, rrup, vs30, rupture_df["backarc"].to_numpy(dtype=np.bool_)
        )
    elif tect_type is TectType.SUBDUCTION_INTERFACE:
        inputs = _core.A22Inputs.interface(mag, rrup, vs30)
    else:
        inputs = _core.A22Inputs.crust(mag, rrup, vs30)
    return evaluate_ims(
        lambda core_im: _core.a22(
            inputs, core_im, _BRANCHES[epistemic_branch], modified_sigma
        ),
        rupture_df.index,
        im,
        periods,
        # modified_sigma as int: netCDF attributes can't be bool.
        attrs={
            "model": "A_22",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
            "modified_sigma": int(modified_sigma),
        },
    )
