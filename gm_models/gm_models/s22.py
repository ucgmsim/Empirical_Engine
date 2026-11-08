"""Python scaffolding around the Rust S_22 (Stafford 2022 NZ NSHM backbone) implementation.

Column/input validation, unit handling and epistemic-branch handling live
here; the actual GMPE math lives in the compiled `gm_models._core` Rust
extension (see `src-rust/s22.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core
from gm_models.a22 import _BRANCHES
from gm_models.constants import EpistemicBranch, TectType
from gm_models.utils import evaluate_ims, validate_columns

_REQUIRED_COLUMNS = {
    "mag",
    "rake",
    "dip",
    "ztor",
    "rrup",
    "rjb",
    "rx",
    "vs30",
    "z1pt0",
}


def run_s22(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: EpistemicBranch = EpistemicBranch.CENTRAL,
) -> xr.Dataset:
    """Run the S_22 (Stafford 2022 NZ NSHM backbone) model.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (active shallow or volcanic, which use the same model).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rake``, ``dip``,
        ``ztor``, ``rrup``, ``rjb``, ``rx``, ``vs30`` and ``z1pt0`` (in km, as
        in `oq_wrapper`).
    im : str
        Intensity measure: ``"PGA"`` or ``"pSA"``.
    periods : Sequence[float], optional
        Periods to compute, required if `im` is ``"pSA"``. Ignored otherwise.
    epistemic_branch : EpistemicBranch
        Epistemic branch of the mean (the sigma branch stays central, as in
        `oq_wrapper`). Defaults to `EpistemicBranch.CENTRAL`.

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
        If `tect_type` is unsupported, `rupture_df` lacks required columns,
        ``z1pt0`` looks like it is in metres, or `im` is not defined by the
        model.
    """
    if tect_type not in (TectType.ACTIVE_SHALLOW, TectType.VOLCANIC):
        raise ValueError(f"Unsupported tect_type {tect_type!r}")
    validate_columns(rupture_df, _REQUIRED_COLUMNS)
    if (rupture_df["z1pt0"] >= 10).any():
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )

    mag, rake, dip, ztor, rrup, rjb, rx, vs30, z1pt0 = (
        rupture_df[col].to_numpy(dtype=np.float64)
        for col in ("mag", "rake", "dip", "ztor", "rrup", "rjb", "rx", "vs30", "z1pt0")
    )
    # The model takes z1pt0 in metres.
    inputs = _core.S22Inputs(mag, rake, dip, ztor, rrup, rjb, rx, vs30, z1pt0 * 1000)
    return evaluate_ims(
        lambda core_im: _core.s22(inputs, core_im, _BRANCHES[epistemic_branch]),
        rupture_df.index,
        im,
        periods,
        attrs={
            "model": "S_22",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
        },
    )
