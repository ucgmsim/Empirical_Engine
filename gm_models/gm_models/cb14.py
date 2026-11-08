"""Python scaffolding around the Rust CB_14 (Campbell and Bozorgnia, 2014) implementation.

Column validation and epistemic-branch handling live here; the actual GMPE
math lives in the compiled `gm_models._core` Rust extension (see
`src-rust/cb14.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core
from gm_models.constants import SIGMA_MU_EPSILON, EpistemicBranch, TectType
from gm_models.utils import evaluate_ims, validate_columns

_REQUIRED_COLUMNS = {
    "mag",
    "rake",
    "dip",
    "ztor",
    "zbot",
    "hypo_depth",
    "rrup",
    "rjb",
    "rx",
    "vs30",
    "z2pt5",
}


def run_cb14(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: EpistemicBranch = EpistemicBranch.CENTRAL,
) -> xr.Dataset:
    """Run the CB_14 (Campbell and Bozorgnia, 2014) model.

    The rupture width is always estimated (Eq. 39 of the paper, from ``mag``,
    ``dip``, ``ztor`` and ``zbot``), as `oq_wrapper` does.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (active shallow or volcanic).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rake``, ``dip``,
        ``ztor``, ``zbot``, ``hypo_depth``, ``rrup``, ``rjb``, ``rx``,
        ``vs30`` and ``z2pt5``.
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
        If `tect_type` is not active shallow or volcanic, or `rupture_df` is
        missing columns.
    """
    if tect_type not in (TectType.ACTIVE_SHALLOW, TectType.VOLCANIC):
        raise ValueError(f"CB_14 does not support tectonic type {tect_type}")
    validate_columns(rupture_df, _REQUIRED_COLUMNS)

    order = (
        "mag",
        "rake",
        "dip",
        "ztor",
        "zbot",
        "hypo_depth",
        "rrup",
        "rjb",
        "rx",
        "vs30",
        "z2pt5",
    )
    inputs = _core.CB14Inputs(
        *(rupture_df[col].to_numpy(dtype=np.float64) for col in order)
    )
    return evaluate_ims(
        lambda core_im: _core.cb14(inputs, core_im, SIGMA_MU_EPSILON[epistemic_branch]),
        rupture_df.index,
        im,
        periods,
        attrs={
            "model": "CB_14",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
        },
    )
