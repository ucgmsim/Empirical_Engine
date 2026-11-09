"""Python scaffolding around the Rust BSSA_14 (Boore et al. 2014) implementation.

Column/input validation and epistemic-branch handling live here; the actual
GMPE math lives in the compiled `gm_models._core` Rust extension (see
`src-rust/bssa14.rs`).
"""

from collections.abc import Sequence

import pandas as pd
import xarray as xr

from . import _core, constants, utils


def run_bssa14(
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: constants.EpistemicBranch = constants.EpistemicBranch.CENTRAL,
) -> xr.Dataset:
    """Run the BSSA_14 (Boore et al. 2014) model, global model without basin term.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (active shallow or volcanic; the model is identical).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rake``, ``rjb``,
        ``rrup`` (only used by the epistemic adjustment, as in OpenQuake) and
        ``vs30``.
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
        If `tect_type` is not active shallow or volcanic, or columns are missing.
    """
    if tect_type not in (
        constants.TectType.ACTIVE_SHALLOW,
        constants.TectType.VOLCANIC,
    ):
        raise ValueError(f"BSSA_14 does not support tect_type {tect_type}")
    columns = ("mag", "rake", "rjb", "rrup", "vs30")
    utils.validate_columns(rupture_df, set(columns))
    inputs = _core.Bssa14Inputs(
        *(rupture_df[col].to_numpy(dtype="float64") for col in columns)
    )
    return utils.evaluate_ims(
        lambda core_im: _core.bssa14(
            inputs, core_im, constants.SIGMA_MU_EPSILON[epistemic_branch]
        ),
        rupture_df.index,
        im,
        periods,
        attrs={
            "model": "BSSA_14",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
        },
    )
