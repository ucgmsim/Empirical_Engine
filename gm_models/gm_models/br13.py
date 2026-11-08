"""Python scaffolding around the Rust Br_13 (Bradley, 2013) implementation.

Column/input validation and epistemic-branch handling live here; the actual
GMPE math lives in the compiled `gm_models._core` Rust extension (see
`src-rust/br13.rs`).
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
    "dip",
    "rake",
    "ztor",
    "rrup",
    "rjb",
    "rx",
    "vs30",
    "z1pt0",
    "vs30measured",
}


def run_br13(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: EpistemicBranch = EpistemicBranch.CENTRAL,
) -> xr.Dataset:
    """Run the Br_13 (Bradley, 2013) model (``Bradley2013`` / ``Bradley2013Volc``).

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (active shallow, or volcanic for `Bradley2013Volc`).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``dip``, ``rake``,
        ``ztor``, ``rrup``, ``rjb``, ``rx``, ``vs30`` and ``z1pt0`` (in km, as
        for `oq_wrapper.run_gmm`). ``vs30measured`` is required. ``is_aftershock`` is optional and
        defaults to False.
    im : str
        Intensity measure: one of ``"PGA"``, ``"pSA"``.
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
    if tect_type not in (TectType.ACTIVE_SHALLOW, TectType.VOLCANIC):
        raise ValueError(f"Br_13 does not support tectonic type {tect_type}")
    validate_columns(rupture_df, _REQUIRED_COLUMNS)
    if (rupture_df["z1pt0"] >= 10).any():
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )

    mag, dip, rake, ztor, rrup, rjb, rx, vs30 = (
        rupture_df[col].to_numpy(dtype=np.float64)
        for col in ("mag", "dip", "rake", "ztor", "rrup", "rjb", "rx", "vs30")
    )
    inputs = _core.Br13Inputs(
        mag,
        dip,
        rake,
        ztor,
        rrup,
        rjb,
        rx,
        vs30,
        rupture_df["vs30measured"].to_numpy(dtype=np.bool_),
        # km -> m, as oq_wrapper does for OpenQuake.
        rupture_df["z1pt0"].to_numpy(dtype=np.float64) * 1000,
        rupture_df.get("is_aftershock", pd.Series(False, index=rupture_df.index))
        .to_numpy()
        .astype(np.bool_),
    )
    return evaluate_ims(
        lambda core_im: _core.br13(
            inputs,
            core_im,
            SIGMA_MU_EPSILON[epistemic_branch],
            tect_type is TectType.VOLCANIC,
        ),
        rupture_df.index,
        im,
        periods,
        attrs={
            "model": "Br_13",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
        },
    )
