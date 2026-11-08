"""Python scaffolding around the Rust AG_20 / AG_20_NZ (Abrahamson and Gulerce 2020) implementation.

Column/input validation and epistemic-branch handling live here; the actual
GMPE math lives in the compiled `gm_models._core` Rust
extension (see `src-rust/ag20.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core
from gm_models.constants import SIGMA_MU_EPSILON, EpistemicBranch, TectType
from gm_models.utils import evaluate_ims, validate_columns

_REQUIRED_COLUMNS = {
    TectType.SUBDUCTION_INTERFACE: {"mag", "rrup", "vs30"},
    TectType.SUBDUCTION_SLAB: {"mag", "rrup", "vs30", "ztor"},
}


def _run(
    model: str,
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None,
    epistemic_branch: EpistemicBranch,
) -> xr.Dataset:
    """Run AG20 as ``model`` (``"AG_20"`` or ``"AG_20_NZ"``)."""
    if tect_type not in _REQUIRED_COLUMNS:
        raise ValueError(f"{model} does not support tectonic type {tect_type}")
    is_nz = model == "AG_20_NZ"
    # Only the NZ22 (global) variant has the slab backarc term.
    has_backarc = not is_nz and tect_type is TectType.SUBDUCTION_SLAB
    validate_columns(
        rupture_df,
        _REQUIRED_COLUMNS[tect_type] | ({"backarc"} if has_backarc else set()),
    )

    mag, rrup, vs30 = (
        rupture_df[col].to_numpy(dtype=np.float64) for col in ("mag", "rrup", "vs30")
    )
    inputs = (
        _core.AG20Inputs.interface(mag, rrup, vs30)
        if tect_type is TectType.SUBDUCTION_INTERFACE
        else _core.AG20Inputs.slab(
            mag,
            rrup,
            vs30,
            rupture_df["ztor"].to_numpy(dtype=np.float64),
            rupture_df["backarc"].to_numpy(dtype=np.bool_)
            if has_backarc
            else np.zeros(len(rupture_df), dtype=np.bool_),
        )
    )
    return evaluate_ims(
        lambda core_im: _core.ag20(
            inputs, core_im, is_nz, SIGMA_MU_EPSILON[epistemic_branch]
        ),
        rupture_df.index,
        im,
        periods,
        attrs={
            "model": model,
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
        },
    )


def run_ag20(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: EpistemicBranch = EpistemicBranch.CENTRAL,
) -> xr.Dataset:
    """Run the AG_20 (NZ NSHM2022 Abrahamson and Gulerce 2020) model, global (GLO) region.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30``, and
        (for `TectType.SUBDUCTION_SLAB`) ``ztor`` and ``backarc``.
    im : str
        Intensity measure: ``"PGA"`` or ``"pSA"`` (PGV is not defined).
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
        If `tect_type` is unsupported or columns are missing.
    """
    return _run("AG_20", tect_type, rupture_df, im, periods, epistemic_branch)


def run_ag20_nz(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
) -> xr.Dataset:
    """Run the AG_20_NZ (Abrahamson and Gulerce 2020, New Zealand region) model.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30``, and
        (for `TectType.SUBDUCTION_SLAB`) ``ztor``.
    im : str
        Intensity measure: ``"PGA"`` or ``"pSA"`` (PGV is not defined).
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
        If `tect_type` is unsupported or columns are missing.
    """
    return _run("AG_20_NZ", tect_type, rupture_df, im, periods, EpistemicBranch.CENTRAL)
