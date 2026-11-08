"""Python scaffolding around the Rust K_20 and K_20_NZ (Kuehn et al. 2020) implementation.

Column/input validation and epistemic-branch handling live here; the actual
GMPE math lives in the compiled `gm_models._core` Rust extension (see
`src-rust/k20.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core
from gm_models.constants import SIGMA_MU_EPSILON, EpistemicBranch, TectType
from gm_models.utils import evaluate_ims, validate_columns


def _inputs(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    required: set[str],
    z1pt0: np.ndarray,
    backarc: np.ndarray | None,
) -> _core.K20Inputs:
    """Build the Rust inputs for `tect_type` after validating `rupture_df`.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe.
    required : set[str]
        Columns `rupture_df` must contain.
    z1pt0 : np.ndarray
        Z1.0 in metres, NaN where the basin term is not applied.
    backarc : np.ndarray or None
        Slab backarc flags, None for none (slab only).

    Returns
    -------
    _core.K20Inputs
        Inputs for `_core.k20`.

    Raises
    ------
    ValueError
        If `tect_type` is unsupported, or `rupture_df` is missing columns.
    """
    if tect_type not in (TectType.SUBDUCTION_INTERFACE, TectType.SUBDUCTION_SLAB):
        raise ValueError(
            f"Kuehn et al. (2020) does not support tectonic type {tect_type}"
        )
    validate_columns(rupture_df, required)
    mag, rrup, vs30, ztor = (
        rupture_df[col].to_numpy(dtype=np.float64)
        for col in ("mag", "rrup", "vs30", "ztor")
    )
    if tect_type is TectType.SUBDUCTION_INTERFACE:
        return _core.K20Inputs.interface(mag, rrup, vs30, ztor, z1pt0)
    return _core.K20Inputs.slab(
        mag,
        rrup,
        vs30,
        ztor,
        z1pt0,
        np.zeros(len(rupture_df), dtype=np.bool_) if backarc is None else backarc,
    )


def run_k20(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: EpistemicBranch = EpistemicBranch.CENTRAL,
    modified_sigma: bool = True,
) -> xr.Dataset:
    """Run the K_20 (NZ NSHM2022 Kuehn et al. 2020) model, global (GLO) region.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30``,
        ``ztor``, and (for `TectType.SUBDUCTION_SLAB`) ``backarc``.
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

    Raises
    ------
    ValueError
        If `tect_type` is unsupported, or `rupture_df` is missing columns.
    """
    required = {"mag", "rrup", "vs30", "ztor"}
    if tect_type is TectType.SUBDUCTION_SLAB:
        required.add("backarc")
    inputs = _inputs(
        tect_type,
        rupture_df,
        required,
        # No basin term in the global region: NaN depths are ignored.
        np.full(len(rupture_df), np.nan),
        rupture_df["backarc"].to_numpy(dtype=np.bool_)
        if tect_type is TectType.SUBDUCTION_SLAB
        else None,
    )
    return evaluate_ims(
        lambda core_im: _core.k20(
            inputs, core_im, SIGMA_MU_EPSILON[epistemic_branch], modified_sigma
        ),
        rupture_df.index,
        im,
        periods,
        # modified_sigma as int: netCDF attributes can't be bool.
        attrs={
            "model": "K_20",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
            "modified_sigma": int(modified_sigma),
        },
    )


def run_k20_nz(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
) -> xr.Dataset:
    """Run the K_20_NZ (Kuehn et al. 2020) model, New Zealand (NZL) region.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30``,
        ``ztor`` and ``z1pt0`` (in km, as for `oq_wrapper.run_gmm`).
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
        If `tect_type` is unsupported, `rupture_df` is missing columns, or
        ``z1pt0`` looks like metres (any value of 10 or more), as in `oq_wrapper`.
    """
    validate_columns(rupture_df, {"z1pt0"})
    z1pt0 = rupture_df["z1pt0"].to_numpy(dtype=np.float64)
    if np.any(z1pt0 >= 10):
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )
    inputs = _inputs(
        tect_type,
        rupture_df,
        {"mag", "rrup", "vs30", "ztor"},
        # The Rust model takes metres.
        z1pt0 * 1000,
        None,
    )
    return evaluate_ims(
        lambda core_im: _core.k20_nz(inputs, core_im),
        rupture_df.index,
        im,
        periods,
        attrs={"model": "K_20_NZ", "tect_type": str(tect_type)},
    )
