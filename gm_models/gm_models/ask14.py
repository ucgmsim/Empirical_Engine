"""Python scaffolding around the Rust ASK_14 (Abrahamson et al. 2014) implementation.

Column/input validation, `oq_wrapper`'s defaults for missing columns and
epistemic-branch handling live here; the actual GMPE math lives in the compiled
`gm_models._core` Rust extension (see `src-rust/ask14.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import _core, constants, utils

_REQUIRED_COLUMNS = {"mag", "rrup", "rx", "dip", "ztor", "rake", "vs30", "z1pt0"}


def run_ask14(
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: constants.EpistemicBranch = constants.EpistemicBranch.CENTRAL,
) -> xr.Dataset:
    """Run the ASK_14 (Abrahamson et al. 2014) model, default California model, mainshocks.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (active shallow or volcanic; the model is identical).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``rx``,
        ``dip``, ``ztor``, ``rake``, ``vs30``, ``z1pt0`` (km) and ``ry0`` (or
        ``ry``). As in `oq_wrapper`, ``vs30measured`` defaults to False and
        ``width`` to the ASK14 estimate from ``dip`` and ``mag``. Aftershocks
        are not supported: ``is_aftershock`` may be present only if all False.
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
        If `tect_type` is not active shallow or volcanic, columns are missing,
        z1pt0 looks like metres, or `rupture_df` flags aftershocks.
    """
    if tect_type not in (
        constants.TectType.ACTIVE_SHALLOW,
        constants.TectType.VOLCANIC,
    ):
        raise ValueError(f"ASK_14 does not support tect_type {tect_type}")
    utils.validate_columns(rupture_df, _REQUIRED_COLUMNS)
    if rupture_df.get("is_aftershock", pd.Series(False)).any():
        raise ValueError("ASK_14 aftershock scaling is not supported")
    if (rupture_df["z1pt0"] >= 10).any():
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )

    def column(name: str) -> np.ndarray:
        """Return column `name` of `rupture_df` as float64.

        Parameters
        ----------
        name : str
            Column name.

        Returns
        -------
        np.ndarray
            The column values.
        """
        return rupture_df[name].to_numpy(dtype=np.float64)

    # Same fallbacks as oq_wrapper's `prepare_model_inputs`.
    ry0 = column("ry0" if "ry0" in rupture_df else "ry")
    width = (
        column("width")
        if "width" in rupture_df
        else utils.estimate_width_ask14(column("dip"), column("mag"))
    )
    vs30measured = (
        rupture_df["vs30measured"].to_numpy(dtype=np.bool_)
        if "vs30measured" in rupture_df
        else np.zeros(len(rupture_df), dtype=np.bool_)
    )
    inputs = _core.Ask14Inputs(
        column("mag"),
        column("rrup"),
        column("rx"),
        ry0,
        column("dip"),
        column("ztor"),
        column("rake"),
        width,
        column("vs30"),
        column("z1pt0") * 1000,  # km -> m, as oq_wrapper does for OpenQuake
        vs30measured,
    )
    return utils.evaluate_ims(
        lambda core_im: _core.ask14(
            inputs, core_im, constants.SIGMA_MU_EPSILON[epistemic_branch]
        ),
        rupture_df.index,
        im,
        periods,
        attrs={
            "model": "ASK_14",
            "tect_type": str(tect_type),
            "epistemic_branch": str(epistemic_branch),
        },
    )
