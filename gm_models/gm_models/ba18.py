"""Python scaffolding around the Rust BA_18 (Bayless and Abrahamson 2018) implementation.

Column/input validation lives here; the model math lives in the compiled
`gm_models._core` Rust extension (see `src-rust/ba18.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core
from gm_models.constants import TectType
from gm_models.utils import evaluate_ims, validate_columns


def run_ba18(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    frequencies: Sequence[float] | None = None,
) -> xr.Dataset:
    """Run the BA_18 (Bayless and Abrahamson 2018) Fourier amplitude model.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type, only `TectType.ACTIVE_SHALLOW` is supported.
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rake``, ``ztor``,
        ``rrup``, ``vs30`` and ``z1pt0`` (in km, as for `oq_wrapper.run_gmm`).
    im : str
        Intensity measure, only ``"EAS"`` is supported.
    frequencies : Sequence[float], optional
        Frequencies (Hz) to compute, required (within [0.1, 97.723724] Hz).

    Returns
    -------
    xr.Dataset
        Variables ``ln_mean`` (ln of the effective amplitude spectrum in g-s),
        ``std_total``, ``std_inter`` and ``std_intra`` with dimensions
        ``(site, frequency)``. See `gm_models.utils.evaluate_ims`. Note
        ``std_total`` includes a term beyond inter- and intra-event
        variability, so it is larger than their root-sum-square.

    Raises
    ------
    ValueError
        If `tect_type` is not `TectType.ACTIVE_SHALLOW`, columns are missing,
        ``z1pt0`` looks like it is in metres, `im` is not ``"EAS"``, or a
        frequency is outside the supported range.
    """
    if tect_type is not TectType.ACTIVE_SHALLOW:
        raise ValueError(f"BA_18 does not support tectonic type {tect_type}")
    if im != "EAS":
        raise ValueError(f"BA_18 only supports im='EAS', got '{im}'")
    columns = ("mag", "rake", "ztor", "rrup", "vs30", "z1pt0")
    validate_columns(rupture_df, set(columns))
    if (rupture_df["z1pt0"] >= 10).any():
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )
    inputs = _core.BA18Inputs(
        *(
            rupture_df[col].to_numpy(dtype=np.float64) * (1000 if col == "z1pt0" else 1)
            for col in columns
        )
    )
    return evaluate_ims(
        lambda core_im: _core.ba18(inputs, core_im),
        rupture_df.index,
        im,
        frequencies=frequencies,
        attrs={"model": "BA_18", "tect_type": str(tect_type)},
    )
