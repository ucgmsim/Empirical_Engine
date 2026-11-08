"""Python scaffolding around the Rust AS_16 (Afshari and Stewart 2016) implementation.

Column/input validation lives here; the model math lives in the compiled
`gm_models._core` Rust extension (see `src-rust/as16.rs`).
"""

import numpy as np
import pandas as pd
import xarray as xr

from gm_models import _core
from gm_models.constants import TectType
from gm_models.utils import evaluate_ims, validate_columns


def run_as16(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
) -> xr.Dataset:
    """Run the AS_16 (Afshari and Stewart 2016) significant duration model.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type, only `TectType.ACTIVE_SHALLOW` is supported.
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rake``, ``rrup``,
        ``vs30`` and ``z1pt0`` (in km, as for `oq_wrapper.run_gmm`).
    im : str
        Intensity measure: one of ``"Ds575"``, ``"Ds595"``, ``"Ds2080"``.

    Returns
    -------
    xr.Dataset
        Variables ``ln_mean`` (ln of the duration in s), ``std_total``,
        ``std_inter`` and ``std_intra`` over `rupture_df`'s index. See
        `gm_models.utils.evaluate_ims`.

    Raises
    ------
    ValueError
        If `tect_type` is not `TectType.ACTIVE_SHALLOW`, columns are missing,
        ``z1pt0`` looks like it is in metres, or `im` is not supported.
    """
    if tect_type is not TectType.ACTIVE_SHALLOW:
        raise ValueError(f"AS_16 does not support tectonic type {tect_type}")
    validate_columns(rupture_df, {"mag", "rake", "rrup", "vs30", "z1pt0"})
    if (rupture_df["z1pt0"] >= 10).any():
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )
    inputs = _core.AS16Inputs(
        *(
            rupture_df[col].to_numpy(dtype=np.float64) * (1000 if col == "z1pt0" else 1)
            for col in ("mag", "rake", "rrup", "vs30", "z1pt0")
        )
    )
    return evaluate_ims(
        lambda core_im: _core.as16(inputs, core_im),
        rupture_df.index,
        im,
        attrs={"model": "AS_16", "tect_type": str(tect_type)},
    )
