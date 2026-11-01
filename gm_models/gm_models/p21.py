"""Python scaffolding around the Rust P_21 (NZ NSHM2022 Parker et al. 2020) implementation.

Column/input validation, epistemic-branch handling, and result assembly live
here; the actual GMPE math lives in the compiled `gm_models._core` Rust
extension (see `src-rust/p21.rs`).
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from gm_models import _core
from gm_models.constants import P21_SIGMA_MU_EPSILON, EpistemicBranch, TectType
from gm_models.utils import validate_columns

_REQUIRED_COLUMNS = {
    TectType.SUBDUCTION_INTERFACE: {"mag", "rrup", "vs30"},
    TectType.SUBDUCTION_SLAB: {"mag", "rrup", "vs30", "hypo_depth", "backarc"},
}


def _run_one(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    period: float,
    sigma_mu_epsilon: float,
    modified_sigma: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run the compiled `_core` extension once, for a single IM/period.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe, already column-validated.
    im : str
        Intensity measure passed through to the extension: ``"PGA"``,
        ``"PGV"``, or ``"SA"``.
    period : float
        Period in seconds, used only when `im` is ``"SA"``.
    sigma_mu_epsilon : float
        Epistemic uncertainty offset for the "modified sigma" branch.
    modified_sigma : bool
        Whether to use the NZ "modified sigma" standard-deviation model.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(mean, std_total, std_inter, std_intra)``, mean in ln space.
    """
    mag = rupture_df["mag"].to_numpy(dtype=np.float64)
    rrup = rupture_df["rrup"].to_numpy(dtype=np.float64)
    vs30 = rupture_df["vs30"].to_numpy(dtype=np.float64)

    if tect_type is TectType.SUBDUCTION_INTERFACE:
        return _core._p21_interface(
            mag, rrup, vs30, im, period, sigma_mu_epsilon, modified_sigma
        )

    hypo_depth = rupture_df["hypo_depth"].to_numpy(dtype=np.float64)
    backarc = rupture_df["backarc"].to_numpy(dtype=np.bool_)
    return _core._p21_slab(
        mag,
        rrup,
        vs30,
        hypo_depth,
        backarc,
        im,
        period,
        sigma_mu_epsilon,
        modified_sigma,
    )


def run_p21(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
    epistemic_branch: EpistemicBranch = EpistemicBranch.CENTRAL,
    modified_sigma: bool = True,
) -> pd.DataFrame:
    """Run the P_21 (NZ NSHM2022 Parker et al. 2020) model, global (GLO) region.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type (subduction interface or slab).
    rupture_df : pd.DataFrame
        Rupture/site dataframe. Must contain ``mag``, ``rrup``, ``vs30``, and
        (for `TectType.SUBDUCTION_SLAB`) ``hypo_depth`` and ``backarc``.
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
    pd.DataFrame
        Columns ``{im}_mean``, ``{im}_std_Total``, ``{im}_std_Inter``,
        ``{im}_std_Intra`` (one set per period for ``pSA``), indexed by
        `rupture_df`'s index. Mean is in natural-log space, matching
        `oq_wrapper.run_gmm`'s output contract.
    """
    validate_columns(rupture_df, _REQUIRED_COLUMNS[tect_type])
    sigma_mu_epsilon = P21_SIGMA_MU_EPSILON[epistemic_branch]

    columns: dict[str, np.ndarray] = {}
    if im == "pSA":
        if not periods:
            raise ValueError("periods must be provided when im='pSA'")
        for period in periods:
            mean, total, inter, intra = _run_one(
                tect_type,
                rupture_df,
                "SA",
                float(period),
                sigma_mu_epsilon,
                modified_sigma,
            )
            prefix = f"pSA_{period}"
            columns[f"{prefix}_mean"] = mean
            columns[f"{prefix}_std_Total"] = total
            columns[f"{prefix}_std_Inter"] = inter
            columns[f"{prefix}_std_Intra"] = intra
    elif im in ("PGA", "PGV"):
        mean, total, inter, intra = _run_one(
            tect_type, rupture_df, im, 0.0, sigma_mu_epsilon, modified_sigma
        )
        columns[f"{im}_mean"] = mean
        columns[f"{im}_std_Total"] = total
        columns[f"{im}_std_Inter"] = inter
        columns[f"{im}_std_Intra"] = intra
    else:
        raise ValueError(f"Unsupported im '{im}': expected one of 'PGA', 'PGV', 'pSA'")

    return pd.DataFrame(columns, index=rupture_df.index)
