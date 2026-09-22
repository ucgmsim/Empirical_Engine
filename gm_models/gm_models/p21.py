"""Python scaffolding around the Rust P_21 (NZ NSHM2022 Parker et al. 2020) implementation.

Column/input validation, epistemic-branch handling, and result assembly live
here; the actual GMPE math lives in the compiled `gm_models._utils` Rust
extension (see `src-rust/p21.rs`).
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
import pandas as pd

from gm_models import _utils


class TectType(StrEnum):
    """Tectonic type supported by P_21."""

    SUBDUCTION_INTERFACE = "SUBDUCTION_INTERFACE"
    SUBDUCTION_SLAB = "SUBDUCTION_SLAB"


class EpistemicBranch(StrEnum):
    """Epistemic uncertainty branch, mirroring `oq_wrapper.constants.EpistemicBranch`."""

    LOWER = "LOWER"
    CENTRAL = "CENTRAL"
    UPPER = "UPPER"


# Matches oq_wrapper.constants.GMM_EPISTEMIC_BRANCH_KWARGS_MAPPING[GMM.P_21].
_SIGMA_MU_EPSILON = {
    EpistemicBranch.LOWER: -1.2815,
    EpistemicBranch.CENTRAL: 0.0,
    EpistemicBranch.UPPER: 1.2815,
}

_REQUIRED_COLUMNS = {
    TectType.SUBDUCTION_INTERFACE: {"mag", "rrup", "vs30"},
    TectType.SUBDUCTION_SLAB: {"mag", "rrup", "vs30", "hypo_depth", "backarc"},
}


def _validate_columns(tect_type: TectType, rupture_df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if `rupture_df` is missing required columns.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type being run, which determines the required columns.
    rupture_df : pd.DataFrame
        Input rupture/site dataframe to validate.
    """
    required = _REQUIRED_COLUMNS[tect_type]
    missing = required - set(rupture_df.columns)
    if missing:
        raise ValueError(
            f"rupture_df is missing required columns for {tect_type}: {sorted(missing)}"
        )


def _run_one(
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    period: float,
    sigma_mu_epsilon: float,
    modified_sigma: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mag = rupture_df["mag"].to_numpy(dtype=np.float64)
    rrup = rupture_df["rrup"].to_numpy(dtype=np.float64)
    vs30 = rupture_df["vs30"].to_numpy(dtype=np.float64)

    if tect_type is TectType.SUBDUCTION_INTERFACE:
        return _utils._p21_interface(
            mag, rrup, vs30, im, period, sigma_mu_epsilon, modified_sigma
        )

    hypo_depth = rupture_df["hypo_depth"].to_numpy(dtype=np.float64)
    backarc = rupture_df["backarc"].to_numpy(dtype=np.bool_)
    return _utils._p21_slab(
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
    modified_sigma: bool = False,
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
        model instead of the original (linear) one. Defaults to False,
        matching `oq_wrapper`'s default for P_21 in the NSHM2022 logic tree.

    Returns
    -------
    pd.DataFrame
        Columns ``{im}_mean``, ``{im}_std_Total``, ``{im}_std_Inter``,
        ``{im}_std_Intra`` (one set per period for ``pSA``), indexed by
        `rupture_df`'s index. Mean is in natural-log space, matching
        `oq_wrapper.run_gmm`'s output contract.
    """
    _validate_columns(tect_type, rupture_df)
    sigma_mu_epsilon = _SIGMA_MU_EPSILON[epistemic_branch]

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
