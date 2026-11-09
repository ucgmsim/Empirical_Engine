"""NZ NSHM 2022 ground motion characterisation model (GMCM) logic tree.

Weights from Bradley et al. (2024), as in `oq_wrapper`'s
``gmm_logic_tree_configs/nshm_2022_config.yaml``: each model's weight is split
0.3 / 0.4 / 0.3 over its LOWER / CENTRAL / UPPER epistemic branches.
"""

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from . import a22, ag20, ask14, br13, bssa14, cb14, constants, cy14, k20, p21, s22

BRANCH_WEIGHTS = {
    constants.EpistemicBranch.LOWER: 0.3,
    constants.EpistemicBranch.CENTRAL: 0.4,
    constants.EpistemicBranch.UPPER: 0.3,
}

# Crustal: A_22 and S_22, plus the four NGA-West2 models and Br_13 sharing 0.33.
_CRUSTAL_MODELS: dict[str, tuple[Callable[..., xr.Dataset], float]] = {
    "A_22": (a22.run_a22, 0.28),
    "S_22": (s22.run_s22, 0.39),
    "ASK_14": (ask14.run_ask14, 0.33 * 0.2),
    "CY_14": (cy14.run_cy14, 0.33 * 0.2),
    "CB_14": (cb14.run_cb14, 0.33 * 0.2),
    "BSSA_14": (bssa14.run_bssa14, 0.33 * 0.2),
    "Br_13": (br13.run_br13, 0.33 * 0.2),
}

# Model name -> (runner, model weight), per tectonic type.
MODELS: dict[constants.TectType, dict[str, tuple[Callable[..., xr.Dataset], float]]] = {
    constants.TectType.ACTIVE_SHALLOW: _CRUSTAL_MODELS,
    constants.TectType.VOLCANIC: _CRUSTAL_MODELS,
    constants.TectType.SUBDUCTION_INTERFACE: {
        "A_22": (a22.run_a22, 0.27),
        "AG_20": (ag20.run_ag20, 0.25),
        "K_20": (k20.run_k20, 0.24),
        "P_21": (p21.run_p21, 0.24),
    },
    constants.TectType.SUBDUCTION_SLAB: {
        "A_22": (a22.run_a22, 0.28),
        "AG_20": (ag20.run_ag20, 0.25),
        "K_20": (k20.run_k20, 0.24),
        "P_21": (p21.run_p21, 0.23),
    },
}


def run_nshm2022(
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[float] | None = None,
) -> xr.Dataset:
    """Run the NZ NSHM 2022 GMCM logic tree.

    The branches are combined as a mixture, as in
    `oq_wrapper.run_gmm_logic_tree`: the mean is the weighted mean of the
    branch ln means, and the total variance is the weighted mean of the
    branch variances plus the between-branch variance of the ln means.

    Parameters
    ----------
    tect_type : TectType
        Tectonic type.
    rupture_df : pd.DataFrame
        Rupture/site dataframe, with the columns every model of `tect_type`
        needs (see the respective ``run_*`` functions).
    im : str
        Intensity measure, only ``"pSA"`` (the logic tree defines no other).
    periods : Sequence[float], optional
        Periods to compute, required.

    Returns
    -------
    xr.Dataset
        ``ln_mean`` and ``std_total`` over ``(site, period)``, and the
        per-branch ``branch_ln_mean`` and ``branch_std_total`` over
        ``(branch, site, period)``. The ``branch`` dimension has coordinates
        ``model``, ``epistemic_branch`` and ``weight``. Inter- and intra-event
        standard deviations are not combined.

    Raises
    ------
    ValueError
        If `im` is not ``"pSA"``, `tect_type` is unsupported, or a model
        rejects `rupture_df` or `periods`.
    """
    if im != "pSA":
        raise ValueError(f"The NSHM 2022 logic tree only defines pSA, not {im}")
    if tect_type not in MODELS:
        raise ValueError(f"NSHM 2022 does not support tectonic type {tect_type}")
    models = MODELS[tect_type]
    branches = [(name, branch) for name in models for branch in BRANCH_WEIGHTS]
    ds = xr.concat(
        [
            models[name][0](
                tect_type, rupture_df, im, periods, epistemic_branch=branch
            )[["ln_mean", "std_total"]]
            for name, branch in branches
        ],
        dim="branch",
        combine_attrs="drop",
    ).assign_coords(
        model=("branch", [name for name, _ in branches]),
        epistemic_branch=("branch", [str(branch) for _, branch in branches]),
        weight=(
            "branch",
            [models[name][1] * BRANCH_WEIGHTS[branch] for name, branch in branches],
        ),
    )
    ln_mean = (ds.weight * ds.ln_mean).sum("branch")
    return xr.Dataset(
        {
            "ln_mean": ln_mean,
            "std_total": np.sqrt(
                (ds.weight * (ds.std_total**2 + (ds.ln_mean - ln_mean) ** 2)).sum(
                    "branch"
                )
            ),
            "branch_ln_mean": ds.ln_mean,
            "branch_std_total": ds.std_total,
        },
        attrs={"model": "NSHM2022", "tect_type": str(tect_type)},
    )
