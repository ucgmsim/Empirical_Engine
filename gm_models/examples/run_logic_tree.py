"""Run the NZ NSHM 2022 GMCM logic tree for historic NZGMDB records.

Records are grouped by tectonic class, and each group is run with the logic
tree of its tectonic type. The result holds the combined ``ln_mean`` and
``std_total`` plus every branch's values along the ``branch`` dimension.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import gm_models as gmm

record_df = pd.read_parquet(
    Path(__file__).parents[2]
    / "oq_wrapper/tests/benchmark_data/nzgmdb_v4p3_rupture_df.parquet"
)
rupture_df = record_df.rename(
    columns={
        "r_rup": "rrup",
        "r_jb": "rjb",
        "r_x": "rx",
        "r_y": "ry0",
        "z_tor": "ztor",
        "z_bor": "zbot",
        "ev_depth": "hypo_depth",
        "Vs30": "vs30",
        "Z1.0": "z1pt0",
        "Z2.5": "z2pt5",
    }
).dropna()
rupture_df["vs30measured"] = True
rupture_df["backarc"] = False
rupture_df["z1pt0"] /= 1000  # m -> km
rupture_df["width"] = (rupture_df["zbot"] - rupture_df["ztor"]) / np.sin(
    np.radians(rupture_df["dip"])
)

periods = [0.1, 0.5, 1.0, 3.0]
results = {
    tect_type: gmm.run_nshm2022(
        tect_type, rupture_df[rupture_df["tect_class"] == tect_class], "pSA", periods
    )
    for tect_class, tect_type in [
        ("Crustal", gmm.constants.TectType.ACTIVE_SHALLOW),
        ("Interface", gmm.constants.TectType.SUBDUCTION_INTERFACE),
        ("Slab", gmm.constants.TectType.SUBDUCTION_SLAB),
    ]
}
crustal = results[gmm.constants.TectType.ACTIVE_SHALLOW]

# Logic tree values: the mixture over all branches, ln units, (record, period).
print("Crustal logic tree at 1 s, median pSA [g] and total sigma:")
print(
    pd.DataFrame(
        {
            "median": xr.ufuncs.exp(crustal["ln_mean"]).sel(period=1.0).to_pandas(),
            "std_total": crustal["std_total"].sel(period=1.0).to_pandas(),
        }
    ).head()
)

# A single branch, selected by model and epistemic branch.
s22_central = crustal.sel(
    model="S_22", epistemic_branch=gmm.constants.EpistemicBranch.CENTRAL
)
print(f"\nS_22 central branch (weight {s22_central['weight'].item():.3f}) at 1 s:")
print(
    pd.DataFrame(
        {
            "median": xr.ufuncs.exp(s22_central["branch_ln_mean"])
            .sel(period=1.0)
            .to_pandas(),
            "std_total": s22_central["branch_std_total"].sel(period=1.0).to_pandas(),
        }
    ).head()
)
