"""Run the NZ NSHM 2022 GMCM logic tree for historic NZGMDB records.

Records are grouped by tectonic class, and each group is run with the logic
tree of its tectonic type. The result holds the combined ``ln_mean`` and
``std_total`` plus every branch's values along the ``branch`` dimension.
"""

from pathlib import Path

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
        "r_y": "ry",
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

interface = results[gmm.constants.TectType.SUBDUCTION_INTERFACE]
# Median pSA(1 s) (g) and total sigma of the combined logic tree.
print(xr.ufuncs.exp(interface["ln_mean"].sel(period=1.0)).to_series().describe())
print(interface["std_total"].sel(period=1.0).to_series().describe())
# Each branch's weight and its median pSA(1 s) (g) averaged over all records.
print(
    xr.ufuncs.exp(interface["branch_ln_mean"].sel(period=1.0).mean("record_id"))
    .to_dataframe(name="psa_1s")
    .drop(columns="period")
)
