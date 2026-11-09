"""Run a single ground motion model (Bradley 2013) for historic NZGMDB records.

Each ``run_*`` function takes the tectonic type, a rupture/site dataframe
(OpenQuake column names, Z1.0 in km) and the IM, and returns an
`xarray.Dataset` with ``ln_mean``, ``std_total``, ``std_inter`` and
``std_intra`` (plus a ``period`` dimension for pSA). The columns a model needs
are listed in its ``run_*`` docstring.
"""

from pathlib import Path

import pandas as pd
import xarray as xr

import gm_models as gmm

record_df = pd.read_parquet(
    Path(__file__).parents[2]
    / "oq_wrapper/tests/benchmark_data/nzgmdb_v4p3_rupture_df.parquet"
)
rupture_df = (
    record_df[record_df["tect_class"] == "Crustal"]
    .rename(
        columns={
            "r_rup": "rrup",
            "r_jb": "rjb",
            "r_x": "rx",
            "z_tor": "ztor",
            "Vs30": "vs30",
            "Z1.0": "z1pt0",
        }
    )[["mag", "rake", "dip", "ztor", "rrup", "rjb", "rx", "vs30", "z1pt0"]]
    .dropna()
)
rupture_df["vs30measured"] = True
rupture_df["z1pt0"] /= 1000  # m -> km

pga = gmm.run_br13(gmm.constants.TectType.ACTIVE_SHALLOW, rupture_df, "PGA")
psa = gmm.run_br13(
    gmm.constants.TectType.ACTIVE_SHALLOW,
    rupture_df,
    "pSA",
    periods=[0.1, 0.5, 1.0, 3.0],
    epistemic_branch=gmm.constants.EpistemicBranch.UPPER,
)

# Median PGA (g) per record, and pSA(1 s) as a pandas Series.
print(xr.ufuncs.exp(pga["ln_mean"]).to_series().describe())
print(psa["ln_mean"].sel(period=1.0).to_series().head())
# Everything as a flat dataframe, one row per (record, period).
print(psa.to_dataframe().head())
