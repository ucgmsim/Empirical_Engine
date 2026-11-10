"""Run a single ground motion model (Bradley 2013) for historic NZGMDB records.

Each model's ``run`` takes the tectonic type, a rupture/site dataframe
(OpenQuake column names, Z1.0 in km) and the IM, and returns an
`xarray.Dataset` with ``ln_mean``, ``std_total``, ``std_inter`` and
``std_intra`` (plus a ``period`` dimension for pSA). The columns a model needs
are the parameters of its inputs constructor, e.g.
``inspect.signature(gmm.Br_13.inputs[TectType.ACTIVE_SHALLOW])``.
"""

from pathlib import Path

import pandas as pd

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

pga = gmm.Br_13.run(gmm.constants.TectType.ACTIVE_SHALLOW, rupture_df, "PGA")
psa = gmm.Br_13.run(
    gmm.constants.TectType.ACTIVE_SHALLOW,
    rupture_df,
    "pSA",
    periods=[0.1, 0.5, 1.0, 3.0],
    epistemic_branch=gmm.constants.EpistemicBranch.UPPER,
)

# Print the first few records of the 1 s pSA median and total sigma.
print(psa["ln_mean"].sel(period=1.0).to_series().head())
