"""
Example of running an empirical GMM using the OpenQuake wrapper
Runs the Bradley 2013 model for a set of historic records

Note: The code produces a bunch of overflow warnings for
a cosh call, this is fine and can be ignored.
"""

from pathlib import Path

import pandas as pd

from empirical.util.classdef import GMM, TectType
from empirical.util.openquake_wrapper_vectorized import oq_run

### Load the data
record_data_ffp = Path(__file__).parent / "resources" / "record_data.csv"
record_df = pd.read_csv(record_data_ffp, index_col=0, dtype={"evid": str})

# Create the rupture data dataframe
# To determine which inputs are required by the model,
# check the model's implementation on openquake
# All their models are in this folder (or subfolders):
# https://github.com/gem/oq-engine/tree/master/openquake/hazardlib/gsim
# In the model's file, look for the "REQUIRES_SITES_PARAMETERS",
# "REQUIRES_RUPTURE_PARAMETERS", and "REQUIRES_DISTANCES" variables
rupture_df = record_df[
    ["mag", "dip", "rake", "z_tor", "r_rup", "r_jb", "r_x", "Vs30", "Z1.0"]
]
# Rename the columns to be in line what openquake expects
rupture_df = rupture_df.rename(
    columns={
        "z_tor": "ztor",
        "r_rup": "rrup",
        "r_jb": "rjb",
        "r_x": "rx",
        "Vs30": "vs30",
        "Z1.0": "z1pt0",
    }
)
rupture_df["vs30measured"] = True

# Drop any records with nan-values
nan_mask = rupture_df.isna().any(axis=1)
rupture_df = rupture_df[~nan_mask]
print(f"Dropped {nan_mask.sum()} records with nan-values")


pga_result = oq_run(
    GMM.Br_10,
    TectType.ACTIVE_SHALLOW,
    rupture_df,
    "PGA",
)
pga_result.index = rupture_df.index

pSA_results = oq_run(
    GMM.Br_10,
    TectType.ACTIVE_SHALLOW,
    rupture_df,
    "pSA",
    [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
)
pSA_results.index = rupture_df.index
