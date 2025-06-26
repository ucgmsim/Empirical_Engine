"""
Example of running an empirical GMM using the OpenQuake wrapper
Runs the Bradley 2013 model for a set of historic records

Note: The code produces a bunch of overflow warnings for
a cosh call, this is fine and can be ignored.
"""

from pathlib import Path

import pandas as pd

import oq_wrapper as oqw

### Load the data
record_data_ffp = (
    Path(__file__).parent.parent.parent
    / "tests"
    / "benchmark_data"
    / "nzgmdb_v4p3_rupture_df.parquet"
)
record_df = pd.read_parquet(record_data_ffp)

# Create the rupture data dataframe
# To determine which inputs are required by the model,
# check the model's implementation on OpenQuake
# All their models are in this folder (or subfolders):
# https://github.com/gem/oq-engine/tree/master/openquake/hazardlib/gsim
# In the model's file, look for the "REQUIRES_SITES_PARAMETERS",
# "REQUIRES_RUPTURE_PARAMETERS", and "REQUIRES_DISTANCES" variables
rupture_df = record_df[
    [
        "mag",
        "dip",
        "rake",
        "z_tor",
        "r_rup",
        "r_jb",
        "r_x",
        "r_y",
        "Vs30",
        "Z1.0",
        "Z2.5",
        "ev_depth",
        "z_bor",
    ]
]
# Rename the columns to be in line with what OpenQuake expects
rupture_df = rupture_df.rename(
    columns={
        "z_tor": "ztor",
        "z_bor": "zbot",
        "r_rup": "rrup",
        "r_jb": "rjb",
        "r_x": "rx",
        "r_y": "ry",
        "Vs30": "vs30",
        "Z1.0": "z1pt0",
        "Z2.5": "z2pt5",
        "ev_depth": "hypo_depth",
    }
)
rupture_df["vs30measured"] = True
rupture_df["z1pt0"] = rupture_df["z1pt0"] / 1000  # Convert Z1.0 to km
rupture_df["backarc"] = False

# Drop any records with nan-values
nan_mask = rupture_df.isna().any(axis=1)
rupture_df = rupture_df[~nan_mask]
print(f"Dropped {nan_mask.sum()} records with nan-values")

# Select GMM logic tree and run models
# Note this treats all records as active shallow!!
# Call run_gmm multiple times for different tectonic types
tect_type = oqw.constants.TectType.ACTIVE_SHALLOW
gmm_lt = oqw.constants.GMMLogicTree.NSHM2022
psa_results, ind_results = oqw.run_gmm_logic_tree(
    gmm_lt,
    tect_type,
    rupture_df,
    "pSA",
    periods=[0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
    return_ind_results=True,
)
