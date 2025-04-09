"""Generate benchmark values for testing purposes."""

import logging
from pathlib import Path

import pandas as pd

from empirical.util.classdef import GMM, TectType
from empirical.util.openquake_wrapper_vectorized import oq_run

PERIODS = [
    0.01,
    0.02,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.12,
    0.15,
    0.17,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.75,
    0.8,
    0.9,
    1.0,
    1.25,
    1.2,
    1.5,
    2.0,
    2.5,
    3.0,
    4.0,
    5.0,
    6.0,
    7.5,
    10.0,
]

record_df = pd.read_csv(
    Path(__file__).parent
    / "data"
    / "input_data"
    / "ground_motion_im_table_rotd50_flat.csv"
)

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

# Configure logging to capture errors and progress
logging.basicConfig(
    filename=Path(__file__).parent / "benchmark_generation.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logging.info("Starting benchmark generation script.")
# Iterate over all combinations of GMM and TectType
for gmm in list(GMM):
    for tect_type in list(TectType):

        for im in ["pSA", "PGA", "PGV"]:
            temp_periods = PERIODS if im == "pSA" else None
            try:
                im_results = oq_run(
                    gmm,
                    tect_type,
                    rupture_df,
                    im,
                    temp_periods,
                )

            except Exception as e:
                logging.error(
                    f"Error generating {im} for {gmm.name} and {tect_type.name}: {e}"
                )
                print(f"Error generating {im} for {gmm.name} and {tect_type.name}: {e}")
                continue

            benchmark_data_output_dir = (
                Path(__file__).parent / "data" / "benchmark_output" / im
            )
            benchmark_data_output_dir.mkdir(parents=True, exist_ok=True)

            im_results.index = rupture_df.index
            im_results.to_parquet(
                benchmark_data_output_dir / f"{gmm.name}_{tect_type.name}.parquet",
                index=False,
            )
            logging.info(
                f"Successfully generated {im} for {gmm.name} and {tect_type.name}."
            )

logging.info("Benchmark generation script completed.")
