"""Generate benchmark values for testing purposes."""

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import oq_wrapper as oqw

TECT_TYPE_MAPPING = {
    oqw.constants.TectType.ACTIVE_SHALLOW: "Crustal",
    oqw.constants.TectType.SUBDUCTION_SLAB: "Slab",
    oqw.constants.TectType.SUBDUCTION_INTERFACE: "Interface",
    oqw.constants.TectType.VOLCANIC: "Crustal",
}
N_RECORDS = 5000
RANDOM_SEED = 42

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
    1.2,
    1.25,
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

rupture_df = pd.read_parquet(Path(__file__).parent / "nzgmdb_v4p3_rupture_df.parquet")

# Rename the columns to be in line with what OpenQuake expects
rupture_df = rupture_df.rename(columns=oqw.constants.NZGMDB_OQ_COL_MAPPING)
rupture_df["vs30measured"] = True
rupture_df["backarc"] = False

# Convert Z1.0 to km
rupture_df["z1pt0"] = rupture_df["z1pt0"] / 1000

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
gmms = list(oqw.constants.GMM) + list(oqw.constants.GMMLogicTree)
# Try to calculate "pSA", "PGA", and "PGV" for all combinations of GMM and TectType.
# If a given model (GMM+TectType) does not support a given IM, or requires input
# parameters that are not provided, we catch the Exception and continue.
for gmm in tqdm(gmms):

    # (V/H) ratio model, i.e. not relevant for benchmark
    if gmm is oqw.constants.GMM.GA_11:
        continue

    tect_types = list(TECT_TYPE_MAPPING.keys()) if isinstance(gmm, oqw.constants.GMMLogicTree) else oqw.OQ_MODEL_MAPPING[gmm].keys()

    # Iterate over all tectonic types supported by the GMM
    for tect_type in tect_types:
        for im in ["pSA", "PGA", "PGV"]:
            cur_periods = PERIODS if im == "pSA" else None

            # Only use records matching the current tectonic type
            cur_rupture_df = rupture_df.loc[
                rupture_df.tect_class == TECT_TYPE_MAPPING[tect_type]
            ]

            # Use a random subset
            if len(cur_rupture_df) > N_RECORDS:
                cur_rupture_df = cur_rupture_df.sample(
                    N_RECORDS, random_state=RANDOM_SEED
                )

            try:
                if isinstance(gmm, oqw.constants.GMMLogicTree):
                    im_results = oqw.run_gmm_lt(
                        gmm,
                        tect_type,
                        cur_rupture_df,
                        im,
                        periods=cur_periods,
                    )
                else:
                    im_results = oqw.run_gmm(
                        gmm,
                        tect_type,
                        cur_rupture_df,
                        im,
                        periods=cur_periods,
                    )
            except (
                ValueError,
            ) as e:
                logging.error(
                    f"Error generating {im} for {gmm.name} and {tect_type.name}: {type(e).__name__}: {e}"
                )
                continue

            benchmark_data_output_dir = (
                Path(__file__).parent / "data" / im
            )
            benchmark_data_output_dir.mkdir(parents=True, exist_ok=True)

            im_results.index = cur_rupture_df.index
            im_results.to_parquet(
                benchmark_data_output_dir / f"{gmm.name}_TectType_{tect_type.name}.parquet",
            )
            logging.info(
                f"Successfully generated {im} for {gmm.name} and {tect_type.name}."
            )

logging.info("Benchmark generation script completed.")
