"""Generate benchmark values for testing purposes."""

import logging
import sys
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


def write_im_results(
    benchmark_data_output_dir: Path,
    im_results: pd.DataFrame,
    gmm: oqw.constants.GMM | oqw.constants.GMMLogicTree,
    tect_type: oqw.constants.TectType,
    epistemic_branch: oqw.constants.EpistemicBranch | None = None,
) -> None:
    im_results.index = cur_rupture_df.index

    file_name = f"{gmm.name}_TectType_{tect_type.name}"
    if epistemic_branch is not None:
        file_name += f"_EpistemicBranch_{epistemic_branch.name}"

    im_results.to_parquet(
        benchmark_data_output_dir / f"{file_name}.parquet",
    )


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
logger = logging.getLogger()
file_handler = logging.FileHandler(Path(__file__).parent / "benchmark_generation.log")
file_handler.setLevel(logging.INFO)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.WARNING)
logger.addHandler(console_handler)

logger.info("Starting benchmark generation script.")
gmms = list(oqw.constants.GMM) + list(oqw.constants.GMMLogicTree)
# Try to calculate "pSA", "PGA", and "PGV" for all combinations of GMM and TectType.
# If a given model (GMM+TectType) does not support a given IM, or requires input
# parameters that are not provided, we catch the Exception and continue.
for gmm in tqdm(gmms):
    # (V/H) ratio model, i.e. not relevant for benchmark
    if gmm is oqw.constants.GMM.GA_11:
        continue

    tect_types = (
        list(TECT_TYPE_MAPPING.keys())
        if isinstance(gmm, oqw.constants.GMMLogicTree)
        else oqw.OQ_MODEL_MAPPING[gmm].keys()
    )

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

            benchmark_data_output_dir = Path(__file__).parent / "data" / im
            benchmark_data_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                if isinstance(gmm, oqw.constants.GMMLogicTree):
                    im_results = oqw.run_gmm_logic_tree(
                        gmm,
                        tect_type,
                        cur_rupture_df,
                        im,
                        periods=cur_periods,
                    )
                    write_im_results(
                        benchmark_data_output_dir,
                        im_results,
                        gmm,
                        tect_type,
                    )
                else:
                    im_results = oqw.run_gmm(
                        gmm,
                        tect_type,
                        cur_rupture_df,
                        im,
                        periods=cur_periods,
                    )
                    write_im_results(
                        benchmark_data_output_dir,
                        im_results,
                        gmm,
                        tect_type,
                    )

                    if (
                        gmm in oqw.constants.GMM_EPISTEMIC_BRANCH_KWARGS_MAPPING
                        or gmm
                        in oqw.constants.GMM_EPISTEMIC_BRANCH_SIGMA_FACTOR_MAPPING
                    ):
                        im_results_lower = oqw.run_gmm(
                            gmm,
                            tect_type,
                            cur_rupture_df,
                            im,
                            periods=cur_periods,
                            epistemic_branch=oqw.constants.EpistemicBranch.LOWER,
                        )
                        write_im_results(
                            benchmark_data_output_dir,
                            im_results_lower,
                            gmm,
                            tect_type,
                            epistemic_branch=oqw.constants.EpistemicBranch.LOWER
                        )

                        im_results_upper = oqw.run_gmm(
                            gmm,
                            tect_type,
                            cur_rupture_df,
                            im,
                            periods=cur_periods,
                            epistemic_branch=oqw.constants.EpistemicBranch.UPPER,
                        )
                        write_im_results(
                            benchmark_data_output_dir,
                            im_results_upper,
                            gmm,
                            tect_type,  
                            epistemic_branch=oqw.constants.EpistemicBranch.UPPER
                        )

            except (ValueError,) as e:
                logger.error(
                    f"Error generating {im} for {gmm.name} and {tect_type.name}: {type(e).__name__}: {e}"
                )
                continue

            logger.info(
                f"Successfully generated {im} for {gmm.name} and {tect_type.name}."
            )


logger.info("Benchmark generation script completed.")
