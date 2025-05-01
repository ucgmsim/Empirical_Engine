from pathlib import Path

import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from empirical.util.classdef import GMM, TectType
from empirical.util.openquake_wrapper_vectorized import oq_run
from empirical.util.empirical import NZGMDB_OQ_COL_MAPPING

benchmark_files = list(
    (Path(__file__).parent / "benchmark_data" / "data").rglob("*.parquet")
)
rupture_df = pd.read_parquet(
    Path(__file__).parent / "benchmark_data" / "nzgmdb_v4p3_rupture_df.parquet"
)


@pytest.mark.parametrize("benchmark_ffp", benchmark_files)
def test_gmm_benchmarks(benchmark_ffp: Path):
    im = benchmark_ffp.parent.name
    bench_df = pd.read_parquet(benchmark_ffp)

    cur_rupture_df = rupture_df.loc[bench_df.index.values]
    cur_rupture_df = cur_rupture_df.rename(columns=NZGMDB_OQ_COL_MAPPING)

    cur_rupture_df["vs30measured"] = True
    cur_rupture_df["backarc"] = False

    # Convert Z1.0 to km
    cur_rupture_df["z1pt0"] = cur_rupture_df["z1pt0"] / 1000

    # Get the GMM and TectType
    gmm_name, tect_type_name = benchmark_ffp.stem.split("TectType")
    gmm_name, tect_type_name = gmm_name.strip("_"), tect_type_name.strip("_")
    gmm, tect_type = GMM[gmm_name], TectType[tect_type_name]

    periods = (
        None
        if im != "pSA"
        else [
            float(col.rsplit("_", maxsplit=1)[0].strip("pSA_"))
            for col in bench_df.columns
            if col.endswith("mean")
        ]
    )
    result_df = oq_run(gmm, tect_type, cur_rupture_df, im, periods=periods)
    result_df.index = cur_rupture_df.index

    assert_frame_equal(result_df, bench_df)


# if __name__ == "__main__":
#     # Run the test function
#     for cur_ffp in benchmark_files:
#         test_gmm_benchmarks(cur_ffp)
#     # test_gmm_benchmarks(benchmark_files)
