from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import oq_wrapper as oqw

benchmark_ffps = list(
    (Path(__file__).parent / "benchmark_data" / "data").rglob("*.parquet")
)
rupture_df = pd.read_parquet(
    Path(__file__).parent / "benchmark_data" / "nzgmdb_v4p3_rupture_df.parquet"
)


@pytest.mark.parametrize("benchmark_ffp", benchmark_ffps)
def test_gmm_benchmarks(benchmark_ffp: Path) -> None:
    im = benchmark_ffp.parent.name
    bench_df = pd.read_parquet(benchmark_ffp)

    cur_rupture_df = rupture_df.loc[bench_df.index.values]
    cur_rupture_df = cur_rupture_df.rename(columns=oqw.constants.NZGMDB_OQ_COL_MAPPING)

    cur_rupture_df["vs30measured"] = True
    cur_rupture_df["backarc"] = False

    # Convert Z1.0 to km
    cur_rupture_df["z1pt0"] = cur_rupture_df["z1pt0"] / 1000

    # Get pSA periods
    periods = (
        None
        if im != "pSA"
        else [
            float(col.rsplit("_", maxsplit=1)[0].strip("pSA_"))
            for col in bench_df.columns
            if col.endswith("mean")
        ]
    )

    # Get the GMM and TectType
    gmm_name, tect_type_name = benchmark_ffp.stem.split("TectType")
    gmm_name, tect_type_name = gmm_name.strip("_"), tect_type_name.strip("_")
    tect_type = oqw.constants.TectType[tect_type_name]

    # Single GMM model
    model = oqw.get_model_from_str(gmm_name)
    if isinstance(model, oqw.constants.GMM):
        result_df = oqw.run_gmm(model, tect_type, cur_rupture_df, im, periods=periods)
        assert_frame_equal(result_df, bench_df)
    # GMM Logic tree
    else:
        result_df = oqw.run_gmm_logic_tree(
            oqw.constants.GMMLogicTree[gmm_name],
            tect_type,
            cur_rupture_df,
            im,
            periods=periods,
        )

        assert_frame_equal(result_df, bench_df)
