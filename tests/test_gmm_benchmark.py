from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from pytest import Metafunc

import oq_wrapper as oqw

DATA_DIR = Path(__file__).parent / "benchmark_data"
RUPTURE_PATH = DATA_DIR / "nzgmdb_v4p3_rupture_df.parquet"


def pytest_generate_tests(metafunc: Metafunc) -> None:
    if "benchmark_ffp" in metafunc.fixturenames:
        data_path = DATA_DIR / "data"
        files = list(data_path.rglob("*.parquet"))

        params = []
        for f in files:
            test_id = f"{f.parent.name}/{f.stem}"
            params.append(pytest.param(f, id=test_id))

        metafunc.parametrize("benchmark_ffp", params)


@pytest.fixture(scope="module")
def shared_rupture_df() -> pd.DataFrame:
    """Loads the heavy rupture dataframe once per module."""
    return pd.read_parquet(RUPTURE_PATH)


def test_gmm_benchmarks(benchmark_ffp: Path, shared_rupture_df: pd.DataFrame) -> None:
    im = benchmark_ffp.parent.name
    bench_df = pd.read_parquet(benchmark_ffp)

    cur_rupture_df = shared_rupture_df.loc[bench_df.index.values]
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
