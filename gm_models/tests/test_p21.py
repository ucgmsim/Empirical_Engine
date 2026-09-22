"""Layer 2: compare gm_models.p21 against oq_wrapper's existing P_21 benchmark fixtures.

This does not prove the model is *correct* (see the Rust `cargo test`
hazardlib-verification suite in `src-rust/p21.rs` for that, Layer 1) — it
proves this reimplementation is a safe drop-in replacement for
`oq_wrapper`'s own P_21, matching its wrapper-level output contract exactly
(column names, units, epistemic-branch wiring) against the same fixtures
`oq_wrapper`'s own test suite is checked against.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pytest import Metafunc

from gm_models.p21 import EpistemicBranch, TectType, run_p21

OQ_WRAPPER_TESTS_DIR = Path(__file__).parent.parent.parent / "oq_wrapper" / "tests"
BENCHMARK_DATA_DIR = OQ_WRAPPER_TESTS_DIR / "benchmark_data"
RUPTURE_PATH = BENCHMARK_DATA_DIR / "nzgmdb_v4p3_rupture_df.parquet"

# Matches oq_wrapper.constants.NZGMDB_OQ_COL_MAPPING, reimplemented standalone
# so this test suite doesn't need oq_wrapper as a dependency.
NZGMDB_OQ_COL_MAPPING = {
    "Vs30": "vs30",
    "Z1.0": "z1pt0",
    "Z2.5": "z2pt5",
    "r_rup": "rrup",
    "r_jb": "rjb",
    "r_x": "rx",
    "r_y": "ry",
    "r_hyp": "rhypo",
    "mag": "mag",
    "tect_class": "tect_class",
    "z_tor": "ztor",
    "z_bor": "zbot",
    "rake": "rake",
    "dip": "dip",
    "ev_depth": "hypo_depth",
}

# Discrepancy tolerance for comparing against the oq_wrapper benchmark
# parquet files. Looser than the exact-ish default of `assert_frame_equal`
# (oq_wrapper's own regression test) since this is a genuinely independent
# implementation, not the same code re-run.
RTOL = 1e-6


def pytest_generate_tests(metafunc: Metafunc) -> None:
    """Parametrize over every P_21 benchmark fixture (PGA/PGV/pSA x interface/slab x branch)."""
    if "benchmark_ffp" in metafunc.fixturenames:
        data_path = BENCHMARK_DATA_DIR / "data"
        files = sorted(data_path.rglob("P_21_TectType_*.parquet"))
        params = [pytest.param(f, id=f"{f.parent.name}/{f.stem}") for f in files]
        metafunc.parametrize("benchmark_ffp", params)


@pytest.fixture(scope="module")
def shared_rupture_df() -> pd.DataFrame:
    """Loads the heavy rupture dataframe once per module."""
    return pd.read_parquet(RUPTURE_PATH)


def test_p21_matches_oq_wrapper_benchmark(
    benchmark_ffp: Path, shared_rupture_df: pd.DataFrame
) -> None:
    im = benchmark_ffp.parent.name
    bench_df = pd.read_parquet(benchmark_ffp)

    cur_rupture_df = shared_rupture_df.loc[bench_df.index.values]
    cur_rupture_df = cur_rupture_df.rename(columns=NZGMDB_OQ_COL_MAPPING)
    cur_rupture_df["backarc"] = False

    periods = (
        None
        if im != "pSA"
        else [
            float(col.rsplit("_", maxsplit=1)[0].removeprefix("pSA_"))
            for col in bench_df.columns
            if col.endswith("mean")
        ]
    )

    gmm_name, tect_type_name = benchmark_ffp.stem.split("TectType")
    epistemic_branch = EpistemicBranch.CENTRAL
    if "EpistemicBranch" in tect_type_name:
        tect_type_name, epistemic_branch_name = tect_type_name.split("EpistemicBranch")
        epistemic_branch = EpistemicBranch[epistemic_branch_name.strip("_")]
    assert gmm_name.strip("_") == "P_21"
    tect_type = TectType[tect_type_name.strip("_")]

    result_df = run_p21(
        tect_type,
        cur_rupture_df,
        im,
        periods=periods,
        epistemic_branch=epistemic_branch,
    )

    assert list(result_df.columns) == list(bench_df.columns)
    np.testing.assert_allclose(
        result_df.to_numpy(),
        bench_df[result_df.columns].to_numpy(),
        rtol=RTOL,
        err_msg=f"gm_models.p21 output diverges from oq_wrapper benchmark for {benchmark_ffp}",
    )
