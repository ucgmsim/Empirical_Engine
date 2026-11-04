"""Layer 2: compare gm_models.p21 against oq_wrapper's existing P_21 benchmark fixtures.

This does not prove the model is *correct* (see the Rust `cargo test`
OpenQuake-verification suite in `src-rust/p21.rs` for that, Layer 1). It
checks that `run_p21` gives the same values as `oq_wrapper`'s P_21 (units,
epistemic-branch wiring, rupture-df handling) on the fixtures `oq_wrapper`'s
own test suite uses. The output format differs (an xarray Dataset rather than
`oq_wrapper`'s flat DataFrame), so each fixture column is matched to its
Dataset variable and period.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gm_models.constants import EpistemicBranch, TectType
from gm_models.p21 import run_p21

BENCHMARK_DATA_DIR = (
    Path(__file__).parents[2] / "oq_wrapper" / "tests" / "benchmark_data"
)
RUPTURE_PATH = BENCHMARK_DATA_DIR / "nzgmdb_v4p3_rupture_df.parquet"

# The P_21 subset of oq_wrapper.constants.NZGMDB_OQ_COL_MAPPING, copied so this
# test suite doesn't need oq_wrapper as a dependency.
NZGMDB_OQ_COL_MAPPING = {"Vs30": "vs30", "r_rup": "rrup", "ev_depth": "hypo_depth"}

# Discrepancy tolerance for comparing against the oq_wrapper benchmark
# parquet files. Looser than the exact-ish default of `assert_frame_equal`
# (oq_wrapper's own regression test) since this is a genuinely independent
# implementation, not the same code re-run.
RTOL = 1e-6

# oq_wrapper fixture columns: "PGA_mean", "pSA_0.1_std_Total", ...
OQ_WRAPPER_COLUMN = re.compile(
    r"(?:PGA|PGV|pSA_(?P<period>[\d.]+))_(?P<stat>mean|std_\w+)"
)


@pytest.fixture(scope="module")
def shared_rupture_df() -> pd.DataFrame:
    """Loads the heavy rupture dataframe once per module."""
    return pd.read_parquet(RUPTURE_PATH)


# Every P_21 benchmark fixture (PGA/PGV/pSA x interface/slab x branch).
@pytest.mark.parametrize(
    "benchmark_ffp",
    sorted((BENCHMARK_DATA_DIR / "data").rglob("P_21_TectType_*.parquet")),
    ids=lambda f: f"{f.parent.name}/{f.stem}",
)
def test_p21_matches_oq_wrapper_benchmark(
    benchmark_ffp: Path, shared_rupture_df: pd.DataFrame
) -> None:
    im = benchmark_ffp.parent.name
    bench_df = pd.read_parquet(benchmark_ffp)

    cur_rupture_df = shared_rupture_df.loc[bench_df.index.values]
    cur_rupture_df = cur_rupture_df.rename(columns=NZGMDB_OQ_COL_MAPPING)
    cur_rupture_df["backarc"] = False

    parsed: dict[str, re.Match[str]] = {}
    for col in bench_df.columns:
        m = OQ_WRAPPER_COLUMN.fullmatch(col)
        assert m is not None, f"unexpected oq_wrapper column '{col}'"
        parsed[col] = m
    periods = (
        None
        if im != "pSA"
        else [float(m["period"]) for col, m in parsed.items() if m["stat"] == "mean"]
    )

    gmm_name, tect_type_name = benchmark_ffp.stem.split("TectType")
    epistemic_branch = EpistemicBranch.CENTRAL
    if "EpistemicBranch" in tect_type_name:
        tect_type_name, epistemic_branch_name = tect_type_name.split("EpistemicBranch")
        epistemic_branch = EpistemicBranch[epistemic_branch_name.strip("_")]
    assert gmm_name.strip("_") == "P_21"
    tect_type = TectType[tect_type_name.strip("_")]

    ds = run_p21(
        tect_type,
        cur_rupture_df,
        im,
        periods=periods,
        epistemic_branch=epistemic_branch,
    )

    assert ds.indexes[bench_df.index.name].equals(bench_df.index)
    assert set(ds.data_vars) == {"ln_mean", "std_total", "std_inter", "std_intra"}
    for col, m in parsed.items():
        values = ds["ln_mean" if m["stat"] == "mean" else m["stat"].lower()]
        if m["period"] is not None:
            values = values.sel(period=float(m["period"]))
        np.testing.assert_allclose(
            values.to_numpy(),
            bench_df[col].to_numpy(),
            rtol=RTOL,
            err_msg=f"gm_models.p21 {col} diverges from oq_wrapper benchmark {benchmark_ffp}",
        )
