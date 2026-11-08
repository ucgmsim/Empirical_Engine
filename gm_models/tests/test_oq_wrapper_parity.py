"""Layer 2: compare gm_models against oq_wrapper's benchmark fixtures.

This does not prove a model is *correct* (see the Rust `cargo test`
OpenQuake-verification suites in `src-rust/` for that, Layer 1). It checks
that each `run_*` gives the same values as `oq_wrapper.run_gmm` (units,
epistemic-branch wiring, rupture-df handling) on the fixtures `oq_wrapper`'s
own test suite uses. The output format differs (an xarray Dataset rather than
`oq_wrapper`'s flat DataFrame), so each fixture column is matched to its
Dataset variable and period.
"""

import re
from collections.abc import Callable
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from gm_models.a18 import run_a18
from gm_models.a22 import run_a22
from gm_models.ag20 import run_ag20, run_ag20_nz
from gm_models.as16 import run_as16
from gm_models.ask14 import run_ask14
from gm_models.ba18 import run_ba18
from gm_models.bch16 import run_bch16
from gm_models.br13 import run_br13
from gm_models.bssa14 import run_bssa14
from gm_models.cb10 import run_cb10
from gm_models.cb14 import run_cb14
from gm_models.constants import EpistemicBranch, TectType
from gm_models.cy14 import run_cy14
from gm_models.k20 import run_k20, run_k20_nz
from gm_models.p20 import run_p20
from gm_models.p21 import run_p21
from gm_models.s22 import run_s22
from gm_models.za06 import run_za06

BENCHMARK_DATA_DIR = (
    Path(__file__).parents[2] / "oq_wrapper" / "tests" / "benchmark_data"
)
RUPTURE_PATH = BENCHMARK_DATA_DIR / "nzgmdb_v4p3_rupture_df.parquet"

# oq_wrapper GMM name -> gm_models runner. Each takes (tect_type, rupture_df,
# im), plus periods= for pSA, frequencies= for EAS and, for models with
# branches, epistemic_branch=.
MODELS: dict[str, Callable[..., xr.Dataset]] = {
    "K_20": run_k20,
    "K_20_NZ": run_k20_nz,
    "P_20": run_p20,
    "P_21": run_p21,
    "AG_20": run_ag20,
    "AG_20_NZ": run_ag20_nz,
    "A_22": run_a22,
    "S_22": run_s22,
    "ASK_14": run_ask14,
    "BSSA_14": run_bssa14,
    "CY_14": run_cy14,
    "Br_13": run_br13,
    "CB_14": run_cb14,
    "CB_10": run_cb10,
    # The fixtures include 6 to 10 s columns, which oq_wrapper extrapolates.
    "ZA_06": partial(run_za06, extrapolate=True),
    "A_18": run_a18,
    "BCH_16": run_bch16,
    "AS_16": run_as16,
    "BA_18": run_ba18,
}

# Copy of oq_wrapper.constants.NZGMDB_OQ_COL_MAPPING, so this test suite
# doesn't need oq_wrapper as a dependency.
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

# oq_wrapper fixture columns: "PGA_mean", "pSA_0.1_std_Total", "EAS_5.0_mean",
# "Ds575_std_Inter", ...
OQ_WRAPPER_COLUMN = re.compile(
    r"(?:PGA|PGV|CAV|Ds575|Ds595|Ds2080|(?:pSA|EAS)_(?P<value>[\d.]+))"
    r"_(?P<stat>mean|std_\w+)"
)
# Spectral IMs: the run_* keyword and Dataset dimension of their column values.
SPECTRAL = {"pSA": ("periods", "period"), "EAS": ("frequencies", "frequency")}


@pytest.fixture(scope="module")
def shared_rupture_df() -> pd.DataFrame:
    """Loads the heavy rupture dataframe once per module."""
    return pd.read_parquet(RUPTURE_PATH)


# Every benchmark fixture of a model in MODELS (IM x tect type x branch).
@pytest.mark.parametrize(
    "benchmark_ffp",
    sorted(
        f
        for f in (BENCHMARK_DATA_DIR / "data").rglob("*.parquet")
        if f.stem.split("_TectType")[0] in MODELS
    ),
    ids=lambda f: f"{f.parent.name}/{f.stem}",
)
def test_matches_oq_wrapper_benchmark(
    benchmark_ffp: Path, shared_rupture_df: pd.DataFrame
) -> None:
    im = benchmark_ffp.parent.name
    bench_df = pd.read_parquet(benchmark_ffp)

    # Same preprocessing as oq_wrapper's tests/test_gmm_benchmark.py.
    cur_rupture_df = shared_rupture_df.loc[bench_df.index.values]
    cur_rupture_df = cur_rupture_df.rename(columns=NZGMDB_OQ_COL_MAPPING)
    cur_rupture_df["vs30measured"] = True
    cur_rupture_df["backarc"] = False
    cur_rupture_df["z1pt0"] = cur_rupture_df["z1pt0"] / 1000

    parsed: dict[str, re.Match[str]] = {}
    for col in bench_df.columns:
        m = OQ_WRAPPER_COLUMN.fullmatch(col)
        assert m is not None, f"unexpected oq_wrapper column '{col}'"
        parsed[col] = m
    gmm_name, tect_type_name = benchmark_ffp.stem.split("_TectType_")
    kwargs = {}
    if im in SPECTRAL:
        kwargs[SPECTRAL[im][0]] = [
            float(m["value"]) for m in parsed.values() if m["stat"] == "mean"
        ]
    if "EpistemicBranch" in tect_type_name:
        tect_type_name, epistemic_branch_name = tect_type_name.split(
            "_EpistemicBranch_"
        )
        kwargs["epistemic_branch"] = EpistemicBranch[epistemic_branch_name]

    ds = MODELS[gmm_name](TectType[tect_type_name], cur_rupture_df, im, **kwargs)

    assert ds.indexes[bench_df.index.name].equals(bench_df.index)
    assert set(ds.data_vars) == {"ln_mean", "std_total", "std_inter", "std_intra"}
    for col, m in parsed.items():
        values = ds["ln_mean" if m["stat"] == "mean" else m["stat"].lower()]
        if m["value"] is not None:
            values = values.sel({SPECTRAL[im][1]: float(m["value"])})
        np.testing.assert_allclose(
            values.to_numpy(),
            bench_df[col].to_numpy(),
            rtol=RTOL,
            err_msg=f"{gmm_name} {col} diverges from oq_wrapper benchmark {benchmark_ffp}",
        )
