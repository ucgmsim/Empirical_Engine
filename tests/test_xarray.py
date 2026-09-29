import numpy as np
import pandas as pd
import pytest
import xarray as xr

import oq_wrapper as oqw
from oq_wrapper import xarray as oqwx


def _site_inputs(num_stations: int = 4, num_models: int = 2) -> xr.Dataset:
    rng = np.random.default_rng(seed=0)
    return xr.Dataset(
        data_vars=dict(
            rrup=("station", np.linspace(1, 100, num=num_stations)),
            vs30=(
                ("model", "station"),
                rng.uniform(500, 1500, size=(num_models, num_stations)),
            ),
            mag=8.3,
        ),
        coords=dict(
            station=[f"a_{i}" for i in range(1, num_stations + 1)],
            model=[f"m_{i}" for i in range(num_models)],
        ),
    )


def test_run_gmm_xarray_plain_im_has_no_extra_dimension() -> None:
    """PGA has no period/frequency suffix, exercising _pack_dataset's default branch."""
    inputs = _site_inputs()
    result = oqwx.run_gmm_xarray(
        oqw.constants.GMM.A_22,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        inputs,
        "PGA",
    )

    assert sorted(result.data_vars) == ["mean", "std_Inter", "std_Intra", "std_Total"]
    assert sorted(result.coords) == ["model", "station"]
    assert result.attrs["intensity_measure"] == "PGA"

    value = result["mean"].sel(station="a_2", model="m_1").item()
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_run_gmm_xarray_psa_adds_period_dimension() -> None:
    inputs = _site_inputs()
    result = oqwx.run_gmm_xarray(
        oqw.constants.GMM.A_22,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        inputs,
        "pSA",
        periods=[1.0, 3.0],
    )

    assert sorted(result.data_vars) == ["mean", "std_Inter", "std_Intra", "std_Total"]
    assert sorted(result.coords) == ["model", "period", "station"]
    assert result.attrs["intensity_measure"] == "pSA"
    assert result["period"].values.tolist() == [1.0, 3.0]

    value = (
        result["mean"]
        .sel(station="a_2", model="m_1")
        .sel(period=1.0, method="nearest")
        .item()
    )
    assert isinstance(value, float)
    assert np.isfinite(value)


def test_run_gmm_logic_tree_xarray_broadcasts_dimensions() -> None:
    rng = np.random.default_rng(seed=0)
    num_stations = 3
    inputs = xr.Dataset(
        dict(
            rrup=("station", np.linspace(1, 100, num=num_stations)),
            rjb=("station", rng.uniform(1, 100, size=(num_stations,))),
            rx=("station", rng.uniform(1, 100, size=(num_stations,))),
            ry=("station", rng.uniform(1, 100, size=(num_stations,))),
            hyp=("station", rng.uniform(1, 100, size=(num_stations,))),
            epi=("station", rng.uniform(1, 100, size=(num_stations,))),
            ztor=0.0,
            dip=60.0,
            rake=15.0,
            hypo_depth=5.0,
            zbot=10.0,
            vs30measured=False,
            vs30=("station", rng.uniform(500, 1500, size=(num_stations,))),
            z1pt0=("station", rng.uniform(0.5, 1.5, size=(num_stations,))),
            z2pt5=("station", rng.uniform(0.5, 1.5, size=(num_stations,))),
            mag=6.5,
        ),
        coords=dict(station=[f"a_{i}" for i in range(1, num_stations + 1)]),
    )

    result = oqwx.run_gmm_logic_tree_xarray(
        oqw.constants.GMMLogicTree.NSHM2022,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        inputs,
        "pSA",
        periods=[1.0],
    )

    assert sorted(result.data_vars) == ["mean", "std_Total"]
    assert sorted(result.coords) == ["period", "station"]
    assert result.attrs["intensity_measure"] == "pSA"

    value = result["mean"].sel(station="a_2").sel(period=1.0, method="nearest").item()
    assert isinstance(value, float)
    assert np.isfinite(value)


# No GMM in oq_wrapper.constants.GMM currently supports the "EAS" intensity
# measure or produces a column name _pack_dataset cannot parse, so those two
# branches are exercised directly against realistic column names instead of
# through run_gmm_xarray/run_gmm_logic_tree_xarray.
def test_pack_dataset_extracts_eas_frequency_dimension() -> None:
    results = pd.DataFrame(
        {
            "EAS_1.0_mean": [-2.1, -1.9],
            "EAS_1.0_std_Total": [0.5, 0.6],
            "EAS_2.0_mean": [-2.5, -2.3],
            "EAS_2.0_std_Total": [0.55, 0.65],
        },
        index=pd.Index(["a", "b"], name="station"),
    )

    dset = oqwx._pack_dataset(results)

    assert sorted(dset.data_vars) == ["mean", "std_Total"]
    assert dset.attrs["intensity_measure"] == "EAS"
    assert sorted(dset["frequency"].values.tolist()) == [1.0, 2.0]
    assert dset["mean"].sel(station="a", frequency=1.0).item() == -2.1
    assert dset["std_Total"].sel(station="b", frequency=2.0).item() == 0.65


def test_pack_dataset_raises_for_unparseable_column_name() -> None:
    results = pd.DataFrame({"nostatisticsuffix": [1.0, 2.0]})

    with pytest.raises(ValueError, match="Could not extract dimensions"):
        oqwx._pack_dataset(results)
