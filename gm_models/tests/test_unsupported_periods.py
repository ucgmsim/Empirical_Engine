"""SA periods above a model's coefficient table are errors, except for ZA_06 with `extrapolate`."""

import math

import pandas as pd
import pytest

import gm_models as gmm

RUPTURES = pd.DataFrame(
    {
        "mag": [7.0],
        "rrup": [50.0],
        "vs30": [400.0],
        "hypo_depth": [30.0],
        "backarc": [False],
    }
)


def test_period_above_table_raises_unless_za06_extrapolates() -> None:
    with pytest.raises(ValueError, match="period"):
        gmm.ZA_06.run(
            gmm.constants.TectType.SUBDUCTION_INTERFACE, RUPTURES, "pSA", periods=[6.0]
        )
    with pytest.raises(ValueError, match="period"):
        gmm.P_21.run(
            gmm.constants.TectType.SUBDUCTION_INTERFACE, RUPTURES, "pSA", periods=[12.0]
        )
    ds = gmm.ZA_06.run(
        gmm.constants.TectType.SUBDUCTION_INTERFACE,
        RUPTURES,
        "pSA",
        periods=[5.0, 6.0],
        extrapolate=True,
    )
    shift = (ds["ln_mean"].sel(period=6.0) - ds["ln_mean"].sel(period=5.0)).item()
    assert shift == pytest.approx(2 * math.log(5 / 6))
    assert ds.attrs["extrapolate"] == 1
