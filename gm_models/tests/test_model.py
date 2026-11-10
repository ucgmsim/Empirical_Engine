"""Argument handling shared by every model (`gm_models.Model.run`)."""

import numpy as np
import pandas as pd
import pytest

import gm_models as gmm

INTERFACE = gmm.constants.TectType.SUBDUCTION_INTERFACE
RUPTURES = pd.DataFrame(
    {"mag": [7.0, 8.0], "rrup": [50.0, 100.0], "vs30": [400.0, 760.0]}
)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"epistemic_branch": gmm.constants.EpistemicBranch.UPPER},
        {"modified_sigma": False},
        {"extrapolate": True},
    ],
)
def test_rejects_arguments_the_model_does_not_take(kwargs: dict) -> None:
    with pytest.raises(ValueError, match="has no"):
        gmm.A_18.run(INTERFACE, RUPTURES, "PGA", **kwargs)


def test_reports_missing_columns_and_unsupported_tect_types() -> None:
    with pytest.raises(ValueError, match="missing required columns: \\['vs30'\\]"):
        gmm.A_18.run(INTERFACE, RUPTURES.drop(columns="vs30"), "PGA")
    with pytest.raises(ValueError, match="tectonic type"):
        gmm.A_18.run(gmm.constants.TectType.ACTIVE_SHALLOW, RUPTURES, "PGA")


def test_periods_may_be_a_numpy_array() -> None:
    ds = gmm.A_18.run(INTERFACE, RUPTURES, "pSA", np.array([0.1, 1.0]))
    assert ds["ln_mean"].shape == (2, 2)
