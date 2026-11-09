"""Input handling of the AS_16 and BA_18 wrappers (no `oq_wrapper` fixtures exist for them)."""

import numpy as np
import pandas as pd
import pytest

import gm_models as gmm

RUPTURES = pd.DataFrame(
    {
        "mag": [5.5, 7.0],
        "rake": [90.0, -90.0],
        "ztor": [1.0, 3.0],
        "rrup": [10.0, 50.0],
        "vs30": [300.0, 600.0],
        "z1pt0": [0.4, 0.1],
    }
)


def test_outputs_have_expected_shapes_and_are_finite() -> None:
    durations = gmm.run_as16(gmm.constants.TectType.ACTIVE_SHALLOW, RUPTURES, "Ds595")
    assert durations["ln_mean"].shape == (2,)
    eas = gmm.run_ba18(
        gmm.constants.TectType.ACTIVE_SHALLOW, RUPTURES, "EAS", frequencies=[0.5, 5.0]
    )
    assert eas["ln_mean"].shape == (2, 2)
    assert np.isfinite(eas["ln_mean"]).all()


@pytest.mark.parametrize("run", [gmm.run_as16, gmm.run_ba18])
def test_rejects_other_tect_types_and_z1pt0_in_metres(run) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="tectonic type"):
        run(gmm.constants.TectType.SUBDUCTION_SLAB, RUPTURES, "EAS")
    with pytest.raises(ValueError, match="Z1.0"):
        run(gmm.constants.TectType.ACTIVE_SHALLOW, RUPTURES.assign(z1pt0=400.0), "EAS")


def test_ba18_rejects_out_of_range_frequency_and_other_ims() -> None:
    with pytest.raises(ValueError, match="not supported"):
        gmm.run_ba18(
            gmm.constants.TectType.ACTIVE_SHALLOW, RUPTURES, "EAS", frequencies=[0.05]
        )
    with pytest.raises(ValueError, match="EAS"):
        gmm.run_ba18(gmm.constants.TectType.ACTIVE_SHALLOW, RUPTURES, "PGA")
