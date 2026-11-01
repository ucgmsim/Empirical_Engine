import pandas as pd
import pytest

import oq_wrapper as oqw


def test_run_gmm_z1pt0_handling() -> None:
    rupture_df = pd.DataFrame(
        {
            "mag": [6.5],
            "vs30": 400.0,
            "rrup": 100.0,
        }
    )

    # Result runs without throwing
    _ = oqw.run_gmm(
        oqw.constants.GMM.A_22, oqw.constants.TectType.ACTIVE_SHALLOW, rupture_df, "PGA"
    )

    rupture_df = pd.DataFrame(
        {"mag": [6.5], "rake": 180.0, "vs30": 400.0, "rrup": 100.0, "z1pt0": 1000.0}
    )

    # Succeeds because, although z1pt0 is "invalid", the model does not require z1pt0.
    _ = oqw.run_gmm(
        oqw.constants.GMM.A_22,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        rupture_df,
        "PGA",
    )

    # Result fails because of z1pt0 error
    with pytest.raises(ValueError, match=".*Z1.0 values.*"):
        _ = oqw.run_gmm(
            oqw.constants.GMM.AS_16,
            oqw.constants.TectType.ACTIVE_SHALLOW,
            rupture_df,
            "Ds575",
        )


def test_run_gmm_logic_tree_integer_periods() -> None:
    """
    Check that integer pSA periods do not crash run_gmm_logic_tree, and that
    extrapolated periods share the same column naming as in-range periods.
    """
    rupture_df = pd.DataFrame(
        {
            "mag": [6.0, 7.0],
            "dip": [60.0, 45.0],
            "rake": [0.0, 90.0],
            "ztor": [0.0, 1.0],
            "rrup": [10.0, 30.0],
            "rjb": [8.0, 25.0],
            "rx": [5.0, 10.0],
            "ry": [3.0, 4.0],
            "vs30": [400.0, 600.0],
            "z1pt0": [0.2, 0.1],
            "z2pt5": [1.0, 1.0],
            "hypo_depth": [8.0, 10.0],
            "zbot": [15.0, 15.0],
            "vs30measured": [True, True],
            "backarc": [False, False],
        }
    )

    result_df = oqw.run_gmm_logic_tree(
        oqw.constants.GMMLogicTree.NSHM2022,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        rupture_df,
        "pSA",
        periods=[1, 2],
    )
    assert "pSA_1.0_mean" in result_df.columns
    assert "pSA_2.0_mean" in result_df.columns

    # In-range and extrapolated periods should share the same float naming.
    columns = oqw.run_gmm(
        oqw.constants.GMM.Br_13,
        oqw.constants.TectType.ACTIVE_SHALLOW,
        rupture_df,
        "pSA",
        periods=[1, 15],
    ).columns
    assert "pSA_1.0_mean" in columns
    assert "pSA_15.0_mean" in columns


def test_backbone_epistemic_uncertainty_handling() -> None:
    """
    Check that an error is thrown if a non-central branch is selected for a model
    that does not have defined epistemic branch mappings for non-central branches.
    """
    rupture_df = pd.DataFrame(
        {"mag": [6.5], "rake": 180.0, "vs30": 400.0, "rrup": 100.0, "z1pt0": 1000.0}
    )

    with pytest.raises(
        ValueError,
        match=".*does not have defined epistemic branch mappings for non-central branches.*",
    ):
        _ = oqw.run_gmm(
            oqw.constants.GMM.AS_16,
            oqw.constants.TectType.ACTIVE_SHALLOW,
            rupture_df,
            "Ds575",
            epistemic_branch=oqw.constants.EpistemicBranch.LOWER,
        )
