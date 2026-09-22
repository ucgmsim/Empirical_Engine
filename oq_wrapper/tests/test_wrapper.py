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


def test_backbone_epistemic_uncertainty_handling() -> None:
    """
    Check that an error is thrown if a non-central branch is selected for a model 
    that does not have defined epistemic branch mappings for non-central branches.
    """
    rupture_df = pd.DataFrame(
        {"mag": [6.5], "rake": 180.0, "vs30": 400.0, "rrup": 100.0, "z1pt0": 1000.0}
    )

    with pytest.raises(ValueError, match=".*does not have defined epistemic branch mappings for non-central branches.*"):
        _ = oqw.run_gmm(
            oqw.constants.GMM.AS_16,
            oqw.constants.TectType.ACTIVE_SHALLOW,
            rupture_df,
            "Ds575",
            epistemic_branch=oqw.constants.EpistemicBranch.LOWER,
        )