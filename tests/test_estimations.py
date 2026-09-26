import oq_wrapper as oqw


def test_calc_z_for_model_falls_back_to_global_for_unsupported_region() -> None:
    """
    CY_14 does not define a "NewZealand" region mapping, only Global (None) and
    "Japan", so an unsupported region should fall back to the Global model
    rather than raising a KeyError.
    """
    vs30 = 400.0

    unsupported_region_result = oqw.estimations.calc_z_for_model(
        oqw.constants.GMM.CY_14, vs30, "NewZealand"
    )
    global_result = oqw.estimations.calc_z_for_model(
        oqw.constants.GMM.CY_14, vs30, None
    )

    assert unsupported_region_result == global_result
