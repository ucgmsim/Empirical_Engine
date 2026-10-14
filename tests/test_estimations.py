from types import SimpleNamespace

import numpy as np
import pytest

from oq_wrapper import estimations

# Increasing Vs30 values spanning soft soil to hard rock (m/s)
VS30 = np.array([150.0, 250.0, 400.0, 760.0, 1100.0, 1500.0])


def test_circ_mean_wraps_around_north() -> None:
    # 350 and 10 degrees average to 0, not 180 as a naive mean would give.
    result = estimations.circ_mean(
        np.radians([350.0, 10.0]), weights=np.array([0.5, 0.5])
    )
    assert isinstance(result, float)
    assert result == pytest.approx(0.0, abs=1e-12)


def test_circ_mean_respects_weights() -> None:
    samples = np.radians([0.0, 90.0])
    assert np.degrees(
        estimations.circ_mean(samples, weights=np.array([1.0, 1.0]))
    ) == pytest.approx(45.0)
    # All the weight on the second sample returns that sample.
    assert np.degrees(
        estimations.circ_mean(samples, weights=np.array([0.0, 1.0]))
    ) == pytest.approx(90.0)


def test_calculate_avg_multi_plane_properties() -> None:
    # Only strike, dip and width are read from the planes, so a stand-in object
    # is enough (source-modelling is not a runtime dependency).
    planes = [
        SimpleNamespace(strike=350.0, dip=30.0, width=10.0),
        SimpleNamespace(strike=10.0, dip=70.0, width=20.0),
    ]
    avg_strike, avg_dip, avg_rake, avg_width = (
        estimations.calculate_avg_multi_plane_properties(
            planes,  # ty: ignore[invalid-argument-type]  # duck-typed Plane stand-ins
            plane_avg_rake=[170.0, -170.0],
            plane_areas=[1.0, 3.0],
        )
    )
    # Weights are 0.25 and 0.75 from the areas.
    offset = np.degrees(np.arctan(0.5 * np.tan(np.radians(10.0))))
    # Strike is averaged across north and returned in [0, 360).
    assert avg_strike == pytest.approx(offset)
    # Rake is averaged across +/-180 and returned in [-180, 180).
    assert avg_rake == pytest.approx(-180.0 + offset)
    assert avg_dip == pytest.approx(0.25 * 30.0 + 0.75 * 70.0)
    assert avg_width == pytest.approx(0.25 * 10.0 + 0.75 * 20.0)


def test_calculate_avg_multi_plane_properties_strike_in_0_360() -> None:
    planes = [
        SimpleNamespace(strike=340.0, dip=45.0, width=5.0),
        SimpleNamespace(strike=350.0, dip=45.0, width=5.0),
    ]
    avg_strike, avg_dip, avg_rake, avg_width = (
        estimations.calculate_avg_multi_plane_properties(
            planes,  # ty: ignore[invalid-argument-type]  # duck-typed Plane stand-ins
            plane_avg_rake=[90.0, 90.0],
            plane_areas=[2.0, 2.0],
        )
    )
    # circ_mean gives -15 degrees, which must be mapped back to 345.
    assert avg_strike == pytest.approx(345.0)
    assert avg_rake == pytest.approx(90.0)
    assert avg_dip == pytest.approx(45.0)
    assert avg_width == pytest.approx(5.0)


@pytest.mark.parametrize("region", ["Cascadia", "Japan", "NewZealand", "Taiwan"])
def test_kuehn_20_calc_z_decreases_with_vs30(region: str) -> None:
    result = estimations.kuehn_20_calc_z(VS30, region)
    assert result.shape == VS30.shape
    assert np.all(np.isfinite(result))
    assert np.all(np.diff(result) < 0)


def test_kuehn_20_calc_z_unsupported_region() -> None:
    with pytest.raises(KeyError, match="Region California not supported"):
        estimations.kuehn_20_calc_z(VS30, "California")


@pytest.mark.parametrize(
    "func",
    [estimations.chiou_young_14_calc_z1p0, estimations.mod_chiou_young_14_calc_z1p0],
)
def test_chiou_young_14_z1p0_global_reference_rock(func) -> None:  # noqa: ANN001
    # The global relation is anchored to 1 m (0.001 km) at Vs30 = 1360 m/s.
    assert func(1360.0) == pytest.approx(0.001)
    result = func(VS30)
    assert result.shape == VS30.shape
    assert np.all(result > 0)
    assert np.all(np.diff(result) < 0)


def test_mod_chiou_young_14_z1p0_shallower_than_original() -> None:
    # The modified global coefficient (610 instead of 570.94) gives shallower
    # basins for any site softer than the 1360 m/s reference.
    soft = VS30[VS30 < 1360.0]
    assert np.all(
        estimations.mod_chiou_young_14_calc_z1p0(soft)
        < estimations.chiou_young_14_calc_z1p0(soft)
    )


@pytest.mark.parametrize(
    "func",
    [
        estimations.chiou_young_14_calc_z1p0,
        estimations.mod_chiou_young_14_calc_z1p0,
        estimations.campbell_bozorgina_14_calc_z2p5,
    ],
)
def test_japan_region_uses_separate_relation(func) -> None:  # noqa: ANN001
    japan = func(VS30, region="Japan")
    global_ = func(VS30)
    assert japan.shape == VS30.shape
    assert np.all(japan > 0)
    assert np.all(np.diff(japan) < 0)
    assert not np.allclose(japan, global_)
    # Any region other than Japan falls back to the global relation.
    np.testing.assert_allclose(func(VS30, region="NewZealand"), global_)


def test_campbell_bozorgina_14_z2p5_global_decreases_with_vs30() -> None:
    result = estimations.campbell_bozorgina_14_calc_z2p5(VS30)
    assert result.shape == VS30.shape
    assert np.all(result > 0)
    assert np.all(np.diff(result) < 0)


def test_chiou_young_08_calc_z1p0() -> None:
    result = estimations.chiou_young_08_calc_z1p0(VS30)
    assert np.all(np.diff(result) < 0)
    # For soft sites the 378.7^8 term dominates: exp(28.5 - 3.82 ln 378.7) m.
    soft_limit = np.exp(28.5 - 3.82 * np.log(378.7)) / 1000
    assert estimations.chiou_young_08_calc_z1p0(50.0) == pytest.approx(
        soft_limit, rel=1e-6
    )
    # For stiff sites the vs30^8 term dominates: exp(28.5 - 3.82 ln vs30) m.
    assert estimations.chiou_young_08_calc_z1p0(5000.0) == pytest.approx(
        np.exp(28.5 - 3.82 * np.log(5000.0)) / 1000, rel=1e-6
    )


def test_chiou_young_08_calc_z2p5() -> None:
    z1p0 = np.array([0.1, 0.5])
    z1p5 = np.array([0.2, 1.0])
    np.testing.assert_allclose(
        estimations.chiou_young_08_calc_z2p5(z1p0=z1p0), 0.519 + 3.595 * z1p0
    )
    np.testing.assert_allclose(
        estimations.chiou_young_08_calc_z2p5(z1p5=z1p5), 0.636 + 1.549 * z1p5
    )
    # Z1.5 takes priority over Z1.0 when both are given.
    np.testing.assert_allclose(
        estimations.chiou_young_08_calc_z2p5(z1p0=z1p0, z1p5=z1p5),
        0.636 + 1.549 * z1p5,
    )
    with pytest.raises(ValueError, match="no z2p5 able to be estimated"):
        estimations.chiou_young_08_calc_z2p5()


@pytest.mark.parametrize(
    "region, soft_vs30, stiff_vs30, ln_range",
    [
        # ln(Z2.5) is clipped to [7.6, 8.52] for Cascadia, [4.1, 7.3] for Japan
        ("Cascadia", [100.0, 200.0], [600.0, 1500.0], 8.52 - 7.6),
        ("Japan", [100.0, 170.0], [850.0, 1500.0], 7.3 - 4.1),
    ],
)
def test_abrahamson_gulerce_20_calc_z2p5_clipping(
    region: str, soft_vs30: list[float], stiff_vs30: list[float], ln_range: float
) -> None:
    soft = estimations.abrahamson_gulerce_20_calc_z2p5(np.array(soft_vs30), region)
    stiff = estimations.abrahamson_gulerce_20_calc_z2p5(np.array(stiff_vs30), region)
    # Outside the transition zone the depth is constant.
    assert soft[0] == pytest.approx(soft[1])
    assert stiff[0] == pytest.approx(stiff[1])
    # The ratio between the deepest and shallowest basin is set by the clip.
    assert np.log(soft[0] / stiff[0]) == pytest.approx(ln_range)
    # Inside the transition zone the depth decreases with Vs30.
    mid = estimations.abrahamson_gulerce_20_calc_z2p5(
        np.array([250.0, 350.0, 450.0]), region
    )
    assert np.all(np.diff(mid) < 0)
    assert np.all((mid <= soft[0]) & (mid >= stiff[0]))


@pytest.mark.parametrize(
    "func", [estimations.abrahamson_gulerce_20_calc_z2p5, estimations.parker_20_calc_z2p5]
)
def test_subduction_z2p5_unsupported_region(func) -> None:  # noqa: ANN001
    with pytest.raises(ValueError, match="Does not support region NewZealand"):
        func(VS30, "NewZealand")


@pytest.mark.parametrize(
    "region, theta1, vmu",
    [("Japan", -0.8, 500.0), ("Cascadia", -0.42, 200.0)],
)
def test_parker_20_calc_z2p5(region: str, theta1: float, vmu: float) -> None:
    result = estimations.parker_20_calc_z2p5(VS30, region)
    assert result.shape == VS30.shape
    assert np.all(np.diff(result) < 0)

    soft_limit, centre, stiff_limit = estimations.parker_20_calc_z2p5(
        np.array([10.0, vmu, 1e5]), region
    )
    # The logistic (erf) transition spans 2 * theta1 decades, centred on vmu.
    assert np.log10(soft_limit / stiff_limit) == pytest.approx(-2 * theta1)
    assert np.log10(soft_limit / centre) == pytest.approx(-theta1)
