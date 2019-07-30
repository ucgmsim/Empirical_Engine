import pytest
from empirical.util import classdef

DIGITS = 3

# test inputs for estimation
VS30 = 500
Z1P0 = 0.1111743647408942
Z1P5 = 0.18248666316560017
Z2P5 = 0.9186718412435146
TEST_PARAMS_Z2P5 = [(Z1P0, Z1P5, Z2P5), (Z1P0, None, Z2P5), (None, Z1P5, Z2P5)]


# test inputs for interpolation closest
T = 8.69749
T_HIGH = 10
T_LOW = 7.5
Y_HIGH = (0.021762019779086095, [0.78030565357429, 0.5542, 0.5493079946623752])
Y_LOW = (0.03801453146006103, [0.7652549333392108, 0.5328, 0.5493079946623752])
EXPECTED_INTERPOLATION_1 = (
    0.028524074121256727,
    [0.7730047562932064, 0.543819154474014, 0.5493079946623752],
)
EXPECTED_INTERPOLATION_2 = (
    0.022971680292776762,
    [0.7791854413678104, 0.5526072163428571, 0.5493079946623752],
)
TEST_PARAMS_INTERPOLATION = [
    (T, T_HIGH, T_LOW, Y_HIGH, Y_LOW, EXPECTED_INTERPOLATION_1),
    (T, T_HIGH, -T_LOW, Y_HIGH, Y_LOW, EXPECTED_INTERPOLATION_2),
]


def test_estimate_z1p0():
    assert round(classdef.estimate_z1p0(VS30), DIGITS) == round(Z1P0, DIGITS)


@pytest.mark.parametrize("test_z1p0, test_z1p5, expected_z2p5", TEST_PARAMS_Z2P5)
def test_estimate_z2p5(test_z1p0, test_z1p5, expected_z2p5):
    assert round(classdef.estimate_z2p5(test_z1p0, test_z1p5), DIGITS) == round(
        expected_z2p5, DIGITS
    )


# Test for fail case, neither z1p0 nor z2p5 is provided
def test_estimate_z2p5_exit():
    with pytest.raises(SystemExit):
        classdef.estimate_z2p5()


@pytest.mark.parametrize(
    "test_t, test_t_high, test_t_low, test_y_high, test_y_low, expected",
    TEST_PARAMS_INTERPOLATION,
)
def test_interpolate_to_closest(
    test_t, test_t_high, test_t_low, test_y_high, test_y_low, expected
):
    sa, sigma_sas = classdef.interpolate_to_closest(
        test_t, test_t_high, test_t_low, test_y_high, test_y_low
    )
    assert (round(sa, DIGITS), [round(sigma_sa, DIGITS) for sigma_sa in sigma_sas]) == (
        round(expected[0], DIGITS),
        [round(s, DIGITS) for s in expected[-1]],
    )
