import numpy as np
from scipy.interpolate import interp1d

empirical_periods = np.asarray(
    [
        0.01,
        0.02,
        0.03,
        0.05,
        0.075,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.75,
        1,
        1.5,
        2,
        3,
        4,
        5,
        7.5,
        10,
    ]
)

empirical_values = np.asarray(
    [
        [np.exp(0.176), 0.08, 0.01, 0.08],
        [np.exp(0.175), 0.08, 0.01, 0.08],
        [np.exp(0.172), 0.08, 0.01, 0.08],
        [np.exp(0.171), 0.08, 0.01, 0.08],
        [np.exp(0.172), 0.08, 0.01, 0.08],
        [np.exp(0.172), 0.08, 0.01, 0.08],
        [np.exp(0.182), 0.08, 0.01, 0.08],
        [np.exp(0.187), 0.08, 0.01, 0.08],
        [np.exp(0.196), 0.08, 0.01, 0.08],
        [np.exp(0.198), 0.08, 0.01, 0.08],
        [np.exp(0.206), 0.08, 0.01, 0.08],
        [np.exp(0.206), 0.09, 0.01, 0.09],
        [np.exp(0.213), 0.08, 0.01, 0.09],
        [np.exp(0.216), 0.08, 0.01, 0.08],
        [np.exp(0.217), 0.08, 0.01, 0.08],
        [np.exp(0.218), 0.08, 0.01, 0.08],
        [np.exp(0.221), 0.08, 0.01, 0.08],
        [np.exp(0.231), 0.08, 0.01, 0.08],
        [np.exp(0.235), 0.08, 0.02, 0.08],
        [np.exp(0.251), 0.08, 0.02, 0.08],
        [np.exp(0.258), 0.07, 0.03, 0.08],
    ]
)

_SB_INTERPOLATOR = interp1d(empirical_periods, empirical_values, axis=0)


def ShahiBaker_2013_RotD100_50(im, periods):
    if im is not "pSA":
        raise ValueError("Shahi and Baker 2013 RotD100/50 is only valid for pSA")

    results = _SB_INTERPOLATOR(periods)
    # results[:, 0] = np.log(results[:, 0])

    return results[:, 0], results[:, -1]


def calculate_shahibaker(im, periods):

    mean, sigma = ShahiBaker_2013_RotD100_50(im, periods)
    return mean, sigma
