import numpy as np
from scipy.interpolate import interp1d

# Stored as [period, ln(mean), total_sigma, intra_sigma, inter_sigma
empirical_values = np.asarray(
    [
        [0.01, np.exp(0.176), 0.08, 0.08, 0.01],
        [0.02, np.exp(0.175), 0.08, 0.08, 0.01],
        [0.03, np.exp(0.172), 0.08, 0.08, 0.01],
        [0.05, np.exp(0.171), 0.08, 0.08, 0.01],
        [0.075, np.exp(0.172), 0.08, 0.08, 0.01],
        [0.1, np.exp(0.172), 0.08, 0.08, 0.01],
        [0.15, np.exp(0.182), 0.08, 0.08, 0.01],
        [0.2, np.exp(0.187), 0.08, 0.08, 0.01],
        [0.25, np.exp(0.196), 0.08, 0.08, 0.01],
        [0.3, np.exp(0.198), 0.08, 0.08, 0.01],
        [0.4, np.exp(0.206), 0.08, 0.08, 0.01],
        [0.5, np.exp(0.206), 0.09, 0.09, 0.01],
        [0.75, np.exp(0.213), 0.09, 0.08, 0.01],
        [1, np.exp(0.216), 0.08, 0.08, 0.01],
        [1.5, np.exp(0.217), 0.08, 0.08, 0.01],
        [2, np.exp(0.218), 0.08, 0.08, 0.01],
        [3, np.exp(0.221), 0.08, 0.08, 0.01],
        [4, np.exp(0.231), 0.08, 0.08, 0.01],
        [5, np.exp(0.235), 0.08, 0.08, 0.02],
        [7.5, np.exp(0.251), 0.08, 0.08, 0.02],
        [10, np.exp(0.258), 0.08, 0.07, 0.03],
    ]
)

_SB_INTERPOLATOR = interp1d(
    np.log(empirical_values[:, 0]), empirical_values[:, 1:], axis=0
)


def ShahiBaker_2013_RotD100_50(im, periods):
    if im != "pSA":
        raise ValueError("Shahi and Baker 2013 RotD100/50 is only valid for pSA")

    results = _SB_INTERPOLATOR(np.log(periods))

    return zip(list(results[:, 0]), list(results[:, 1:]))


def calculate_shahibaker(im, periods):

    mean, sigma = ShahiBaker_2013_RotD100_50(im, periods)
    return mean, sigma
