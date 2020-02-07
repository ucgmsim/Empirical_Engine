from scipy.interpolate import interp1d
import numpy as np

from empirical.util.classdef import Fault

# fmt: off
# Disabled because every cell gets its own line otherwise
#    Period B1       B2       B3       B4       B5       B6     C1      C2     C3      C4     A    B
c_table = np.asarray([
    [0.30, -0.1037,  0.0138, -0.0059,  0.0105, -0.0022, -0.074, 0.002, -0.010, 0.096, -0.077, 0.9, 3.8],
    [0.40,  0.1226, -0.0197, -0.0696,  0.0160, -0.0008,  0.013, 0.008, -0.040, 0.095, -0.045, 0.7, 3.8],
    [0.50,  0.0449, -0.0073, -0.0489,  0.0106, -0.0007,  0.017, 0.006, -0.028, 0.075, -0.036, 0.9, 3.0],
    [0.60,  0.1116, -0.0211, -0.0953,  0.0178, -0.0003,  0.037, 0.008, -0.040, 0.065, -0.010, 1.0, 2.9],
    [0.70, -0.0759,  0.0066, -0.0350,  0.0076, -0.0003,  0.050, 0.007, -0.033, 0.059, -0.013, 1.8, 5.0],
    [0.75, -0.1307,  0.0157, -0.0156,  0.0038,  0.0000,  0.030, 0.006, -0.030, 0.056, -0.012, 1.8, 4.5],
    [0.80, -0.3233,  0.0434,  0.0568, -0.0069,  0.0000, -0.025, 0.009, -0.047, 0.048,  0.016, 1.2, 2.0],
    [0.85, -0.3798,  0.0504,  0.0971, -0.0117, -0.0004, -0.050, 0.011, -0.056, 0.037,  0.035, 1.2, 1.8],
    [0.90, -0.3791,  0.0489,  0.0913, -0.0102, -0.0005, -0.050, 0.014, -0.072, 0.046,  0.041, 1.2, 1.4],
    [1.00, -0.3800,  0.0515,  0.0660, -0.0084, -0.0005, -0.061, 0.013, -0.064, 0.073,  0.007, 1.4, 3.4],
    [1.10, -0.3324,  0.0444,  0.0378, -0.0051,  0.0000, -0.050, 0.011, -0.056, 0.078, -0.008, 1.8, 5.0],
    [1.25, -0.3582,  0.0463,  0.0796, -0.0112,  0.0006, -0.020, 0.012, -0.062, 0.039,  0.037, 1.4, 2.0],
    [1.50, -0.4244,  0.0561,  0.1125, -0.0157,  0.0005,  0.000, 0.008, -0.040, 0.020,  0.038, 1.0, 2.2],
    [2.00, -0.1304,  0.0130, -0.0042,  0.0009,  0.0010,  0.000, 0.008, -0.041, 0.045,  0.014, 1.4, 3.5],
    [2.50, -0.2275,  0.0264,  0.0537, -0.0072,  0.0009,  0.000, 0.008, -0.038, 0.053, -0.006, 1.6, 5.0],
    [3.00,  0.0693, -0.0195, -0.0603,  0.0120,  0.0001,  0.000, 0.013, -0.067, 0.085, -0.009, 1.2, 2.0],
    [4.00, -0.1692,  0.0161,  0.0348, -0.0038,  0.0004,  0.000, 0.018, -0.088, 0.091, -0.005, 1.0, 4.5],
    [5.00, -0.4170,  0.0517,  0.0788, -0.0108,  0.0000,  0.000, 0.035, -0.177, 0.165,  0.017, 0.8, 5.0],
])
# fmt: on

_RATIO_INTERPOLATOR = interp1d(
    c_table[:, 0], c_table[:, 1:], axis=0, fill_value="extrapolate", assume_sorted=True
)


def Burks_Baker_2013_iesdr(T, R_hat, fault: Fault):
    return [fn_sdi_atten(period, R_hat, fault.Mw) for period in T]


def fn_sdi_atten(period, R_hat, magnitude):
    """

    :param period: The period to generate the empirical for
    :param R_hat: The
    :param magnitude:
    :return:
    """
    mu_ratio = np.zeros(len(R_hat))
    sigma_ratio = np.zeros(len(R_hat))

    R_cur = np.min([R_hat, 10 * np.ones_like(R_hat)], axis=0)
    coefs = _RATIO_INTERPOLATOR(period).T

    g1 = (
            (coefs[0] + coefs[1] * magnitude) * R_cur
            + (coefs[2] + coefs[3] * magnitude) * R_cur * np.log(R_cur)
            + coefs[4] * R_cur ** 2.5
    )

    g1_p = (
            (coefs[0] + coefs[1] * magnitude) * 0.2
            + (coefs[2] + coefs[3] * magnitude) * 0.2 * np.log(0.2)
            + coefs[4] * 0.2 ** 2.5
    )

    g2 = np.ones_like(R_cur) * coefs[5]
    g2[R_cur < 3] = (
        0.37 * coefs[5] * np.ones_like(R_cur[R_cur < 3]) * (R_cur[R_cur < 3] - 0.3)
    )
    g2[R_cur < 0.3] = 0

    mu_ratio[R_cur >= 0.2] = g1[R_cur >= 0.2] + g2[R_cur >= 0.2] * (magnitude - 6.5) - g1_p
    sigma_ratio[R_cur >= 0.2] = (
        coefs[6]
        + coefs[7] * R_cur
        + coefs[8]
        * np.max(np.asarray([np.zeros_like(R_cur), R_cur - coefs[10]]), axis=0)
        + coefs[9]
        * np.max(np.asarray([np.zeros_like(R_cur), R_cur - coefs[11]]), axis=0)
    )[R_cur >= 0.2]

    mu_ratio = np.exp(mu_ratio)
    return mu_ratio, sigma_ratio
