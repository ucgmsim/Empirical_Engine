
import math

import numpy as np

from empirical.util.classdef import TectType


# fmt: off
periods = np.array([0.000, 0.020, 0.050, 0.075, 0.100, 0.150, 0.200, 0.250, 0.300, 0.400, 0.500, 0.600, 0.750, 1.000, 1.500, 2.000, 2.500, 3.000, 4.000, 5.000, 6.000, 7.500, 10.000])
Vlin = [865.1, 865.1, 1053.5, 1085.7, 1032.5, 877.6, 748.2, 654.3, 587.1, 503.0, 456.6, 430.3, 410.5, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]
b = [-1.186, -1.186, -1.346, -1.471, -1.624, -1.931, -2.188, -2.381, -2.518, -2.657, -2.669, -2.599, -2.401, -1.955, -1.025, -0.299, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000]
n = 1.18
c = 1.88
C1 = 7.8
C4 = 10
theta1 = [4.2203, 4.2203, 4.5371, 5.0733, 5.2892, 5.4563, 5.2684, 5.0594, 4.7945, 4.4644, 4.0181, 3.6055, 3.2174, 2.7981, 2.0123, 1.4128, 0.9976, 0.6443, 0.0657, -0.4624, -0.9809, -1.6017, -2.2937]
theta2 = [-1.35, -1.35, -1.40, -1.45, -1.45, -1.45, -1.40, -1.35, -1.28, -1.18, -1.08, -0.99, -0.91, -0.85, -0.77, -0.71, -0.67, -0.64, -0.58, -0.54, -0.50, -0.46, -0.40]
theta3 = 0.1
theta4 = 0.9
theta5 = 0.0
theta6 = [-0.0012, -0.0012, -0.0012, -0.0012, -0.0012, -0.0014, -0.0018, -0.0023, -0.0027, -0.0035, -0.0044, -0.0050, -0.0058, -0.0062, -0.0064, -0.0064, -0.0064, -0.0064, -0.0064, -0.0064, -0.0064, -0.0064, -0.0064]
theta7 = [1.0988, 1.0988, 1.2536, 1.4175, 1.3997, 1.3582, 1.1648, 0.9940, 0.8821, 0.7046, 0.5799, 0.5021, 0.3687, 0.1746, -0.0820, -0.2821, -0.4108, -0.4466, -0.4344, -0.4368, -0.4586, -0.4433, -0.4828]
theta8 = [-1.42, -1.42, -1.65, -1.80, -1.80, -1.69, -1.49, -1.30, -1.18, -0.98, -0.82, -0.70, -0.54, -0.34, -0.05, 0.12, 0.25, 0.30, 0.30, 0.30, 0.30, 0.30, 0.30]
theta9 = 0.4
theta10 = [3.12, 3.12, 3.37, 3.37, 3.33, 3.25, 3.03, 2.80, 2.59, 2.20, 1.92, 1.70, 1.42, 1.10, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70, 0.70]
theta11 = [0.0130, 0.0130, 0.0130, 0.0130, 0.0130, 0.0130, 0.0129, 0.0129, 0.0128, 0.0127, 0.0125, 0.0124, 0.0120, 0.0114, 0.0100, 0.0085, 0.0069, 0.0054, 0.0027, 0.0005, -0.0013, -0.0033, -0.0060]
theta12 = [0.980, 0.980, 1.288, 1.483, 1.613, 1.882, 2.076, 2.248, 2.348, 2.427, 2.399, 2.273, 1.993, 1.470, 0.408, -0.401, -0.723, -0.673, -0.627, -0.596, -0.566, -0.528, -0.504]
theta13 = [-0.0135, -0.0135, -0.0138, -0.0142, -0.0145, -0.0153, -0.0162, -0.0172, -0.0183, -0.0206, -0.0231, -0.0256, -0.0296, -0.0363, -0.0493, -0.0610, -0.0711, -0.0798, -0.0935, -0.0980, -0.0980, -0.0980, -0.0980]
theta14 = [-0.40, -0.40, -0.40, -0.40, -0.40, -0.40, -0.35, -0.31, -0.28, -0.23, -0.19, -0.16, -0.12, -0.07, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
theta15 = [0.9996, 0.9996, 1.1030, 1.2732, 1.3042, 1.2600, 1.2230, 1.1600, 1.0500, 0.8000, 0.6620, 0.5800, 0.4800, 0.3300, 0.3100, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000, 0.3000]
theta16 = [-1.00, -1.00, -1.18, -1.36, -1.36, -1.30, -1.25, -1.17, -1.06, -0.78, -0.62, -0.50, -0.34, -0.14, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]
phi = 0.6
tau = 0.43
sigma_ss = 0.6
# fmt: on


def bc_hydro_2016_subduction(siteprop, faultprop, im, period):
    """
    % Created by Reagan Chandramohan, circa 2017
    % modified by Jack Baker, 2/27/2019, to limit hypocentral depths to 120 km
    %
    % Predict response spectra for subduction earthquakes, per the following
    % model:
    %
    % Abrahamson, N., Gregor, N., and Addo, K. (2016). "BC Hydro Ground Motion 
    % Prediction Equations for Subduction Earthquakes." Earthquake Spectra, 
    % 32(1), 23-44.
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % INPUT
    %
    % siteprop.Rrup = Interface: Closest distance to rupture (km)
    %                 Intraslab: Distance to hypocenter (km)
    % siteprop.backarc = False for forearc or unknown sites
    %                    True for backarc sites
    % siteprop.vs30 = Average shear wave velocity over the top 30 m of the soil
    %                 profile
    % faultprop.Mw  = Moment magnitude
    % faultprop.tect_type = SUBDUCTION_INTERFACE or SUBDUCTION_SLAB only
    % faultprop.hdepth = Hypocentral depth (km) (only for intraslab events)
    % period  = period
    %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    % OUTPUT
    % sa_int  = vector of median response spectral ordinates (g)
    % sigma   = vector of logarithmic standard deviation
    %
    """
    if im == 'PGA':
        period = [0]
    try:
        period[0]
    except (TypeError, IndexError) as e:
        period = [period]

    M = faultprop.Mw
    R = siteprop.Rrup
    f_slab = faultprop.tect_type == TectType.SUBDUCTION_SLAB
    base = np.zeros(len(periods))
    f_mag = np.zeros(len(periods))
    f_dep = np.zeros(len(periods))
    f_faba = np.zeros(len(periods))
    f_site = np.zeros(len(periods))

    if f_slab:
        deltaC1 = -0.3 * np.ones(len(periods))
    else:
        deltaC1 = np.interp(
            np.log(periods),
            np.log([1e-10, 0.3, 0.5, 1.0, 2.0, 3.0, 1e10]),
            [0.2, 0.2, 0.1, 0.0, -0.1, -0.2, -0.2],
        )
        deltaC1[0] = 0.2

    Vs_star = min(siteprop.vs30, 1000)

    # seems silly to loop over all periods
    # TODO: remove periods unused for interpolation keeping 0
    for i in range(len(periods)):
        base[i] = (
            theta1[i]
            + theta4 * deltaC1[i]
            + (theta2[i] + theta14[i] * f_slab + theta3 * (M - C1))
            * math.log(R + C4 * math.exp((M - 6) * theta9))
            + theta6[i] * R
            + theta10[i] * f_slab
        )

        if M <= C1 + deltaC1[i]:
            f_mag[i] = theta4 * (M - (C1 + deltaC1[i])) + theta13[i] * (10 - M) ** 2
        else:
            # f_mag[i] = theta5 * (M - (C1 + deltaC1[i])) + ... (removed 0 + )
            f_mag[i] = theta13[i] * (10 - M) ** 2

        if f_slab:
            # modified by JWB, 2/27/2019, to include the "min" term, per equation 3 of the paper
            f_dep[i] = theta11[i] * (min(120, faultprop.hdepth) - 60) * f_slab
        else:
            f_dep[i] = 0

        if f_slab:
            f_faba[i] = (
                theta7[i] + theta8[i] * math.log(max(R, 85) / 40)
            ) * siteprop.backarc
        else:
            f_faba[i] = (
                theta15[i] + theta16[i] * math.log(max(R, 100) / 40)
            ) * siteprop.backarc

        if i == 0:
            PGA1000 = math.exp(
                base[0]
                + f_mag[0]
                + f_dep[0]
                + f_faba[0]
                + theta12[0] * math.log(1000 / Vlin[0])
                + b[0] * n * math.log(1000 / Vlin[0])
            )
        if siteprop.vs30 < Vlin[i]:
            f_site[i] = (
                theta12[i] * math.log(Vs_star / Vlin[i])
                - b[i] * math.log(PGA1000 + c)
                + b[i] * math.log(PGA1000 + c * (Vs_star / Vlin[i]) ** n)
            )
        else:
            f_site[i] = theta12[i] * math.log(Vs_star / Vlin[i]) + b[i] * n * math.log(
                Vs_star / Vlin[i]
            )

    sa = np.exp(base + f_mag + f_dep + f_faba + f_site)
    periods_int = np.copy(periods)
    periods_int[0] = 1e-10
    if im != "PGA":
        sa_int = np.interp(np.log(period), np.log(periods_int), sa)
    else:
        sa_int = sa[0]
    sigma = compute_stdev()

    if im == 'PGA' or len(period) == 1:
        return sa_int, sigma
    else:
        return list(zip(sa_int, [sigma] * len(sa_int)))


def compute_stdev():
    return math.sqrt(phi ** 2 + tau ** 2), tau, phi
