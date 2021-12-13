import math

import numpy as np

from empirical.util.classdef import interpolate_to_closest
from empirical.util.classdef import TectType


# fmt: off
imt = [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 10, 0]
# constants
n = 1.18
c = 1.88
C4 = 10.0
a3 = 0.10
a5 = 0.0
a9 = 0.4
a10 = 1.73
C1slab = 7.2
phiamp = 0.3
# coefficients
c1inter = [8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.2, 8.15, 8.1, 8.05, 8, 7.95, 7.9, 7.85, 7.8, 7.8, 7.8, 7.8, 8.2]
vlin = [865.1, 865.1, 907.8, 1053.5, 1085.7, 1032.5, 877.6, 748.2, 654.3, 587.1, 503, 456.6, 430.3, 410.5, 400, 400, 400, 400, 400, 400, 400, 400, 400, 400, 865.1]
b = [-1.186, -1.219, -1.273, -1.346, -1.471, -1.624, -1.931, -2.188, -2.381, -2.518, -2.657, -2.669, -2.599, -2.401, -1.955, -1.025, -0.299, 0, 0, 0, 0, 0, 0, 0, -1.186]
a1 = [2.34047, 2.36005, 2.38396, 2.44598, 2.75111, 3.01943, 3.34867, 3.28397, 3.21131, 3.14548, 2.99656, 2.83921, 2.65827, 2.34577, 1.85131, 1.21559, 0.64875, 0.08221, -0.36926, -1.03439, -1.51967, -1.81025, -2.17269, -2.71182, 2.34047]
a2 = [-1.044, -1.044, -1.08, -1.11, -1.11, -1.11, -1.084, -1.027, -0.983, -0.947, -0.89, -0.845, -0.809, -0.76, -0.698, -0.612, -0.55, -0.501, -0.46, -0.455, -0.45, -0.45, -0.45, -0.45, -1.044]
a4 = [0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.59, 0.62, 0.64, 0.66, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.68, 0.73, 0.78, 0.84, 0.93, 0.59]
a6 = [-0.00705, -0.00707, -0.0071, -0.00725, -0.00758, -0.00788, -0.0082, -0.00835, -0.00835, -0.00828, -0.00797, -0.0077, -0.0074, -0.00698, -0.00645, -0.0057, -0.0051, -0.00465, -0.0043, -0.0039, -0.0037, -0.00357, -0.0034, -0.00327, -0.00705]
a11 = [0.017, 0.017, 0.017, 0.018, 0.018, 0.018, 0.0175, 0.017, 0.016, 0.0152, 0.014, 0.013, 0.0122, 0.0113, 0.01, 0.0082, 0.007, 0.006, 0.0052, 0.004, 0.003, 0.0022, 0.0013, 0, 0.017]
a12 = [0.818, 0.857, 0.921, 1.007, 1.225, 1.457, 1.849, 2.082, 2.24, 2.341, 2.415, 2.359, 2.227, 1.949, 1.402, 0.329, -0.487, -0.77, -0.7, -0.607, -0.54, -0.479, -0.393, -0.35, 0.818]
a13 = [-0.0135, -0.0135, -0.0135, -0.0138, -0.0142, -0.0145, -0.0153, -0.0162, -0.0172, -0.0183, -0.0206, -0.0231, -0.0256, -0.0296, -0.0363, -0.0493, -0.061, -0.0711, -0.0798, -0.0935, -0.098, -0.098, -0.098, -0.098, -0.0135]
a14 = [-0.223, -0.196, -0.128, -0.13, -0.13, -0.13, -0.156, -0.172, -0.184, -0.194, -0.21, -0.223, -0.233, -0.245, -0.261, -0.285, -0.301, -0.313, -0.323, -0.282, -0.25, -0.25, -0.25, -0.25, -0.223]
# adjustment variables to match Cascadia to global average
adj_int = [1.04398007004686, 1.0460352342765, 1.23377425709977, 1.34342293514146, 1.32123133203975, 1.32307413547709, 1.20604738361364, 1.14232895808973, 1.04808298000789, 0.94565364697989, 0.79394049645376, 0.66202915411124, 0.53601960759514, 0.35555118688641, 0.242825, -0.0768025, -0.20847, -0.2095575, -0.2166025, -0.06406, 0.0645975, 0.09014, 0.1379325, 0.31434, 1.04398007004686]
adj_slab = [0.8342922160217, 0.79173401239125, 0.70576030699887, 0.9760103290697, 0.9920217692103, 0.99825322734373, 0.91891214881125, 0.87856284947151, 0.81234114576059, 0.74536286357934, 0.62046390803996, 0.54491903264494, 0.45568525725929, 0.32581112214331, 0.22809166666667, -0.01107333333333, -0.08503, -0.05003166666667, -0.01508333333333, 0.06239166666667, 0.06338666666667, 0.09661, 0.125775, 0.23962833333333, 0.8342922160217]
phi0 = [0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61, 0.61]
tau0 = [0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.56, 0.54, 0.52, 0.505, 0.48, 0.46, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.45, 0.58]
rho_w = [1, 1, 0.991, 0.973, 0.952, 0.929, 0.896, 0.874, 0.856, 0.841, 0.818, 0.783, 0.7315, 0.68, 0.607, 0.5004, 0.4301, 0.3795, 0.328, 0.2505, 0.2, 0.2, 0.2, 0.2, 1]
rho_b = [1, 1, 0.991, 0.973, 0.952, 0.929, 0.896, 0.874, 0.856, 0.841, 0.818, 0.783, 0.7315, 0.68, 0.607, 0.504, 0.431, 0.3795, 0.328, 0.255, 0.2, 0.2, 0.2, 0.2, 1]
# epistemic adjustment variables
sinter_low = [-0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3]
sinter_high = [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
sslab_low = [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.46, -0.42, -0.38, -0.34, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.3, -0.5]
sslab_high = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.46, 0.42, 0.38, 0.34, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.5]
# fmt: on


def Abrahamson_2018(site, fault, im=None, periods=None, epistemic_adj=None):
    """

    component returned: geometric mean

    Input Variables

    siteprop.Rrup: closest distance (km) to the ruptured plane
    siteprop.vs30: for site amplification

    faultprop.Mw: moment magnitude
    faultprop.tect_type: SUBDUCTION_INTERFACE or SUBDUCTION_SLAB only
    faultprop.ztor: if SUBDUCTION_SLAB only. depth (km) to the top of ruptured plane

    im: PGA or SA only

    epistemic_adj: None: disabled epistemic adjustment
                   "HIGH": subduction interface or in-slab GMPE with the positive epistemic adjustment factor applied
                   "LOW":  subduction interface or in-slab GMPE with the negative epistemic adjustment factor applied
    """
    results = []
    if im == "PGA":
        periods = [0]
    try:
        periods[0]
    except (TypeError, IndexError) as e:
        periods = [periods]

    for t in periods:
        sorted_t = np.array(sorted(imt))
        closest_index = int(np.argmin(np.abs(sorted_t - t)))
        closest_period = sorted_t[closest_index]
        if np.isclose(closest_period, t):
            result = calculate_Abrahamson(site, fault, closest_period, epistemic_adj)
        else:
            t_low = sorted_t[t >= sorted_t][-1]
            t_high = sorted_t[t <= sorted_t][0]

            a_low = calculate_Abrahamson(site, fault, t_low, epistemic_adj)
            a_high = calculate_Abrahamson(site, fault, t_high, epistemic_adj)

            result = interpolate_to_closest(t, t_high, t_low, a_high, a_low)
        results.append(result)

    if len(periods) == 1:
        results = results[0]

    return results


def calculate_Abrahamson(site, fault, period, epistemic_adj):
    C = imt.index(period)
    C_PGA = imt.index(0)
    if fault.tect_type == TectType.SUBDUCTION_INTERFACE:
        base = a1[C]
        base_pga = a1[C_PGA]
        depth = 0.0
        depth_pga = 0.0
        # magnitude scaling term that modifies the distance attenuation
        mag_scale = a2[C] + a3 * (fault.Mw - 7.8)
        mag_scale_pga = a2[C_PGA] + a3 * (fault.Mw - 7.8)
        adj = adj_int
        hinge_mw = c1inter[C]
        if epistemic_adj is not None:
            if epistemic_adj == "HIGH":
                eadj = sinter_high[C]
            elif epistemic_adj == "LOW":
                eadj = sinter_low[C]
    elif fault.tect_type == TectType.SUBDUCTION_SLAB:
        base = a1[C] + a4[C] * (C1slab - c1inter[C]) + a10
        base_pga = a1[C_PGA] + a4[C_PGA] * (C1slab - c1inter[C_PGA]) + a10
        if fault.ztor <= 100.0:
            depth = a11[C] * (fault.ztor - 60.0)
            depth_pga = a11[C_PGA] * (fault.ztor - 60.0)
        else:
            depth = a11[C] * (100.0 - 60.0)
            depth_pga = a11[C_PGA] * (100.0 - 60.0)
        mag_scale = a2[C] + a14[C] + a3 * (fault.Mw - 7.8)
        mag_scale_pga = a2[C_PGA] + a14[C_PGA] + a3 * (fault.Mw - 7.8)
        adj = adj_slab
        hinge_mw = C1slab
        if epistemic_adj is not None:
            if epistemic_adj == "HIGH":
                eadj = sslab_high
            elif epistemic_adj == "LOW":
                eadj = sslab_low
    else:
        raise ValueError(
            "TectType must be SUBDUCTION_SLAB or SUBDUCTION_INTERFACE for this empirical model"
        )

    # magnitude scaling term
    f_mag = a13[C] * ((10.0 - fault.Mw) ** 2.0)
    f_mag_pga = a13[C_PGA] * ((10.0 - fault.Mw) ** 2.0)
    if fault.Mw <= hinge_mw:
        f_mag = a4[C] * (fault.Mw - hinge_mw) + f_mag
        f_mag_pga = a4[C_PGA] * (fault.Mw - hinge_mw) + f_mag
    # distance attenuation
    f_dist = (
        mag_scale * np.log(site.Rrup + C4 * np.exp(a9 * (fault.Mw - 6.0)))
        + a6[C] * site.Rrup
    )
    f_dist_pga = (
        mag_scale_pga * np.log(site.Rrup + C4 * np.exp(a9 * (fault.Mw - 6.0)))
        + a6[C_PGA] * site.Rrup
    )
    # linear site term for the case where vs30 = 1000.0
    f_lin = (a12[C_PGA] + b[C_PGA] * n) * np.log(1000.0 / vlin[C_PGA])
    # PGA on rock (vs30 = 1000 m / s) + linear site term
    lpga1000 = base_pga + f_mag_pga + depth_pga + f_dist_pga + f_lin
    # compute median pga on rock (vs30=1000), needed for site response
    # term calculation
    pga1000 = np.exp(lpga1000 + adj[C_PGA])

    if site.vs30 >= 1000.0:
        vsstar = 1000.0
    else:
        vsstar = site.vs30
    if site.vs30 >= vlin[C]:
        f_site = (a12[C] + b[C] * n) * np.log(vsstar / vlin[C])
    else:
        # linear term
        flin = a12[C] * np.log(vsstar / vlin[C])
        # nonlinear term
        fnl = (-b[C] * np.log(pga1000 + c)) + (
            b[C] * np.log(pga1000 + c * (vsstar / vlin[C]) ** n)
        )
        f_site = flin + fnl
    # Get full model
    mean = base + f_mag + depth + f_dist + f_site

    std_dev = compute_stdev(C, C_PGA, pga1000, site.vs30)
    if epistemic_adj is not None:
        mean = np.exp(mean + adj[C] + eadj)
    else:
        mean = np.exp(mean + adj[C])

    return mean, std_dev


def compute_stdev(C, C_PGA, pga1000, vs30):
    # partial derivative of the amplification term with respect to pga1000
    if vs30 < vlin[C]:
        dln = (
            b[C]
            * pga1000
            * (-1.0 / (pga1000 + c) + (1.0 / (pga1000 + c * (vs30 / vlin[C]) ** n)))
        )
    else:
        dln = 0.0

        # between event aleatory uncertainty, tau
    tau = math.sqrt(
        (tau0[C] ** 2.0)
        + (dln ** 2.0 * tau0[C] ** 2.0)
        + (2.0 * dln * tau0[C] * tau0[C_PGA] * rho_b[C])
    )

    # within-event aleatory uncertainty, phi
    phi_amp2 = phiamp ** 2.0
    phi_b = math.sqrt(phi0[C] ** 2.0 - phi_amp2)
    phi_b_pga = np.sqrt((phi0[C_PGA] ** 2.0) - phi_amp2)
    phi = math.sqrt(
        phi0[C] ** 2.0
        + dln ** 2.0 * phi_b ** 2.0
        + 2.0 * dln * phi_b * phi_b_pga * rho_w[C]
    )
    # assume total
    total = math.sqrt(tau ** 2.0 + phi ** 2.0)
    return total, tau, phi
