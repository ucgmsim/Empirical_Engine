import numpy as np
#import numba
from scipy.constants import g

from empirical.util.classdef import (
    TectType,
    FaultStyle,
    SiteClass,
    interpolate_to_closest,
)

# fmt: off
period_list = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50,
                        2.00, 2.50, 3.00, 4.00, 5.00])
a = np.array([1.101, 1.076, 1.118, 1.134, 1.147, 1.149, 1.163, 1.200, 1.25, 1.293, 1.336, 1.386, 1.433, 1.479, 1.551,
              1.621, 1.694, 1.748, 1.759, 1.826, 1.825])
b = np.array([-0.00564, -0.00671, -0.00787, -0.00722, -0.00659, -0.00590, -0.00520, -0.00422, -0.00338, -0.00282,
              -0.00258, -0.00242, -0.00232, -0.00220, -0.00207, -0.00224, -0.00201, -0.00187, -0.00147, -0.00195,
              -0.00237])
c = np.array([0.0055, 0.0075, 0.0090, 0.0100, 0.0120, 0.0140, 0.0150, 0.0100, 0.0060, 0.0030, 0.0025, 0.0022, 0.0020,
              0.0020, 0.0020, 0.0020, 0.0025, 0.0028, 0.0032, 0.0040, 0.0050])
d = np.array([1.080, 1.060, 1.083, 1.053, 1.014, 0.966, 0.934, 0.959, 1.008, 1.088, 1.084, 1.088, 1.109, 1.115, 1.083,
              1.091, 1.055, 1.052, 1.025, 1.044, 1.065])
e = np.array([0.01412, 0.01463, 0.01423, 0.01509, 0.01462, 0.01459, 0.01458, 0.01257, 0.01114, 0.01019, 0.00979,
              0.00944, 0.00972, 0.01005, 0.01003, 0.00928, 0.00833, 0.00776, 0.00644, 0.00590, 0.00510])
SR = np.array([0.251, 0.251, 0.240, 0.251, 0.260, 0.269, 0.259, 0.248, 0.247, 0.233, 0.22, 0.232, 0.220, 0.211, 0.251,
               0.248, 0.263, 0.262, 0.307, 0.353, 0.248])
SI = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.041, -0.053, -0.103, -0.146, -0.164, -0.206, -0.239,
               -0.256, -0.306, -0.321, -0.337, -0.331, -0.390, -0.498])
SS = np.array([2.607, 2.764, 2.156, 2.161, 1.901, 1.814, 2.181, 2.432, 2.629, 2.702, 2.654, 2.480, 2.332, 2.233, 2.029,
               1.589, 0.966, 0.789, 1.037, 0.561, 0.225])
SSL = np.array([-0.528, -0.551, -0.420, -0.431, -0.372, -0.360, -0.450, -0.506, -0.554, -0.575, -0.572, -0.540, -0.522,
                -0.509, -0.469, -0.379, -0.248, -0.221, -0.263, -0.169, -0.120])
CH = np.array([0.293, 0.939, 1.499, 1.462, 1.280, 1.121, 0.852, 0.365, -0.207, -0.705, -1.144, -1.609, -2.023, -2.451,
               -3.243, -3.888, -4.783, -5.444, -5.839, -6.598, -6.752])
C1 = np.array([1.111, 1.684, 2.061, 1.916, 1.669, 1.468, 1.172, 0.655, 0.071, -0.429, -0.866, -1.325, -1.732, -2.152,
               -2.923, -3.548, -4.410, -5.049, -5.431, -6.181, -6.347])
C2 = np.array([1.344, 1.793, 2.135, 2.168, 2.085, 1.942, 1.683, 1.127, 0.515, -0.003, -0.449, -0.928, -1.349, -1.776,
               -2.542, -3.169, -4.039, -4.698, -5.089, -5.882, -6.051])
C3 = np.array([1.355, 1.747, 2.031, 2.052, 2.001, 1.941, 1.808, 1.482, 0.934, 0.394, -0.111, -0.620, -1.066, -1.523,
               -2.327, -2.979, -3.871, -4.496, -4.893, -5.698, -5.873])
C4 = np.array([1.420, 1.814, 2.082, 2.113, 2.030, 1.937, 1.770, 1.397, 0.955, 0.559, 0.188, -0.246, -0.643, -1.084,
               -1.936, -2.661, -3.64, -4.341, -4.758, -5.588, -5.798])
sigma = np.array([0.604, 0.640, 0.694, 0.702, 0.692, 0.682, 0.670, 0.659, 0.653, 0.653, 0.652, 0.647, 0.653, 0.657,
                  0.660, 0.664, 0.669, 0.671, 0.667, 0.647, 0.643])
tau_s = np.array([0.321, 0.378, 0.420, 0.372, 0.324, 0.294, 0.284, 0.278, 0.272, 0.285, 0.29, 0.299, 0.289, 0.286,
                  0.277, 0.282, 0.300, 0.292, 0.274, 0.281, 0.296])
tau_i = np.array([0.308, 0.343, 0.403, 0.367, 0.328, 0.289, 0.280, 0.271, 0.277, 0.296, 0.313, 0.329, 0.324, 0.328,
                  0.339, 0.352, 0.360, 0.356, 0.338, 0.307, 0.272])
tau_c = np.array([0.303, 0.326, 0.342, 0.331, 0.312, 0.298, 0.300, 0.346, 0.338, 0.349, 0.351, 0.356, 0.348, 0.338,
                  0.313, 0.306, 0.283, 0.287, 0.278, 0.273, 0.275])

Ps = np.array([0.1392, 0.1636, 0.1690, 0.1669, 0.1631, 0.1588, 0.1544, 0.146, 0.1381, 0.1307, 0.1239, 0.1176, 0.1116,
               0.106, 0.0933, 0.0821, 0.0628, 0.0465, 0.0322, 0.0083, -0.0117])
Qs = np.array([0.1584, 0.1932, 0.2057, 0.1984, 0.1856, 0.1714, 0.1573, 0.1309, 0.1078, 0.0878, 0.0705, 0.0556, 0.0426,
               0.0314, 0.0093, -0.0062, -0.0235, -0.0287, -0.0261, -0.0065, 0.0246])
Ws = np.array([-0.0529, -0.0841, -0.0877, -0.0773, -0.0644, -0.0515, -0.0395, -0.0183, -0.0008, 0.0136, 0.0254, 0.0352,
               0.0432, 0.0498, 0.0612, 0.0674, 0.0692, 0.0622, 0.0496, 0.0150, -0.0268])

Qi = np.array([0.0000, 0.0000, 0.0000, -0.0138, -0.0256, -0.0348, -0.0423, -0.0541, -0.0632, -0.0707, -0.0771, -0.0825,
               -0.0874, -0.0917, -0.1009, -0.1083, -0.1202, -0.1293, -0.1368, -0.1486, -0.1578])
Wi = np.array([0.0000, 0.0000, 0.0000, 0.0286, 0.0352, 0.0403, 0.0445, 0.0511, 0.0562, 0.0604, 0.0639, 0.0670, 0.0697,
               0.0721, 0.0772, 0.0814, 0.0880, 0.0931, 0.0972, 0.1038, 0.1090])

Qc = np.array([0.000, 0.000, 0.000, 0.000, 0.000, 0.0000, 0.0000, 0.0000, -0.0126, -0.0329, -0.0501, -0.065, -0.0781,
               -0.0899, -0.1148, -0.1351, -0.1672, -0.1921, -0.2124, -0.2445, -0.2694])
Wc = np.array([0.0000, 0.0000, 0.0000, 0.000, 0.0000, 0.000, 0.0000, 0.0000, 0.0116, 0.0202, 0.0274, 0.0336, 0.0391,
               0.044, 0.0545, 0.063, 0.0764, 0.0869, 0.0954, 0.1088, 0.1193])
# fmt: on

zeros = np.zeros(21)


def Zhaoetal_2006_Sa(site, fault, im, periods=None):
    if im == "PGA":
        periods = [0]
    try:
        periods[0]
    except (TypeError, IndexError) as e:
        periods = [periods]

    results = []
    for period in periods:

        T = period

        max_period = period_list[-1]
        if period > max_period:
            zhao_max = calculate_zhao(site, fault, max_period)
            median_max = zhao_max[0]
            median = median_max * (max_period / period) ** 2
            result = (median, zhao_max[1])
        else:
            # interpolate between periods if necessary
            closest_index = np.argmin(np.abs(period_list - T))
            closest_period = period_list[closest_index]

            if not np.isclose(closest_period, T):
                T_low = period_list[T >= period_list][-1]
                T_hi = period_list[T <= period_list][0]

                zhao_low = calculate_zhao(site, fault, T_low)
                zhao_high = calculate_zhao(site, fault, T_hi)

                result = interpolate_to_closest(T, T_hi, T_low, zhao_high, zhao_low)
            else:
                result = calculate_zhao(site, fault, period)
        results.append(result)

    if len(periods) == 1:
        results = results[0]

    return results


def calculate_zhao(site, fault, period):
    m = fault.Mw
    r = site.Rrup

    closest_index = np.argmin(np.abs(period_list - period))
    i = int(closest_index)
    fault_s = zeros
    fault_ssl = zeros

    p_fa = q_fa = w_fa = zeros
    mc = 6.3  # mc defaults to 6.3, unless subduction slab

    if fault.tect_type == TectType.ACTIVE_SHALLOW:
        q_fa = Qc
        w_fa = Wc

        if fault.faultstyle == FaultStyle.REVERSE:
            fault_s = SR
        else:
            fault_s = zeros
    elif fault.tect_type == TectType.SUBDUCTION_INTERFACE:
        fault_s = SI
        q_fa = Qi
        w_fa = Wi
    elif fault.tect_type == TectType.SUBDUCTION_SLAB:
        fault_s = SS
        fault_ssl = SSL
        p_fa = Ps
        q_fa = Qs
        w_fa = Ws

        mc = 6.5

    site_c = zeros
    if site.siteclass == SiteClass.HARDROCK:
        site_c = CH
    elif site.siteclass == SiteClass.ROCK:
        site_c = C1
    elif site.siteclass == SiteClass.HARDSOIL:
        site_c = C2
    elif site.siteclass == SiteClass.MEDIUMSOIL:
        site_c = C3
    elif site.siteclass == SiteClass.SOFTSOIL:
        site_c = C4
    h = fault.hdepth

    tau = get_tau(fault.tect_type)

    return ZA06(i, m, h, r, fault_s, fault_ssl, p_fa, q_fa, w_fa, site_c, tau, mc)


#@numba.jit(nopython=True)
def ZA06(i, m, h, r, fault_s, fault_ssl, p_fa, q_fa, w_fa, site_c, tau, mc):
    hc = 15
    r_star = r + c[i] * np.exp(d[i] * m)
    log_sa = (
        a[i] * m
        + b[i] * r
        - np.log(r_star)
        + e[i] * max(min(h, 125) - hc, 0)
        + fault_s[i]
        + fault_ssl[i] * (np.log(r))  # - np.log(125.0))
        + site_c[i]
    )
    m2_corr_fact = p_fa[i] * (m - mc) + (q_fa[i] * (m - mc) ** 2) + w_fa[i]
    log_sa += m2_corr_fact
    # convert to median in g
    sa = np.exp(log_sa) * 1e-2 / g
    sigma_sa = determine_stdev(i, tau)
    return sa, sigma_sa


#@numba.jit(nopython=True)
def determine_stdev(i, tau):
    sigma_intra = tau[i]
    sigma_inter = sigma[i]
    sigma_total = np.sqrt(sigma_intra ** 2 + sigma_inter ** 2)
    sigma_sa = [sigma_total, sigma_inter, sigma_intra]
    return sigma_sa


def get_tau(tect_type):
    if tect_type == TectType.ACTIVE_SHALLOW:
        tau = tau_c
    elif tect_type == TectType.SUBDUCTION_INTERFACE:
        tau = tau_i
    elif tect_type == TectType.SUBDUCTION_SLAB:
        tau = tau_s
    else:
        print("TectType is unset assuming SHALLOW CRUSTAL")
        tau = tau_c
    return tau
