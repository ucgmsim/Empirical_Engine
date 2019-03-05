import numpy as np
from .classdef import TectType
from .classdef import FaultStyle
from .classdef import SiteClass
from .classdef import interpolate_to_closest

period_list = np.array([0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.25, 1.50,
                        2.00, 2.50, 3.00, 4.00, 5.00])
a = [1.101, 1.076, 1.118, 1.134, 1.147, 1.149, 1.163, 1.200, 1.25, 1.293, 1.336, 1.386, 1.433, 1.479, 1.551, 1.621,
     1.694, 1.748, 1.759, 1.826, 1.825]
b = [-0.00564, -0.00671, -0.00787, -0.00722, -0.00659, -0.00590, -0.00520, -0.00422, -0.00338, -0.00282, -0.00258,
     -0.00242, -0.00232, -0.00220, -0.00207, -0.00224, -0.00201, -0.00187, -0.00147, -0.00195, -0.00237]
c = [0.0055, 0.0075, 0.0090, 0.0100, 0.0120, 0.0140, 0.0150, 0.0100, 0.0060, 0.0030, 0.0025, 0.0022, 0.0020, 0.0020,
     0.0020, 0.0020, 0.0025, 0.0028, 0.0032, 0.0040, 0.0050]
d = [1.080, 1.060, 1.083, 1.053, 1.014, 0.966, 0.934, 0.959, 1.008, 1.088, 1.084, 1.088, 1.109, 1.115, 1.083, 1.091,
     1.055, 1.052, 1.025, 1.044, 1.065]
e = [0.01412, 0.01463, 0.01423, 0.01509, 0.01462, 0.01459, 0.01458, 0.01257, 0.01114, 0.01019, 0.00979, 0.00944,
     0.00972, 0.01005, 0.01003, 0.00928, 0.00833, 0.00776, 0.00644, 0.00590, 0.00510]
SR = [0.251, 0.251, 0.240, 0.251, 0.260, 0.269, 0.259, 0.248, 0.247, 0.233, 0.22, 0.232, 0.220, 0.211, 0.251, 0.248,
      0.263, 0.262, 0.307, 0.353, 0.248]
SI = [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, -0.041, -0.053, -0.103, -0.146, -0.164, -0.206, -0.239, -0.256,
      -0.306, -0.321, -0.337, -0.331, -0.390, -0.498]
SS = [2.607, 2.764, 2.156, 2.161, 1.901, 1.814, 2.181, 2.432, 2.629, 2.702, 2.654, 2.480, 2.332, 2.233, 2.029, 1.589,
      0.966, 0.789, 1.037, 0.561, 0.225]
SSL = [-0.528, -0.551, -0.420, -0.431, -0.372, -0.360, -0.450, -0.506, -0.554, -0.575, -0.572, -0.540, -0.522, -0.509,
       -0.469, -0.379, -0.248, -0.221, -0.263, -0.169, -0.120]
CH = [0.293, 0.939, 1.499, 1.462, 1.280, 1.121, 0.852, 0.365, -0.207, -0.705, -1.144, -1.609, -2.023, -2.451, -3.243,
      -3.888, -4.783, -5.444, -5.839, -6.598, -6.752]
C1 = [1.111, 1.684, 2.061, 1.916, 1.669, 1.468, 1.172, 0.655, 0.071, -0.429, -0.866, -1.325, -1.732, -2.152, -2.923,
      -3.548, -4.410, -5.049, -5.431, -6.181, -6.347]
C2 = [1.344, 1.793, 2.135, 2.168, 2.085, 1.942, 1.683, 1.127, 0.515, -0.003, -0.449, -0.928, -1.349, -1.776, -2.542,
      -3.169, -4.039, -4.698, -5.089, -5.882, -6.051]
C3 = [1.355, 1.747, 2.031, 2.052, 2.001, 1.941, 1.808, 1.482, 0.934, 0.394, -0.111, -0.620, -1.066, -1.523, -2.327,
      -2.979, -3.871, -4.496, -4.893, -5.698, -5.873]
C4 = [1.420, 1.814, 2.082, 2.113, 2.030, 1.937, 1.770, 1.397, 0.955, 0.559, 0.188, -0.246, -0.643, -1.084, -1.936,
      -2.661, -3.64, -4.341, -4.758, -5.588, -5.798]
sigma = [0.604, 0.640, 0.694, 0.702, 0.692, 0.682, 0.670, 0.659, 0.653, 0.653, 0.652, 0.647, 0.653, 0.657, 0.660, 0.664,
         0.669, 0.671, 0.667, 0.647, 0.643]
tau_s = [0.321, 0.378, 0.420, 0.372, 0.324, 0.294, 0.284, 0.278, 0.272, 0.285, 0.29,  0.299, 0.289, 0.286, 0.277, 0.282, 0.300, 0.292, 0.274, 0.281, 0.296]
tau_i = [0.308, 0.343, 0.403, 0.367, 0.328, 0.289, 0.280, 0.271, 0.277, 0.296, 0.313, 0.329, 0.324, 0.328, 0.339, 0.352, 0.360, 0.356, 0.338, 0.307, 0.272]
tau_c = [0.303, 0.326, 0.342, 0.331, 0.312, 0.298, 0.300, 0.346, 0.338, 0.349, 0.351, 0.356, 0.348, 0.338, 0.313, 0.306, 0.283, 0.287, 0.278, 0.273, 0.275]


def Zhaoetal_2006_Sa(site, fault, im, periods=None):

    if im is 'PGA':
        periods = [0]

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

    if im == 'PGA':
        results = results[0]

    return results


def calculate_zhao(site, fault, period):
    M = fault.Mw
    R = site.Rrup

    closest_index = np.argmin(np.abs(period_list - period))
    i = int(closest_index)
    faultSR = 0
    faultSI = 0
    faultSS = 0
    faultSSL = 0
    if fault.tect_type == TectType.ACTIVE_SHALLOW and fault.faultstyle == FaultStyle.REVERSE:
        faultSR = 1
    elif fault.tect_type == TectType.SUBDUCTION_INTERFACE:
        faultSI = 1
    elif fault.tect_type == TectType.SUBDUCTION_SLAB:
        faultSS = 1
        faultSSL = 1
    siterock = 0
    siteSCI = 0
    siteSCII = 0
    siteSCIII = 0
    siteSCIV = 0
    if site.siteclass == SiteClass.HARDROCK:
        siterock = 1
    elif site.siteclass == SiteClass.ROCK:
        siteSCI = 1
    elif site.siteclass == SiteClass.HARDSOIL:
        siteSCII = 1
    elif site.siteclass == SiteClass.MEDIUMSOIL:
        siteSCIII = 1
    elif site.siteclass == SiteClass.SOFTSOIL:
        siteSCIV = 1
    h = fault.hdepth
    hc = 15
    R_star = R + c[i] * np.exp(d[i] * M)
    logSA = (a[i] * M + b[i] * R - np.log(R_star) + e[i] * max(min(h, 125), - hc, 0) + faultSR * SR[i] +
             faultSI * SI[i] + faultSS * SS[i] + faultSSL * SSL[i] * np.log(R) + siterock * CH[i] + siteSCI * C1[i] +
             siteSCII * C2[i] + siteSCIII * C3[i] + siteSCIV * C4[i])
    # convert to median in g
    SA = np.exp(logSA) / 981
    sigma_SA = determine_stdev(i, fault.tect_type)
    return SA, sigma_SA


def determine_stdev(i, tect_type):
    sigma_intra = get_tau(tect_type)[i]
    sigma_inter = sigma[i]
    sigma_total = np.sqrt(sigma_intra ** 2 + sigma_inter ** 2)
    sigma_SA = [sigma_total, sigma_inter, sigma_intra]
    return sigma_SA


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
