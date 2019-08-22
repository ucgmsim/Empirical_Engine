
import math

import numpy as np

# fmt: off
periods = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0, 0, -1])
PGAi = 21
# fmt: on


def CB_2014_nga(
    M,
    T,
    Rrup,
    Rjb,
    Rx,
    W=None,
    Ztor=None,
    Zbot=None,
    delta=None,
    lamda=90,
    f_hw=0,
    vs30=500,
    Z25=None,
    Zhyp=None,
    region=0,
):

    """
    Campbell and Bozorgnia 2014 ground motion prediciton model. Citation for
    the model:

    Campbell, K. W., and Bozorgnia, Y. (2014). ?NGA-West2 Ground Motion Model 
    for the Average Horizontal Components of PGA, PGV, and 5% Damped Linear 
    Acceleration Response Spectra.? Earthquake Spectra, 30(3), 1087?1115.

    coded by Yue Hua,  4/22/14
        Stanford University, yuehua@stanford.edu
    Modified 4/5/2015 by Jack Baker to fix a few small bugs

    Provides ground-mtion prediction equations for computing medians and
    standard deviations of average horizontal components of PGA, PGV and 5%
    damped linear pseudo-absolute aceeleration response spectra

    Input Variables
    M             = Magnitude
    T             = Period (sec)
                    Use 1000 for output the array of Sa with period
    Rrup          = Closest distance coseismic rupture (km)
    Rjb           = Joyner-Boore distance (km)
    Rx            = Closest distance to the surface projection of the
                    coseismic fault rupture plane
    W             = down-dip width of the fault rupture plane
                    if unknown, input: 999
    Ztor          = Depth to the top of coseismic rupture (km)
                    if unknown, input: 999
    Zbot          = Depth to the bottom of the seismogenic crust
                needed only when W is unknown
    delta         = average dip of the rupture place (degree)
    lamda        = rake angle (degree) - average angle of slip measured in
                    the plance of rupture
    f_hw           = hanging wall effect
                = 1 for including
                = 0 for excluding
    vs30          = shear wave velocity averaged over top 30 m (m/s)
    Z25           = Depth to the 2.5 km/s shear-wave velocity horizon (km)
                    None if in California or Japan and Z2.5 is unknow
    Zhyp          = Hypocentral depth of the earthquake measured from sea level
                    if unknown, input: 999
    region        = 0 for global (incl. Taiwan)
                = 1 for California
                = 2 for Japan
                = 3 for China or Turkey
                = 4 for Italy
    Output Variables
    Sa            = Median spectral acceleration prediction
    sigma         = logarithmic standard deviation of spectral acceleration
                    prediction
    """

    # style of faulting
    f_rv = int(30 < lamda < 150)
    f_nm = int(-150 < lamda < -30)

    # Ztor is unknown
    if Ztor is None:
        if f_rv:
            Ztor = max(2.704 - 1.226 * max(M - 5.849, 0), 0) ** 2
        else:
            Ztor = max(2.673 - 1.136 * max(M - 4.970, 0), 0) ** 2

    if W is None:
        if f_rv:
            Ztori = max(2.704 - 1.226 * max(M - 5.849, 0), 0) ** 2
        else:
            Ztori = max(2.673 - 1.136 * max(M - 4.970, 0), 0) ** 2
        try:
            W = min(
                math.sqrt(10 ** ((M - 4.07) / 0.98)),
                (Zbot - Ztori) / math.sin(math.pi / 180 * delta),
            )
        except ZeroDivisionError:
            W = math.sqrt(10 ** ((M - 4.07) / 0.98))
        Zhyp = 9

    elif Zhyp is None:
        fdZM = min(-4.317 + 0.984 * M, 2.325)
        fdZD = 0.0445 * (min(delta, 40) - 40)

        if f_rv:
            Ztori = max(2.704 - 1.226 * max(M - 5.849, 0), 0) ** 2
        else:
            Ztori = max(2.673 - 1.136 * max(M - 4.970, 0), 0) ** 2

        # depth to bottom of rupture plane
        Zbor = Ztori + W * math.sin(math.pi / 180 * delta)
        try:
            d_Z = math.exp(min(fdZM + fdZD, math.log(0.9 * (Zbor - Ztori))))
        except ValueError:
            # Zbor == Ztori
            d_Z = 0
        Zhyp = d_Z + Ztori

    def sa_sigma(period_i):
        """
        Return Sa and sigma for given period index.
        """
        return CB_2014_nga_sub(
            M,
            period_i,
            Rrup,
            Rjb,
            Rx,
            W,
            Ztor,
            Zbot,
            delta,
            f_hw,
            vs30,
            Z25,
            Zhyp,
            lamda,
            f_rv,
            f_nm,
            region,
        )

    # compute Sa and sigma with pre-defined period
    if T is None:
        periods_out = periods[:-2]
        Sa = np.zeros(len(periods_out))
        sigma = np.zeros(len(periods_out))

        for ipT in range(len(periods_out)):
            Sa[ipT], sigma[ipT] = sa_sigma(ipT)
            PGA = sa_sigma(PGAi)[0]
            if Sa[ipT] < PGA and periods_out[ipT] < 0.25:
                Sa[ipT] = PGA
        return Sa, sigma, periods_out

    # compute Sa and sigma with user-defined period
    try:
        T[0]
    except TypeError:
        T = [T]
    Sa = np.zeros(len(T))
    sigma = np.zeros(len(T))
    for i, Ti in enumerate(T):
        if not np.isclose(periods, Ti, atol=0.0001).any():
            # user defined period requires interpolation
            ip_high = np.argmin(periods < Ti)
            ip_low = ip_high - 1

            Sa_low, sigma_low = sa_sigma(ip_low)
            Sa_high, sigma_high = sa_sigma(ip_high)
            PGA = sa_sigma(PGAi)[0]
            x = math.log(periods[ip_low]), math.log(periods[ip_high])
            Y_sa = math.log(Sa_low), math.log(Sa_high)
            Y_sigma = np.array([sigma_low, sigma_high]).T
            Sa[i] = math.exp(np.interp(math.log(Ti), x, Y_sa))
            sigma[i] = np.interp(math.log(Ti), x, Y_sigma)
            if Sa[i] < PGA and Ti < 0.25:
                Sa[i] = PGA
        else:
            ip_T = np.argmin(np.abs(periods - Ti))
            Sa[i], sigma[i] = sa_sigma(ip_T)
            if ip_T != PGAi:
                PGA = sa_sigma(PGAi)[0]
                if Sa[i] < PGA and Ti < 0.25:
                    Sa[i] = PGA

    if not i:
        return Sa[0], sigma[0]
    return Sa, sigma


def CB_2014_nga_sub(
    M,
    ip,
    Rrup,
    Rjb,
    Rx,
    W,
    Ztor,
    Zbot,
    delta,
    f_hw,
    vs30,
    Z25in,
    Zhyp,
    lamda,
    f_rv,
    f_nm,
    region,
    A1100=None,
):

    # fmt: off
    c0 = [-4.365, -4.348, -4.024, -3.479, -3.293, -3.666, -4.866, -5.411, -5.962, -6.403, -7.566, -8.379, -9.841, -11.011, -12.469, -12.969, -13.306, -14.02, -14.558, -15.509, -15.975, -4.416, -2.895]
    c1 = [0.977, 0.976, 0.931, 0.887, 0.902, 0.993, 1.267, 1.366, 1.458, 1.528, 1.739, 1.872, 2.021, 2.180, 2.270, 2.271, 2.150, 2.132, 2.116, 2.223, 2.132, 0.984, 1.510]
    c2 = [0.533, 0.549, 0.628, 0.674, 0.726, 0.698, 0.510, 0.447, 0.274, 0.193, -0.020, -0.121, -0.042, -0.069, 0.047, 0.149, 0.368, 0.726, 1.027, 0.169, 0.367, 0.537, 0.270]
    c3 = [-1.485, -1.488, -1.494, -1.388, -1.469, -1.572, -1.669, -1.750, -1.711, -1.770, -1.594, -1.577, -1.757, -1.707, -1.621, -1.512, -1.315, -1.506, -1.721, -0.756, -0.800, -1.499, -1.299]
    c4 = [-0.499, -0.501, -0.517, -0.615, -0.596, -0.536, -0.490, -0.451, -0.404, -0.321, -0.426, -0.440, -0.443, -0.527, -0.630, -0.768, -0.890, -0.885, -0.878, -1.077, -1.282, -0.496, -0.453]
    c5 = [-2.773, -2.772, -2.782, -2.791, -2.745, -2.633, -2.458, -2.421, -2.392, -2.376, -2.303, -2.296, -2.232, -2.158, -2.063, -2.104, -2.051, -1.986, -2.021, -2.179, -2.244, -2.773, -2.466]
    c6 = [0.248, 0.247, 0.246, 0.240, 0.227, 0.210, 0.183, 0.182, 0.189, 0.195, 0.185, 0.186, 0.186, 0.169, 0.158, 0.158, 0.148, 0.135, 0.140, 0.178, 0.194, 0.248, 0.204]
    c7 = [6.753, 6.502, 6.291, 6.317, 6.861, 7.294, 8.031, 8.385, 7.534, 6.990, 7.012, 6.902, 5.522, 5.650, 5.795, 6.632, 6.759, 7.978, 8.538, 8.468, 6.564, 6.768, 5.837]
    c8 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    c9 = [-0.214, -0.208, -0.213, -0.244, -0.266, -0.229, -0.211, -0.163, -0.150, -0.131, -0.159, -0.153, -0.090, -0.105, -0.058, -0.028, 0, 0, 0, 0, 0, -0.212, -0.168]
    c10 = [0.720, 0.730, 0.759, 0.826, 0.815, 0.831, 0.749, 0.764, 0.716, 0.737, 0.738, 0.718, 0.795, 0.556, 0.480, 0.401, 0.206, 0.105, 0, 0, 0, 0.720, 0.305]
    c11 = [1.094, 1.149, 1.290, 1.449, 1.535, 1.615, 1.877, 2.069, 2.205, 2.306, 2.398, 2.355, 1.995, 1.447, 0.330, -0.514, -0.848, -0.793, -0.748, -0.664, -0.576, 1.090, 1.713]
    c12 = [2.191, 2.189, 2.164, 2.138, 2.446, 2.969, 3.544, 3.707, 3.343, 3.334, 3.544, 3.016, 2.616, 2.470, 2.108, 1.327, 0.601, 0.568, 0.356, 0.075, -0.027, 2.186, 2.602]
    c13 = [1.416, 1.453, 1.476, 1.549, 1.772, 1.916, 2.161, 2.465, 2.766, 3.011, 3.203, 3.333, 3.054, 2.562, 1.453, 0.657, 0.367, 0.306, 0.268, 0.374, 0.297, 1.420, 2.457]
    c14 = [-0.0070, -0.0167, -0.0422, -0.0663, -0.0794, -0.0294, 0.0642, 0.0968, 0.1441, 0.1597, 0.1410, 0.1474, 0.1764, 0.2593, 0.2881, 0.3112, 0.3478, 0.3747, 0.3382, 0.3754, 0.3506, -0.0064, 0.1060]
    c15 = [-0.207, -0.199, -0.202, -0.339, -0.404, -0.416, -0.407, -0.311, -0.172, -0.084, 0.085, 0.233, 0.411, 0.479, 0.566, 0.562, 0.534, 0.522, 0.477, 0.321, 0.174, -0.202, 0.332]
    c16 = [0.390, 0.387, 0.378, 0.295, 0.322, 0.384, 0.417, 0.404, 0.466, 0.528, 0.540, 0.638, 0.776, 0.771, 0.748, 0.763, 0.686, 0.691, 0.670, 0.757, 0.621, 0.393, 0.585]
    c17 = [0.0981, 0.1009, 0.1095, 0.1226, 0.1165, 0.0998, 0.0760, 0.0571, 0.0437, 0.0323, 0.0209, 0.0092, -0.0082, -0.0131, -0.0187, -0.0258, -0.0311, -0.0413, -0.0281, -0.0205, 0.0009, 0.0977, 0.0517]
    c18 = [0.0334, 0.0327, 0.0331, 0.0270, 0.0288, 0.0325, 0.0388, 0.0437, 0.0463, 0.0508, 0.0432, 0.0405, 0.0420, 0.0426, 0.0380, 0.0252, 0.0236, 0.0102, 0.0034, 0.0050, 0.0099, 0.0333, 0.0327]
    c19 = [0.00755, 0.00759, 0.00790, 0.00803, 0.00811, 0.00744, 0.00716, 0.00688, 0.00556, 0.00458, 0.00401, 0.00388, 0.00420, 0.00409, 0.00424, 0.00448, 0.00345, 0.00603, 0.00805, 0.00280, 0.00458, 0.00757, 0.00613]
    c20 = [-0.0055, -0.0055, -0.0057, -0.0063, -0.0070, -0.0073, -0.0069, -0.0060, -0.0055, -0.0049, -0.0037, -0.0027, -0.0016, -0.0006, 0, 0, 0, 0, 0, 0, 0, -0.0055, -0.0017]
    Dc20 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    Dc20_JI = [-0.0035, -0.0035, -0.0034, -0.0037, -0.0037, -0.0034, -0.0030, -0.0031, -0.0033, -0.0035, -0.0034, -0.0034, -0.0032, -0.0030, -0.0019, -0.0005, 0, 0, 0, 0, 0, -0.0035, -0.0006]
    Dc20_CH = [0.0036, 0.0036, 0.0037, 0.0040, 0.0039, 0.0042, 0.0042, 0.0041, 0.0036, 0.0031, 0.0028, 0.0025, 0.0016, 0.0006, 0, 0, 0, 0, 0, 0, 0, 0.0036, 0.0017]
    a2 = [0.168, 0.166, 0.167, 0.173, 0.198, 0.174, 0.198, 0.204, 0.185, 0.164, 0.160, 0.184, 0.216, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.596, 0.167, 0.596]
    h1 = [0.242, 0.244, 0.246, 0.251, 0.260, 0.259, 0.254, 0.237, 0.206, 0.210, 0.226, 0.217, 0.154, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.117, 0.241, 0.117]
    h2 = [1.471, 1.467, 1.467, 1.449, 1.435, 1.449, 1.461, 1.484, 1.581, 1.586, 1.544, 1.554, 1.626, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.616, 1.474, 1.616]
    h3 = [-0.714, -0.711, -0.713, -0.701, -0.695, -0.708, -0.715, -0.721, -0.787, -0.795, -0.770, -0.770, -0.780, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.733, -0.715, -0.733]
    h4 = [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
    h5 = [-0.336, -0.339, -0.338, -0.338, -0.347, -0.391, -0.449, -0.393, -0.339, -0.447, -0.525, -0.407, -0.371, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.128, -0.337, -0.128]
    h6 = [-0.270, -0.263, -0.259, -0.263, -0.219, -0.201, -0.099, -0.198, -0.210, -0.121, -0.086, -0.281, -0.285, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.756, -0.270, -0.756]
    k1 = [865, 865, 908, 1054, 1086, 1032, 878, 748, 654, 587, 503, 457, 410, 400, 400, 400, 400, 400, 400, 400, 400, 865, 400]
    k2 = [-1.186, -1.219, -1.273, -1.346, -1.471, -1.624, -1.931, -2.188, -2.381, -2.518, -2.657, -2.669, -2.401, -1.955, -1.025, -0.299, 0.000, 0.000, 0.000, 0.000, 0.000, -1.186, -1.955]
    k3 = [1.839, 1.840, 1.841, 1.843, 1.845, 1.847, 1.852, 1.856, 1.861, 1.865, 1.874, 1.883, 1.906, 1.929, 1.974, 2.019, 2.110, 2.200, 2.291, 2.517, 2.744, 1.839, 1.929]
    c = 1.88
    n = 1.18
    f1 = [0.734, 0.738, 0.747, 0.777, 0.782, 0.769, 0.769, 0.761, 0.744, 0.727, 0.690, 0.663, 0.606, 0.579, 0.541, 0.529, 0.527, 0.521, 0.502, 0.457, 0.441, 0.734, 0.655]
    f2 = [0.492, 0.496, 0.503, 0.520, 0.535, 0.543, 0.543, 0.552, 0.545, 0.568, 0.593, 0.611, 0.633, 0.628, 0.603, 0.588, 0.578, 0.559, 0.551, 0.546, 0.543, 0.492, 0.494]
    t1 = [0.404, 0.417, 0.446, 0.508, 0.504, 0.445, 0.382, 0.339, 0.340, 0.340, 0.356, 0.379, 0.430, 0.470, 0.497, 0.499, 0.500, 0.543, 0.534, 0.523, 0.466, 0.409, 0.317]
    t2 = [0.325, 0.326, 0.344, 0.377, 0.418, 0.426, 0.387, 0.338, 0.316, 0.300, 0.264, 0.263, 0.326, 0.353, 0.399, 0.400, 0.417, 0.393, 0.421, 0.438, 0.438, 0.322, 0.297]
    flnAF = [0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300, 0.300]
    rlnPGA_lnY = [1.000, 0.998, 0.986, 0.938, 0.887, 0.870, 0.876, 0.870, 0.850, 0.819, 0.743, 0.684, 0.562, 0.467, 0.364, 0.298, 0.234, 0.202, 0.184, 0.176, 0.154, 1.000, 0.684]
    # fmt: on

    # many areas Japan specific
    japan = region == 2
    if japan or region == 4:
        # Japan/Italy
        Dc20 = Dc20_JI
    elif region == 3:
        # China
        Dc20 = Dc20_CH

    # Z2.5
    if Z25in is None:
        if not japan:
            Z25 = math.exp(7.089 - 1.144 * math.log(vs30))
            Z25A = math.exp(7.089 - 1.144 * math.log(1100))
        else:
            Z25 = math.exp(5.359 - 1.102 * math.log(vs30))
            Z25A = math.exp(5.359 - 1.102 * math.log(1100))
    else:
        # assign Z2.5 from user input into Z25 and calc Z25A for vs30=1100m/s
        Z25 = Z25in
        if not japan:
            Z25A = math.exp(7.089 - 1.144 * math.log(1100))
        else:
            Z25A = math.exp(5.359 - 1.102 * math.log(1100))

    # magnitude dependence
    fmag = (
        c0[ip]
        + c1[ip] * M
        + c2[ip] * max(0, (M - 4.5))
        + c3[ip] * max(0, (M - 5.5))
        + c4[ip] * max(0, (M - 6.5))
    )
    # geometric attenuation term
    fdis = (c5[ip] + c6[ip] * M) * math.log(math.sqrt(Rrup ** 2 + c7[ip] ** 2))

    # style of faulting
    F_fltm = min(1, max(0, M - 4.5))
    fflt = ((c8[ip] * f_rv) + (c9[ip] * f_nm)) * F_fltm

    # hanging-wall effects
    def hw_comp():
        if not f_hw:
            return 0
        R1 = W * math.cos(math.pi / 180 * delta)

        if Rx < R1:
            f_hngRx = h1[ip] + h2[ip] * (Rx / R1) + h3[ip] * (Rx / R1) ** 2
        else:
            R2 = 62 * M - 350
            f_hngRx = max(
                h4[ip]
                + h5[ip] * ((Rx - R1) / (R2 - R1))
                + h6[ip] * ((Rx - R1) / (R2 - R1)) ** 2,
                0,
            )

        try:
            f_hngRup = (Rrup - Rjb) / Rrup
        except ZeroDivisionError:
            f_hngRup = 1

        if M <= 5.5:
            f_hngM = 0
        elif M <= 6.5:
            f_hngM = (M - 5.5) * (1 + a2[ip] * (M - 6.5))
        else:
            f_hngM = 1 + a2[ip] * (M - 6.5)

        f_hngZ = max(0, 1 - 0.06 * Ztor)

        f_hngdelta = (90 - delta) / 45

        return c10[ip] * f_hngRx * f_hngRup * f_hngM * f_hngZ * f_hngdelta

    fhng = hw_comp()

    # site conditions

    if vs30 <= k1[ip]:
        if A1100 is None:
            A1100 = CB_2014_nga(
                M,
                0,
                Rrup,
                Rjb,
                Rx,
                W=W,
                Ztor=Ztor,
                Zbot=Zbot,
                delta=delta,
                lamda=lamda,
                f_hw=f_hw,
                vs30=1100,
                Z25=Z25A,
                Zhyp=Zhyp,
                region=region,
            )[0]
        f_siteG = c11[ip] * math.log(vs30 / k1[ip]) + k2[ip] * (
            math.log(A1100 + c * (vs30 / k1[ip]) ** n) - math.log(A1100 + c)
        )
    elif vs30 > k1[ip]:
        f_siteG = (c11[ip] + k2[ip] * n) * math.log(vs30 / k1[ip])

    if vs30 <= 200:
        f_siteJ = (
            (c12[ip] + k2[ip] * n)
            * (math.log(vs30 / k1[ip]) - math.log(200 / k1[ip]))
            * japan
        )
    else:
        f_siteJ = (c13[ip] + k2[ip] * n) * math.log(vs30 / k1[ip]) * japan

    fsite = f_siteG + f_siteJ
    # basin response term - sediment effects

    if Z25 < 1:
        fsed = (c14[ip] + c15[ip] * japan) * (Z25 - 1)
    elif Z25 <= 3:
        fsed = 0
    else:
        fsed = c16[ip] * k3[ip] * math.exp(-0.75) * (1 - math.exp(-0.25 * (Z25 - 3)))

    # hypocenteral depth term
    f_hypH = min(13, max(0, Zhyp - 7))

    if M <= 5.5:
        f_hypM = c17[ip]
    elif M <= 6.5:
        f_hypM = c17[ip] + (c18[ip] - c17[ip]) * (M - 5.5)
    else:
        f_hypM = c18[ip]

    fhyp = f_hypH * f_hypM

    # fault dip term
    if M <= 4.5:
        f_dip = c19[ip] * delta
    elif M <= 5.5:
        f_dip = c19[ip] * (5.5 - M) * delta
    else:
        f_dip = 0

    # anelastic attenuation term
    f_atn = (c20[ip] + Dc20[ip]) * (Rrup - 80) if Rrup > 80 else 0

    # median value
    Sa = math.exp(fmag + fdis + fflt + fhng + fsite + fsed + fhyp + f_dip + f_atn)

    # standard deviation computations
    if M <= 4.5:
        tau_lny = t1[ip]
        tau_lnPGA = t1[PGAi]
        phi_lny = f1[ip]
        phi_lnPGA = f1[PGAi]
    elif M < 5.5:
        tau_lny = t2[ip] + (t1[ip] - t2[ip]) * (5.5 - M)
        tau_lnPGA = t2[PGAi] + (t1[PGAi] - t2[PGAi]) * (5.5 - M)
        phi_lny = f2[ip] + (f1[ip] - f2[ip]) * (5.5 - M)
        phi_lnPGA = f2[PGAi] + (f1[PGAi] - f2[PGAi]) * (5.5 - M)
    else:
        tau_lny = t2[ip]
        tau_lnPGA = t2[PGAi]
        phi_lny = f2[ip]
        phi_lnPGA = f2[PGAi]

    tau_lnyB = tau_lny
    tau_lnPGAB = tau_lnPGA
    phi_lnyB = math.sqrt(phi_lny ** 2 - flnAF[ip] ** 2)
    phi_lnPGAB = math.sqrt(phi_lnPGA ** 2 - flnAF[ip] ** 2)

    if vs30 < k1[ip]:
        alpha = (
            k2[ip]
            * A1100
            * ((A1100 + c * (vs30 / k1[ip]) ** n) ** (-1) - (A1100 + c) ** (-1))
        )
    else:
        alpha = 0

    tau = math.sqrt(
        tau_lnyB ** 2
        + alpha ** 2 * tau_lnPGAB ** 2
        + 2 * alpha * rlnPGA_lnY[ip] * tau_lnyB * tau_lnPGAB
    )
    phi = math.sqrt(
        phi_lnyB ** 2
        + flnAF[ip] ** 2
        + alpha ** 2 * phi_lnPGAB ** 2
        + 2 * alpha * rlnPGA_lnY[ip] * phi_lnyB * phi_lnPGAB
    )

    sigma = math.sqrt(tau ** 2 + phi ** 2)

    return Sa, sigma
