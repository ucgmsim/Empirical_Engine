
import numpy as np

from empirical.util.classdef import FaultStyle


def McVerry_2006_Sa(siteprop, faultprop, im=None, period=None):
    """
    puropse: provide the geometric mean and dispersion in the McVerry attenuation
    relationship for a given M,R and faulting style, soil conditions etc

    reference: McVerry GH, Zhao JX, Abrahamson NA, Somerville PG. New Zealand
    Accelerations Response Spectrum Attenuation Relations for Crustal and
    Subduction Zone Earthquakes.  Bulletin of the New Zealand Society of
    Earthquake Engineering. Vol 39, No 4. March 2006

    Inputvariables
    siteprop
       Rrup  -- Shortest distance from source to site (km) (i.e. Rrup)
       siteclass - 'A','B','C','D','E'; as per NZS1170.5
       rvol-- length in km of the part of the source to site distance in volcanic
              zone (not needed for slab event)

    faultprop
        Mw   -- Moment magnitude
        faultstyle   
           - crustal events - 'normal','reverse','oblique'
           - subduction events - 'interface','slab'
        hdepth  -- the centroid depth in km

    period-- period of vibration to compute attenuation for
              uses linear interpolation between actual parameter values

    Output Variables:
     SA           = median SA  (or PGA or PGV) (geometric mean)
     sigma_SA     = lognormal standard deviation in SA
                    sigma_SA[0] = total std
                    sigma_SA[1] = interevent std
                    sigma_SA[2] = intraevent std
    """

    # parameters - first column corresponds to the 'prime' values
    # fmt: off
    periods = np.array([-1.0, 0.0, 0.075, 0.1, 0.2, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
    C1 = [0.14274, 0.07713, 1.22050, 1.53365, 1.22565, 0.21124, -0.10541, -0.14260, -0.65968, -0.51404, -0.95399, -1.24167, -1.56570]
    C3AS = [0.0, 0.0, 0.03, 0.028, -0.0138, -0.036, -0.0518, -0.0635, -0.0862, -0.102, -0.12, -0.12, -0.17260]
    C4AS = -0.144
    C5 = [-0.00989, -0.00898, -0.00914, -0.00903, -0.00975, -0.01032, -0.00941, -0.00878, -0.00802, -0.00647, -0.00713, -0.00713, -0.00623]
    C6AS = 0.17
    C8 = [-0.68744, -0.73728, -0.93059, -0.96506, -0.75855, -0.52400, -0.50802, -0.52214, -0.47264, -0.58672, -0.49268, -0.49268, -0.52257]
    C10AS = [5.6, 5.6, 5.58, 5.5, 5.1, 4.8, 4.52, 4.3, 3.9, 3.7, 3.55, 3.55, 3.5]
    C11 = [8.57343, 8.08611, 8.69303, 9.30400, 10.41628, 9.21783, 8.0115, 7.87495, 7.26785, 6.98741, 6.77543, 6.48775, 5.05424]
    C12y = 1.414
    C13y = [0.0, 0.0, 0.0, -0.0011, -0.0027, -0.0036, -0.0043, -0.0048, -0.0057, -0.0064, -0.0073, -0.0073, -0.0089]
    C15 = [-2.552, -2.552, -2.707, -2.655, -2.528, -2.454, -2.401, -2.36, -2.286, -2.234, -2.16, -2.16, -2.033]
    C17 = [-2.56592, -2.49894, -2.55903, -2.61372, -2.70038, -2.47356, -2.30457, -2.31991, -2.28460, -2.28256, -2.27895, -2.27895, -2.05560]
    C18y = 1.7818
    C19y = 0.554
    C20 = [0.01545, 0.0159, 0.01821, 0.01737, 0.01531, 0.01304, 0.01426, 0.01277, 0.01055, 0.00927, 0.00748, 0.00748, -0.00273]
    C24 = [-0.49963, -0.43223, -0.52504, -0.61452, -0.65966, -0.56604, -0.33169, -0.24374, -0.01583, 0.02009, -0.07051, -0.07051, -0.23967]
    C29 = [0.27315, 0.38730, 0.27879, 0.28619, 0.34064, 0.53213, 0.63272, 0.58809, 0.50708, 0.33002, 0.07445, 0.07445, 0.09869]
    C30AS = [-0.23, -0.23, -0.28, -0.28, -0.245, -0.195, -0.16, -0.121, -0.05, 0.0, 0.04, 0.04, 0.04]
    C32 = 0.2
    C33AS = [0.26, 0.26, 0.26, 0.26, 0.26, 0.198, 0.154, 0.119, 0.057, 0.013, -0.049, -0.049, -0.156]
    C43 = [-0.33716, -0.31036, -0.49068, -0.46604, -0.31282, -0.07565, 0.17615, 0.34775, 0.72380, 0.89239, 0.77743, 0.77743, 0.60938]
    C46 = [-0.03255, -0.0325, -0.03441, -0.03594, -0.03823, -0.03535, -0.03354, -0.03211, -0.02857, -0.025, -0.02008, -0.02008, -0.01587]
    Sigma6 = [0.4871, 0.5099, 0.5297, 0.5401, 0.5599, 0.5456, 0.5556, 0.5658, 0.5611, 0.5573, 0.5419, 0.5419, 0.5809]
    Sigslope = [-0.1011, -0.0259, -0.0703, -0.0292, 0.0172, -0.0566, -0.1064, -0.1123, -0.0836, -0.0620, 0.0385, 0.0385, 0.1403]
    Tau = [0.2677, 0.2469, 0.3139, 0.3017, 0.2583, 0.1967, 0.1802, 0.1440, 0.1871, 0.2073, 0.2405, 0.2405, 0.2053]
    # fmt: on

    M = faultprop.Mw
    R = siteprop.Rrup
    # not sure about this because it is not documented
    if im == "PGA":
        period = -1
    elif im == "PGV":
        period = 0
    assert period <= (periods[-1] + 0.0001)

    # interpolate between periods if neccesary
    if not np.isclose(periods, period, atol=0.0001).any():
        p = np.argmin(periods < period)
        assert p > 0
        T_low = periods[p - 1]
        T_hi = periods[p]

        SA_low, sigma_SA_low = McVerry_2006_Sa(siteprop, faultprop, period=T_low)
        SA_high, sigma_SA_high = McVerry_2006_Sa(siteprop, faultprop, period=T_hi)

        sigma_SA_lh = np.array([sigma_SA_low, sigma_SA_high]).T
        if T_low > 0:
            # log interpolation
            x = np.log(T_low), np.log(T_hi)

            SA = np.exp(np.interp(np.log(period), x, (np.log(SA_low), np.log(SA_high))))
            sigma_SA = [np.interp(np.log(period), x, sigma_SA_lh[i]) for i in range(3)]
        else:
            # linear interpolation
            x = T_low, T_hi

            SA = interp1(period, x, (SA_low, SA_high))
            sigma_SA = [interp1(x, sigma_SA_lh[i], period) for i in range(3)]

        return SA, sigma_SA

    # Identify the period
    i = np.argmin(np.abs(periods - period))

    # site class
    delC = int(siteprop.siteclass.value == "C")
    delD = int(siteprop.siteclass.value == "D")

    # rvol, volcanic path term doesn't matter with subduction slab
    rvol = siteprop.rvol if faultprop.faultstyle != FaultStyle.SLAB else 0
    # CS, crustal event
    CS = faultprop.faultstyle in [FaultStyle.SLAB, FaultStyle.INTERFACE]

    if not CS:
        CN = -1 if faultprop.faultstyle is FaultStyle.NORMAL else 0
        CR = {
            FaultStyle.NORMAL: 0,
            FaultStyle.REVERSE: 1,
            FaultStyle.OBLIQUE: 0.5,
            FaultStyle.STRIKESLIP: 0,
        }[faultprop.faultstyle]

        def cpe(p):
            """
            crustal prediction equation
            """
            return np.exp(
                C1[p]
                + C4AS * (M - 6)
                + C3AS[p] * (8.5 - M) ** 2
                + C5[p] * R
                + (C8[p] + C6AS * (M - 6)) * np.log(np.sqrt(R ** 2 + C10AS[p] ** 2))
                + C46[p] * rvol
                + C32 * CN
                + C33AS[p] * CR
            )

        funca = cpe

    else:
        SI = int(faultprop.faultstyle is FaultStyle.INTERFACE)
        DS = int(faultprop.faultstyle is FaultStyle.SLAB)

        def suba(p):
            """
            subduction attenuation
            """
            return np.exp(
                C11[p]
                + (C12y + (C15[p] - C17[p]) * C19y) * (M - 6)
                + C13y[p] * (10 - M) ** 3
                + C17[p] * np.log(R + C18y * np.exp(C19y * M))
                + C20[p] * faultprop.hdepth
                + C24[p] * SI
                + C46[p] * rvol * (1 - DS)
            )

        funca = suba

    def cd(p):
        """
        <insert description>
        """
        return funca(p) * np.exp(
            C29[p] * delC + (C30AS[p] * np.log(funca(1) + 0.03) + C43[p]) * delD
        )

    Sa = cd(i) * cd(0) / cd(1)

    # standard deviation
    if M < 5:
        sig_intra = Sigma6[i] - Sigslope[i]
    elif M > 7:
        sig_intra = Sigma6[i] + Sigslope[i]
    else:
        sig_intra = Sigma6[i] + Sigslope[i] * (M - 6)

    # output
    sigma_SA = np.sqrt(sig_intra ** 2 + Tau[i] ** 2), sig_intra, Tau[i]
    return Sa, sigma_SA
