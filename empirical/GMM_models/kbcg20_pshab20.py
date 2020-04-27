#!/usr/bin/env python

### #############################################################
### #############################################################
### --------------------------------------------------------------------
### --------------------------------------------------------------------
### -- NGA-Subduction:
###                  Ground-Motion Characterization Tool
###                  Python Version
### --------------------------------------------------------------------
###                             April 2020
### --------------------------------------------------------------------
### Ground-Motion Model Tool developed by
###            Silvia Mazzoni
###                smazzoni@ucla.edu
###            B. John Garrick Institute for the Risk Sciences
###                https://www.risksciences.ucla.edu/nhr3/gmtools/home
### --------------------------------------------------------------------
### Source:
###  KBCG20:
###    Kuehn, Bozorgnia, Campbell, and Gregor,
###         ?Partially Nonergodic Ground-Motion Model for Subduction Regions using NGA-Subduction Database,?
###         Report xx/2020, Pacific Earthquake Engineering Research Center, UC Berkeley
###  PSHAB20:
###    Parker, G.A., Stewart, J.P., Hassani, B., Atkinson, G.M., and Boore, D.M. (2020).
###         "NGA-Subduction Global Ground Motion Models with Regional Adjustment Factors."
###         Report xx/2020, Pacific Earthquake Engineering Research Center, UC Berkeley
### --------------------------------------------------------------------
### Instruction:
###   0. Scroll to the bottom of the script, to the USER INPUT section.
###   1. Enter the values of the user-input parameters.
###       Input ranges are given with each input parameters.
###   2. Run full script
###   3. Wait while script runs
###   4. Result values are printed in Command Window (you may copy and
###   paste them elsewhere)
### --------------------------------------------------------------------
###   Notes:
###	      This implementation was translated from an R script, where array indices start at 1
###           to minimize error, a place-holder was used at the zero index in this version
###       The Median and Sigma models are described in the reports.
###       Damping Ratio = 5#
###         You may speed up the program by managing the coefficients
###         tables more efficiently.
###       The program computes the geometric mean of the median ground-motion models,
###           and the SRSS of the aleatory variability and epistemic uncertainty.
###
###       Weighted-Average Model:
###             Median PSA (g) = exp((ln(PSHAB20) * W_PSHAB20 + ln(KBCG20) * W_KBCG20) / (W_PSHAB20 + W_KBCG20))
###             Median +/- N*Sigmatotal = exp(ln(MedianPSA) +/- N*Sigmatotal)
###             Sigmatotal = sqr(W_PSHAB20 * PSHAB20SigmaTotal * PSHAB20SigmaTotal + W_KBCG20 * KBCG20SigmaTotal * KBCG20SigmaTotal) / (W_PSHAB20 + W_KBCG20)
###       KBCG20 Sigmaaleatory = sqr(Phi * Phi + Tau * Tau)
###       KBCG20 Sigmaepistemic = stdev(PSAposteriori)
###       KBCG20 Sigmatotal = sqr(SigmaAleatoryKBCG20**2 + SigmaEpistemicKBCG20**2)
###       PSHAB20 Sigmaaleatory = sqr(PhiTot * PhiTot + Tau * Tau)
###       PSHAB20 ftotal = sqr(phiSquared + deltaVar)
###       PSHAB20 Sigmatotal = sqr(SigmaAleatoryPSHAB20**2 + SigmaEpistemicKBCG20**2)
### --------------------------------------------------------------------
###
### #############################################################
### #############################################################


import math

import numpy as np
import pandas as pd

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)


def getTvalue(thisTin):
    # code developed and written by Silvia Mazzoni
    #       smazzoni@ucla.edu, April 2020
    if type(thisTin) == str:
        if thisTin.lower() == "pga".lower():
            return 0
        elif thisTin.lower() == "pgv".lower():
            return -1
        else:
            return float(thisTin)
    else:
        return thisTin


# logistc hinge def
def loghinge(x, x0, a, b0, b1, delta):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    loghingeValue = (
        a + b0 * (x - x0) + (b1 - b0) * delta * math.log(1 + math.exp((x - x0) / delta))
    )
    return loghingeValue


# interpolation of adjustment to magnitude break point
def interp_dmb(period):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    # ap = approxfun(c(math.log(0.01), math.log[1], math.log[3], math.log(10)), c(0, 0, -0.4, -0.4), rule = 2)

    Aarray = [0.01, 1, 3, 10]
    Carray = [0, 0, -0.4, -0.4]
    if period == 0:
        #   interp_dmb = (ap(math.log(0.01)))
        interp_dmbValue = interpolateArray(
            0.01, Aarray, Carray, "log", "linear", "constant"
        )
    elif period == -1:
        #   interp_dmb = (ap(log(0.01)))
        interp_dmbValue = 0
    else:
        #   interp_dmb = (ap(log(period)))
        interp_dmbValue = interpolateArray(
            period, Aarray, Carray, "log", "linear", "constant"
        )

    return interp_dmbValue


def interp_k1k2(period):
    """
    Interpolation of k1/k2 (values taken from Campbell and Bozorgnia (2014).
    
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    """
    # fmt: off
    periods = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]
    k1 = [865.0, 865.0, 865.0, 908.0, 1054.0, 1086.0, 1032.0, 878.0, 748.0, 654.0, 587.0, 503.0, 457.0, 410.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0, 400.0]
    k2 = [-1.186, -1.186, -1.219, -1.273, -1.346, -1.471, -1.624, -1.931, -2.188, -2.381, -2.518, -2.657, -2.669, -2.401, -1.955, -1.025, -0.299, 0.0, 0.0, 0.0, 0.0, 0.0]
    # fmt: on

    if period == 0:
        # PGA
        ap_k1 = interpolateArray(0.005, periods, k1, "log", "linear", "constant")
        ap_k2 = interpolateArray(0.005, periods, k2, "log", "linear", "constant")
    elif period == -1:
        # PGV
        ap_k1 = 400.0
        ap_k2 = -1.995
    else:
        ap_k1 = interpolateArray(period, periods, k1, "log", "linear", "constant")
        ap_k2 = interpolateArray(period, periods, k2, "log", "linear", "constant")

    # inserted a spacer for zero-starting index
    interp_k1k2Value = [-999, ap_k1, ap_k2]

    return interp_k1k2Value


def calc_z_from_Vs30(Vs30, coeffs):
    """
     funcation to calculate Z_1/Z2pt5 from Vs30
    
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    """

    return coeffs[1] + (coeffs[2] - coeffs[1]) * math.exp(
        (math.log(Vs30) - coeffs[3]) / coeffs[4]
    ) / (1 + math.exp((math.log(Vs30) - coeffs[3]) / coeffs[4]))


def interp_dmb(period):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    # ap = approxfun(c(log(0.01), math.log(1), math.log[3], math.log(10)), c(0, 0, -0.4, -0.4), rule = 2)

    Aarray = [0.01, 1, 3, 10]
    Carray = [0, 0, -0.4, -0.4]
    if period == 0:
        interp_dmbValue = interpolateArray(
            0.01, Aarray, Carray, "log", "linear", "constant"
        )
    elif period == -1:
        interp_dmbValue = 0
    else:
        interp_dmbValue = interpolateArray(
            period, Aarray, Carray, "log", "linear", "constant"
        )

    return interp_dmbValue


def interpolateArray(
    Xpoint,
    Xlist,
    Ylist,
    XinterpType,
    YinterpType,
    extrapolateType,
    XinterpMin=-1e16,
    XinterpMax=1e16,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    interpolateArrayValue = -999

    start0 = -1
    # start-index 0
    for irow in range(0, len(Xlist)):
        this = Xlist[irow]
        if Xpoint == this:
            return Ylist[irow]
        if this >= XinterpMin and this <= XinterpMax:
            # not sure why this condition is needed, and start0 before loop, end0 after
            if start0 < 0:
                start0 = irow
            end0 = irow
        else:
            print("is this even possible?")

    iend = irow
    istart = start0
    Xstart = Xlist[start0]
    Xend = Xlist[end0]
    if (Xend - Xpoint) * (Xstart - Xpoint) <= 0:
        for i in range(istart, iend):
            thisx = Xlist[i]
            nextX = Xlist[i + 1]
            if (Xpoint - thisx) * (Xpoint - nextX) <= 0:
                l1 = i
                l2 = i + 1
                break
    else:
        # extrapolating
        if (Xend - Xstart) * (Xend - Xpoint) > 0:
            l1 = istart
            if extrapolateType.lower().startswith("extra"):
                l2 = istart + 1
            else:
                l2 = istart
        else:
            l1 = iend
            if extrapolateType.lower().startswith("extra"):
                l2 = iend - 1
            else:
                interpolateArrayValue = Ylist[end0]
                return interpolateArrayValue

    x0 = Xlist[l1]
    x1 = Xlist[l2]
    y0 = Ylist[l1]
    y1 = Ylist[l2]
    hereX = Xpoint
    if XinterpType.lower().startswith("log"):
        if x0 <= 0:
            x0 = 0.000000001
        if x1 <= 0:
            x1 = 0.000000002
        if hereX <= 0:
            hereX = 0.000000001
        x0 = math.log(x0)
        x1 = math.log(x1)
        hereX = math.log(hereX)
    if YinterpType.lower().startswith("log"):
        if y0 <= 0:
            y0 = 0.000000001
        if y1 <= 0:
            y1 = 0.000000001
        y0 = math.log(y0)
        y1 = math.log(y1)

    interpolateArrayValue = y0 + (y1 - y0) * (hereX - x0) / (x1 - x0)

    if YinterpType.lower().startswith("log"):
        return math.exp(interpolateArrayValue)
    return interpolateArrayValue


### def to calculate median prediction

# The following def calculates the median prediction of KBCG20, as described in chapter 4 of the PEER report.
# It takes as input the predictor variables, as well as a set of coefficients.
# Later, we define a def that acts as a wrapper around this def, and will take as input period, region index, and  select the appropriate coefficients and pass them on.
#
# The inputs are

# `m`: moment magnitude
# `rlist`: a vector of length 3, which contains the distance in subregion 1,2,3 relative to volcanic arc; `rlist = c(R1,R2,R3)`. Typically, `R1=R3=0` (corresponding to forearc).
# `ztor`: depth to top of rupture in km.
# `EventType`: flag for interface (`EventType = 0`) and intraslab (`EventType = 1`). Must be 0 or 1.
# `Vs30`: $V_{S30}$ n m/s.
# `fx`: arc crossing flag. Must be 0 or 1.
# `delta_ln_z`: difference between natural log of observed Z1.0/Z2.5 value and reference Z1.0/Z2.5 from $V_{S30}$.
# `coeffs`: vector containing coefficients needed to calculate median prediction.
# `coeffs_attn`: vector of length 6 to calcualte anelastic attenuation.
# `mbreak` and `zbreak`: magnitude and depth scaling break point
# `k1` and `k2`: parameters needed for site amplification
# `nft1` and `nft2`: coefficients needed for pseudo-depth term.
# `pgarock`: median pga prediction at $V_{S30} = 1100$

# def to calculate median prediction


def KBCG20_med(
    M,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    EventType,
    Vs30,
    delta_ln_z,
    coeffs,
    coeffs_attn,
    coeffs_z,
    mbreak,
    zbreak,
    k1,
    k2,
    nft1,
    nft2,
    pgarock,
    region,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    #  x = "KBCG20_med"

    if region == 0:
        distR1 = AlphaBackarc * Rrup
        distR3 = 0
    elif region == 1:
        distR1 = 0
        distR3 = 0
    elif region == 2:
        distR1 = 0
        distR3 = 0
    elif region == 3:
        distR1 = AlphaBackarc * Rrup
        distR3 = 0
    elif region == 4:
        distR1 = AlphaBackarc * Rrup
        distR3 = AlphaNankai * Rrup
    elif region == 5:
        distR1 = 0
        distR3 = 0
    elif region == 6:
        distR1 = AlphaBackarc * Rrup
        distR3 = 0
    elif region == 7:
        distR1 = 0
        distR3 = 0

    distR2 = Rrup - distR1 - distR3
    theta10 = 0
    vsrock = 1100
    c = 1.88
    n = 1.18
    minmb = 6.0
    delta = 0.1
    deltaz = 1
    refzif = 15
    refzslab = 50
    thisFs = EventType

    # check if cross arc:
    fx = 1
    if distR1 == 0 and distR2 == 0:
        fx = 0
    if distR2 == 0 and distR3 == 0:
        fx = 0
    if distR1 == 0 and distR3 == 0:
        fx = 0

    fmag = (1 - thisFs) * loghinge(
        M, mbreak, coeffs[6] * (mbreak - minmb), coeffs[6], coeffs[8], delta
    ) + thisFs * loghinge(
        M, mbreak, coeffs[7] * (mbreak - minmb), coeffs[7], coeffs[8], delta
    )
    fgeom = (
        (1 - thisFs)
        * (coeffs[3] + coeffs[5] * M)
        * math.log(Rrup + 10 ** (nft1 + nft2 * (M - 6)))
    )
    fgeom_slab = (
        thisFs
        * (coeffs[4] + coeffs[5] * M)
        * math.log(Rrup + 10 ** (nft1 + nft2 * (M - 6)))
    )
    fdepth = (1 - thisFs) * loghinge(
        Ztor, zbreak, coeffs[12] * (zbreak - refzif), coeffs[12], theta10, deltaz
    ) + thisFs * loghinge(
        Ztor, zbreak, coeffs[13] * (zbreak - refzslab), coeffs[13], theta10, deltaz
    )

    DotProduct123 = (
        distR1 * coeffs_attn[1] + distR2 * coeffs_attn[2] + distR3 * coeffs_attn[3]
    )
    DotProduct456 = (
        distR1 * coeffs_attn[4] + distR2 * coeffs_attn[5] + distR3 * coeffs_attn[6]
    )
    fattn = fx * DotProduct123 + (1 - fx) * DotProduct456 + fx * coeffs[14]

    if Vs30 < k1:
        fsite = coeffs[11] * math.log(Vs30 / k1) + k2 * (
            math.log(pgarock + c * (Vs30 / k1) ** n) - math.log(pgarock + c)
        )
    else:
        fsite = (coeffs[11] + k2 * n) * math.log(Vs30 / k1)

    fbasin = coeffs_z[1] + coeffs_z[2] * delta_ln_z

    Median = (
        (1 - thisFs) * coeffs[1]
        + thisFs * coeffs[2]
        + fmag
        + fgeom
        + fgeom_slab
        + fdepth
        + fattn
        + fsite
        + fbasin
    )
    return Median


### def to calculate median prediction for a given scenario and period
# This is a def that calculates median predictions of KBCG20 for a given scenario.
# I takes as input period, the predictor variables for the scenarios, selects the appropriate coefficients/parameters, and calls the def defined in the previous section.
#
# The arguments are similar to the def before.

# `m`: moment magnitude
# `rlist`: a vector of length 3, which contains the distances in subregion 1,2,3 relative to volcanic arc; `rlist = array(R1,R2,R3)`. for regions Alaska, Cascadia, New Zealand, and Taiwan, $R1 = R2 = 0$.
# `ztor`: depth to top of rupture in km.
# `EventType`: flag for interface (`EventType = 0`) and intraslab (`EventType = 1`). Must be 0 or 1.
# `Vs30`: $V_{S30}$ n m/s.
# `Z1pt0`: depth to a shear wave horizon of 1000 m/s, in m. Used for regions Alaska and New Zealand.
# `Z2pt5`: depth to a shear wave horizon of 2500 m/s, in m. Used for regions Cascadia and Japan.
# `fx`: arc crossing flag. Must be 0 or 1.
# `mb`: Magnitude scaling break point. Should be set regionally dependent based on Campbell (2020).

# The last input is a region index `region`, which is as follows:
#
# * 0: global
# * 1: Alaska
# * 2: Cascadia
# * 3: Central America & Mexico
# * 4: Japan
# * 5: New Zealand
# * 6: South America
# * 7: Taiwan
#
# The opional argumen `Seattle_Basin` is a flag that should be set to `TRUE` if the site is in the Seattle Basin, and `FALSE` otherwise.
# This flag determines which basin depth amplification model is used for Cascadia.
# It does not have an impact on any other region.
#


def KBCG20_medPSA(
    period,
    Magnitude,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    EventType,
    Vs30,
    Z1pt0,
    Z2pt5,
    Mb,
    UserRegion,
    Seattle_Basin,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    thisOut = -999
    XinterpMin = 0
    XinterpMax = 10
    XinterpType = "log"
    YinterpType = "log"
    extrapolateType = "extrapolate"

    Parameters = pd.read_csv("NGAsubGMM_KBCG20_coefficients.csv")
    TvalueList = Parameters["T"]
    InterpArray = interpolateFunction(
        period, TvalueList, XinterpType, extrapolateType, XinterpMin, XinterpMax
    )

    period0 = TvalueList[InterpArray[1]]
    y0 = KBCG20_medPSA_AtTlist(
        period0,
        Magnitude,
        Rrup,
        AlphaBackarc,
        AlphaNankai,
        Ztor,
        EventType,
        Vs30,
        Z1pt0,
        Z2pt5,
        Mb,
        UserRegion,
        Seattle_Basin,
    )

    if InterpArray[2] <= 0:
        thisOut = y0
    else:
        period1 = TvalueList[InterpArray[2]]
        y1 = KBCG20_medPSA_AtTlist(
            period1,
            Magnitude,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            Vs30,
            Z1pt0,
            Z2pt5,
            Mb,
            UserRegion,
            Seattle_Basin,
        )
        if YinterpType.lower().startswith("log"):
            if y0 <= 0:
                y0 = 0.000000001
            if y1 <= 0:
                y1 = 0.000000001
            y0 = math.log(y0)
            y1 = math.log(y1)

        thisOut = y0 + (y1 - y0) * InterpArray[3]
        if YinterpType.lower().startswith("log"):
            thisOut = math.exp(thisOut)

    return thisOut


# def to calculate median prediction using mean coefficients
def KBCG20_medPSA_AtTlist(
    period,
    M,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    EventType,
    Vs30,
    Z1pt0,
    Z2pt5,
    Mb,
    region,
    Seattle_Basin,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    KBCG20_medPSA_AtTlistValue = -999
    # need to add some checks for input (period, region)
    # coefficients to calculate zref from vs30
    pars_z_ja = [
        -999,
        7.6893685375,
        2.30258509299405,
        6.3091864,
        0.7528670225,
        1.2952369625,
    ]
    pars_z_casc = [
        -999,
        8.29404964010203,
        2.30258509299405,
        6.39692965521615,
        0.27081459,
        1.7381352625,
    ]
    pars_z_nz = [
        -999,
        6.859789675,
        2.30258509299405,
        5.745692775,
        0.91563524375,
        1.03531412375,
    ]
    pars_z_tw = [
        -999,
        6.30560665,
        2.30258509299405,
        6.1104992125,
        0.43671102,
        0.7229702975,
    ]

    Parameters = pd.read_csv("NGAsubGMM_KBCG20_coefficients.csv")
    parameters_zmod = pd.read_csv(
        "NGAsubGMM_KBCG20_params_Z_ALL_allregca_attn3_corrreg_cs_dmb.csv"
    )

    # calculate rock PGA
    period_used = 0.0
    vsrock = 1100
    pars_period = getRangeRowT(Parameters, period_used)
    pars_period_zmod = getRangeRowT(parameters_zmod, period_used)
    # silviamazzoni: I added this

    coeffs = getSubArrayRange(pars_period, 2, 15)
    k1k2 = interp_k1k2(period_used)
    dmb = interp_dmb(period_used)

    delta_ln_z = 0
    coeffs_z = [-999, 0, 0]
    coeffs_z2 = [-999, 0, 0]

    if region == 0:
        coeffs_attn = makeArray(pars_period, [11, 11, 11, 11, 10, 11])
        delta_ln_z = 0
        coeffs_z = [-999, 0, 0]
    elif region == 1:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [17, 24, 31])
        coeffs_attn = makeArray(pars_period, [38, 45, 52, 59, 66, 73])
    elif region == 2:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [18, 25, 32])
        coeffs_attn = makeArray(pars_period, [39, 46, 53, 60, 67, 74])
        coeffs_z2 = makeArray(pars_period_zmod, [2, 3])
        if Seattle_Basin:
            coeff_seattle = getRangeValueT(
                parameters_zmod, period_used, "mean_residual_Seattle_basin"
            )
            coeffs_z2 = [-999, coeff_seattle, 0]
    elif region == 3:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [19, 26, 33])
        coeffs_attn = makeArray(pars_period, [40, 47, 54, 61, 68, 75])
    elif region == 4:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [20, 27, 34])
        coeffs_attn = makeArray(pars_period, [41, 48, 55, 62, 69, 76])
        coeffs_z2 = makeArray(pars_period_zmod, [5, 6])
    elif region == 5:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [21, 28, 35])
        coeffs_attn = makeArray(pars_period, [42, 49, 56, 63, 70, 77])
        coeffs_z2 = makeArray(pars_period_zmod, [7, 8])
    elif region == 6:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [22, 29, 36])
        coeffs_attn = makeArray(pars_period, [43, 50, 57, 64, 71, 78])
    elif region == 7:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [23, 30, 37])
        coeffs_attn = makeArray(pars_period, [44, 51, 58, 65, 72, 79])
        coeffs_z2 = makeArray(pars_period_zmod, [9, 10])

    delta_bz = makeArray(pars_period, [80, 81])
    coeffs_nft = makeArray(pars_period, [82, 83])
    thisFs = EventType
    mbreak = (1 - thisFs) * (Mb + dmb) + thisFs * Mb

    zbreak = (1 - thisFs) * (30 + delta_bz[1]) + thisFs * (80 + delta_bz[2])

    mbreak_pga = mbreak
    zbreak_pga = zbreak
    k1k2_pga = k1k2
    coeffs_pga = coeffs
    # silviamazzoni: I added this so I alway have coeffs
    coeffs_attn_pga = coeffs_attn
    coeffs_z_pga = coeffs_z
    coeffs_z_pga2 = coeffs_z2
    coeffs_nft_pga = coeffs_nft
    pgarock = math.exp(
        KBCG20_med(
            M,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            vsrock,
            delta_ln_z,
            coeffs_pga,
            coeffs_attn_pga,
            coeffs_z_pga,
            mbreak_pga,
            zbreak_pga,
            k1k2_pga[1],
            k1k2_pga[2],
            coeffs_nft_pga[1],
            coeffs_nft_pga[2],
            0,
            region,
        )
    )

    # calculate PSA
    period_used = period
    pars_period = getRangeRowT(Parameters, period_used)
    pars_period_zmod = getRangeRowT(parameters_zmod, period_used)
    # silviamazzoni: I added this
    coeffs = getSubArrayRange(pars_period, 2, 15)
    k1k2 = interp_k1k2(period_used)
    dmb = interp_dmb(period_used)

    delta_ln_z = 0
    coeffs_z = [-999, 0, 0]
    if region == 0:
        coeffs_attn = makeArray(pars_period, [11, 11, 11, 11, 10, 11])
    elif region == 1:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [17, 24, 31])
        coeffs_attn = makeArray(pars_period, [38, 45, 52, 59, 66, 73])
    elif region == 2:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [18, 25, 32])
        coeffs_attn = makeArray(pars_period, [39, 46, 53, 60, 67, 74])
        delta_ln_z = math.log(Z2pt5) - calc_z_from_Vs30(Vs30, pars_z_casc)
        coeffs_z = makeArray(pars_period_zmod, [2, 3])
        if Seattle_Basin:
            coeff_seattle = getRangeValueT(
                parameters_zmod, period_used, "mean_residual_Seattle_basin"
            )
            coeffs_z = [-999, coeff_seattle, 0]
    elif region == 3:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [19, 26, 33])
        coeffs_attn = makeArray(pars_period, [40, 47, 54, 61, 68, 75])
    elif region == 4:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [20, 27, 34])
        coeffs_attn = makeArray(pars_period, [41, 48, 55, 62, 69, 76])
        delta_ln_z = math.log(Z2pt5) - calc_z_from_Vs30(Vs30, pars_z_ja)
        coeffs_z = makeArray(pars_period_zmod, [5, 6])
    elif region == 5:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [21, 28, 35])
        coeffs_attn = makeArray(pars_period, [42, 49, 56, 63, 70, 77])
        delta_ln_z = math.log(Z1pt0) - calc_z_from_Vs30(Vs30, pars_z_nz)
        coeffs_z = makeArray(pars_period_zmod, [7, 8])
    elif region == 6:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [22, 29, 36])
        coeffs_attn = makeArray(pars_period, [43, 50, 57, 64, 71, 78])
    elif region == 7:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [23, 30, 37])
        coeffs_attn = makeArray(pars_period, [44, 51, 58, 65, 72, 79])
        delta_ln_z = math.log(Z1pt0) - calc_z_from_Vs30(Vs30, pars_z_tw)
        coeffs_z = makeArray(pars_period_zmod, [9, 10])

    delta_bz = makeArray(pars_period, [80, 81])
    coeffs_nft = makeArray(pars_period, [82, 83])

    thisFs = EventType
    mbreak = (1 - thisFs) * (Mb + dmb) + thisFs * Mb
    zbreak = (1 - thisFs) * (30 + delta_bz[1]) + thisFs * (80 + delta_bz[2])

    Med = KBCG20_med(
        M,
        Rrup,
        AlphaBackarc,
        AlphaNankai,
        Ztor,
        EventType,
        Vs30,
        delta_ln_z,
        coeffs,
        coeffs_attn,
        coeffs_z,
        mbreak,
        zbreak,
        k1k2[1],
        k1k2[2],
        coeffs_nft[1],
        coeffs_nft[2],
        pgarock,
        region,
    )
    med_pga = KBCG20_med(
        M,
        Rrup,
        AlphaBackarc,
        AlphaNankai,
        Ztor,
        EventType,
        Vs30,
        delta_ln_z,
        coeffs_pga,
        coeffs_attn_pga,
        coeffs_z_pga2,
        mbreak_pga,
        zbreak_pga,
        k1k2_pga[1],
        k1k2_pga[2],
        coeffs_nft_pga[1],
        coeffs_nft_pga[2],
        pgarock,
        region,
    )

    if Med < med_pga and period <= 0.1:
        Med = med_pga

    return math.exp(Med)


def getRangeValue(thisRange, Xvalue, xHeader, yHeader):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    # not sure how to make it work with xHeader as a variable
    getRangeRowValue = thisRange.loc[thisRange[xHeader] == Xvalue]
    getRangeValueValue = getRangeRowValue[yHeader]
    return getRangeValueValue.squeeze()


def getRangeValueT(thisRange, Xvalue, yHeader):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    getRangeRowTValue = thisRange.loc[thisRange["T"] == Xvalue]
    getRangeValueTValue = getRangeRowTValue[yHeader]
    return getRangeValueTValue.squeeze()


def getRangeRowT(thisRange, Xvalue):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    getRangeRowTValue = thisRange.loc[thisRange["T"] == Xvalue]
    this = getRangeRowTValue.values
    this = np.insert(this, 0, -999)
    out = this.flatten()
    return out


def getRangeRowByIndex(thisRange, thisRowIndex):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    getRangeRowTValue = thisRange.loc[thisRowIndex - 1]
    this = getRangeRowTValue.values
    this = np.insert(this, 0, -999)
    out = this.flatten()
    return out


def getSubArrayRange(inArray, startIndex, endIndex):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    temP = [-999]
    # insert something at the zero start index spot
    for thisIndex in range(startIndex, endIndex + 1):
        # should offset
        temP.append(inArray[thisIndex])

    getSubArrayRangeValue = temP
    return getSubArrayRangeValue


def makeArray(oldArray, oldArrayIndices):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    nrow = len(oldArrayIndices)
    nrowOldArray = len(oldArrayIndices)
    newArray = [-999]
    # insert spacer for zero start indexing
    nrowMax = nrow
    for irow in range(len(oldArrayIndices)):  # zero-start array
        oldIndex = oldArrayIndices[irow]
        newArray.append(oldArray[oldIndex])

    makeArrayValue = newArray

    return makeArrayValue


def updateArray(newArray, newArrayIndices, oldArray, oldArrayIndices):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    NN = len(oldArrayIndices)
    for irow in range(0, NN - 1 + 1):  # zero-start index
        oldIndex = oldArrayIndices[irow]
        newIndex = newArrayIndices[irow]

        newArray[newIndex] = oldArray[oldIndex]
        # zero-start index not here as I have added spacers to the arrays

    updateArrayValue = newArray
    return updateArrayValue


def interpolateFunction(
    Xpoint, Xlist, XinterpType, extrapolateType, XinterpMin=-1e16, XinterpMax=+1e16
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    interpolateFunctionValue = -999
    irow = 0
    start0 = -1
    endrow0 = -1
    NN = len(Xlist)
    for irow in range(0, NN):
        this = Xlist[irow]
        if Xpoint == this:
            interpolateFunctionValue = [-999, irow, -1, 0]
            # I have added a value at the zero-index location
            return interpolateFunctionValue

        if this >= XinterpMin and this <= XinterpMax:
            if start0 < 0:
                start0 = irow
            end0 = irow

    iend = irow - 1

    istart = start0
    Xstart = Xlist[start0]
    Xend = Xlist[end0]

    if (Xend - Xpoint) * (Xstart - Xpoint) <= 0:
        for i in range(istart, iend - 1 + 1):
            thisx = Xlist[i, 1]
            nextX = Xlist[i + 1, 1]
            if (Xpoint - thisx) * (Xpoint - nextX) <= 0:
                l1 = i
                l2 = i + 1
                break
    else:
        if (Xend - Xstart) * (Xend - Xpoint) > 0:
            l1 = istart
            if extrapolateType.lower().startswith("extra"):
                l2 = istart + 1
            else:
                l2 = istart
        else:
            l1 = iend
            if extrapolateType.lower().startswith("extra"):
                l2 = iend - 1
            else:
                interpolateFunctionValue = [-999, end0, -1, 0]
                return interpolateFunctionValue

    # Interp:
    x0 = Xlist[l1]
    x1 = Xlist[l2]

    hereX = Xpoint
    if XinterpType.lower().startswith("log"):
        if x0 <= 0:
            x0 = 0.000000001
        if x1 <= 0:
            x1 = 0.000000002
        if hereX <= 0:
            hereX = 0.000000001

        x0 = math.log(x0)
        x1 = math.log(x1)
        hereX = math.log(hereX)

    interpolateFunctionValue = [-999, l1, l2, (hereX - x0) / (x1 - x0)]
    return interpolateFunctionValue


def KBCG20_posteriorAtTlist(
    period,
    M,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    EventType,
    Vs30,
    Z1pt0,
    Z2pt5,
    Mb,
    region,
    num_samples=100,
    Seattle_Basin=0,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    # need to add some checks for input (period, region)

    KBCG20_posteriorAtTlistValue = -999
    # coefficients to calculate zref from vs30
    pars_z_ja = [
        -999,
        7.6893685375,
        2.30258509299405,
        6.3091864,
        0.7528670225,
        1.2952369625,
    ]
    pars_z_casc = [
        -999,
        8.29404964010203,
        2.30258509299405,
        6.39692965521615,
        0.27081459,
        1.7381352625,
    ]
    pars_z_nz = [
        -999,
        6.859789675,
        2.30258509299405,
        5.745692775,
        0.91563524375,
        1.03531412375,
    ]
    pars_z_tw = [
        -999,
        6.30560665,
        2.30258509299405,
        6.1104992125,
        0.43671102,
        0.7229702975,
    ]

    Parameters = pd.read_csv("NGAsubGMM_KBCG20_coefficients.csv")
    parameters_zmod = pd.read_csv(
        "NGAsubGMM_KBCG20_params_Z_ALL_allregca_attn3_corrreg_cs_dmb.csv"
    )

    # calculate rock PGA
    thisPeriod = 0.0
    vsrock = 1100

    pars_period = getRangeRowT(Parameters, thisPeriod)

    pars_period_zmod = getRangeRowT(parameters_zmod, thisPeriod)
    # silviamazzoni, I added this
    coeffs = getSubArrayRange(pars_period, 2, 15)
    k1k2 = interp_k1k2(thisPeriod)
    dmb = interp_dmb(thisPeriod)

    delta_ln_z = 0
    coeffs_z = [-999, 0, 0]
    if region == 0:
        coeffs_attn = makeArray(pars_period, [11, 11, 11, 11, 10, 11])
    elif region == 1:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [17, 24, 31])
        coeffs_attn = makeArray(pars_period, [38, 45, 52, 59, 66, 73])
    elif region == 2:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [18, 25, 32])
        coeffs_attn = makeArray(pars_period, [39, 46, 53, 60, 67, 74])
    elif region == 3:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [19, 26, 33])
        coeffs_attn = makeArray(pars_period, [40, 47, 54, 61, 68, 75])
    elif region == 4:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [20, 27, 34])
        coeffs_attn = makeArray(pars_period, [41, 48, 55, 62, 69, 76])
    elif region == 5:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [21, 28, 35])
        coeffs_attn = makeArray(pars_period, [42, 49, 56, 63, 70, 77])
    elif region == 6:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [22, 29, 36])
        coeffs_attn = makeArray(pars_period, [43, 50, 57, 64, 71, 78])
    elif region == 7:
        coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [23, 30, 37])
        coeffs_attn = makeArray(pars_period, [44, 51, 58, 65, 72, 79])

    delta_bz = makeArray(pars_period, [80, 81])
    coeffs_nft = makeArray(pars_period, [82, 83])
    thisFs = EventType
    mbreak = (1 - thisFs) * (Mb + dmb) + thisFs * Mb

    zbreak = (1 - thisFs) * (30 + delta_bz[1]) + thisFs * (80 + delta_bz[2])

    pgarock = math.exp(
        KBCG20_med(
            M,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            vsrock,
            delta_ln_z,
            coeffs,
            coeffs_attn,
            coeffs_z,
            mbreak,
            zbreak,
            k1k2[1],
            k1k2[2],
            coeffs_nft[1],
            coeffs_nft[2],
            0,
            region,
        )
    )

    # calculate PSA
    thisPeriod = period
    Tnumeric = [
        -1,
        0,
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
    Tstring = [
        "T-1.00",
        "T00.00",
        "T00.01",
        "T00.02",
        "T00.03",
        "T00.05",
        "T00.07",
        "T00.10",
        "T00.15",
        "T00.20",
        "T00.25",
        "T00.30",
        "T00.40",
        "T00.50",
        "T00.75",
        "T01.00",
        "T01.50",
        "T02.00",
        "T03.00",
        "T04.00",
        "T05.00",
        "T07.50",
        "T10.00",
    ]

    thisIndex = Tnumeric.index(thisPeriod)
    thisTstring = Tstring[thisIndex]

    parameters_posterior = pd.read_csv(
        "posterior_coefficients_KBCG20_" + thisTstring + ".csv"
    )

    pars_period_zmod = getRangeRowT(parameters_zmod, thisPeriod)
    # silviamazzoni, I added this

    coeffs = getSubArrayRange(pars_period, 2, 15)
    k1k2 = interp_k1k2(thisPeriod)
    dmb = interp_dmb(thisPeriod)
    med_predictions = []
    for k in range(1, num_samples + 1):
        pars_period = getRangeRowByIndex(parameters_posterior, k)

        coeffs = getSubArrayRange(pars_period, 2, 15)

        delta_ln_z = 0
        coeffs_z = [-999, 0, 0]
        if region == 0:
            coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [172, 173, 174])
            coeffs_attn = makeArray(pars_period, [175, 176, 177, 178, 179, 180])
        elif region == 1:
            coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [17, 24, 31])
            coeffs_attn = makeArray(pars_period, [38, 45, 52, 59, 66, 73])
        elif region == 2:
            coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [18, 25, 32])
            coeffs_attn = makeArray(pars_period, [39, 46, 53, 60, 67, 74])
            delta_ln_z = math.log(Z2pt5) - calc_z_from_Vs30(Vs30, pars_z_casc)
            coeffs_z = makeArray(pars_period_zmod, [2, 3])
            if Seattle_Basin:
                coeff_seattle = getRangeValueT(
                    parameters_zmod, thisPeriod, "mean_residual_Seattle_basin"
                )
                coeffs_z = [coeff_seattle, 0]
        elif region == 3:
            coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [19, 26, 33])
            coeffs_attn = makeArray(pars_period, [40, 47, 54, 61, 68, 75])
        elif region == 4:
            coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [20, 27, 34])
            coeffs_attn = makeArray(pars_period, [41, 48, 55, 62, 69, 76])
            delta_ln_z = math.log(Z2pt5) - calc_z_from_Vs30(Vs30, pars_z_ja)
            coeffs_z = makeArray(pars_period_zmod, [5, 6])
        elif region == 5:
            coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [21, 28, 35])
            coeffs_attn = makeArray(pars_period, [42, 49, 56, 63, 70, 77])
            delta_ln_z = math.log(Z1pt0) - calc_z_from_Vs30(Vs30, pars_z_nz)
            coeffs_z = makeArray(pars_period_zmod, [7, 8])
        elif region == 6:
            coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [22, 29, 36])
            coeffs_attn = makeArray(pars_period, [43, 50, 57, 64, 71, 78])
        elif region == 7:
            coeffs = updateArray(coeffs, [1, 2, 11], pars_period, [23, 30, 37])
            coeffs_attn = makeArray(pars_period, [44, 51, 58, 65, 72, 79])
            delta_ln_z = math.log(Z1pt0) - calc_z_from_Vs30(Vs30, pars_z_tw)
            coeffs_z = makeArray(pars_period_zmod, [9, 10])

        delta_bz = makeArray(pars_period, [80, 81])
        coeffs_nft = makeArray(pars_period, [82, 83])
        thisFs = EventType

        mbreak = (1 - thisFs) * (Mb + dmb) + thisFs * Mb

        zbreak = (1 - thisFs) * (30 + delta_bz[1]) + thisFs * (80 + delta_bz[2])

        Med = KBCG20_med(
            M,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            Vs30,
            delta_ln_z,
            coeffs,
            coeffs_attn,
            coeffs_z,
            mbreak,
            zbreak,
            k1k2[1],
            k1k2[2],
            coeffs_nft[1],
            coeffs_nft[2],
            pgarock,
            region,
        )
        med_predictions.append(Med)

    KBCG20_posteriorAtTlist_All = [-999]
    KBCG20_posteriorAtTlist_All.append(np.average(med_predictions))
    KBCG20_posteriorAtTlist_All.append(np.median(med_predictions))
    KBCG20_posteriorAtTlist_All.append(np.std(med_predictions))
    for i in range(1, len(med_predictions)):
        KBCG20_posteriorAtTlist_All.append(med_predictions[i])

    KBCG20_posteriorAtTlistValue = KBCG20_posteriorAtTlist_All
    return KBCG20_posteriorAtTlistValue


def KBCG20_posterior(
    period,
    M,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    EventType,
    Vs30,
    Z1pt0,
    Z2pt5,
    Mb,
    region,
    num_samples=100,
    Seattle_Basin=0,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    thisOut = -999
    XinterpMin = 0
    XinterpMax = 10
    XinterpType = "log"
    YinterpType = "linear"
    # KBCG20_posteriorAtTlist values are logs already
    extrapolateType = "extrapolate"

    Parameters = pd.read_csv("NGAsubGMM_KBCG20_coefficients.csv")
    TvalueList = Parameters["T"]
    InterpArray = interpolateFunction(
        period, TvalueList, XinterpType, extrapolateType, XinterpMin, XinterpMax
    )

    period0 = TvalueList[InterpArray[1]]
    KBCG20_posteriorAtTlist0 = KBCG20_posteriorAtTlist(
        period0,
        M,
        Rrup,
        AlphaBackarc,
        AlphaNankai,
        Ztor,
        EventType,
        Vs30,
        Z1pt0,
        Z2pt5,
        Mb,
        region,
        num_samples,
        Seattle_Basin,
    )
    y0 = InterpArray[1]
    if InterpArray[2] <= 0:
        KBCG20_posteriorValue = KBCG20_posteriorAtTlist0
        return KBCG20_posteriorValue
    else:
        period1 = Tvalue[InterpArray[2]]
        KBCG20_posteriorAtTlist1 = KBCG20_posteriorAtTlist(
            period1,
            M,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            Vs30,
            Z1pt0,
            Z2pt5,
            Mb,
            region,
            num_samples,
            Seattle_Basin,
        )
        for iCase in range(1, len(KBCG20_posteriorAtTlist0)):
            y0 = KBCG20_posteriorAtTlist0[iCase]
            y1 = KBCG20_posteriorAtTlist1[iCase]

            if YinterpType.lower().startswith("log"):
                if y0 <= 0:
                    y0 = 0.000000001
                if y1 <= 0:
                    y1 = 0.000000001
                y0 = math.log(y0)
                y1 = math.log(y1)

            y0
            y1
            InterpArray[3]
            y0 + (y1 - y0) * InterpArray[3]
            med_predictions[iCase] = y0 + (y1 - y0) * InterpArray[3]
            if YinterpType.lower().startswith("log"):
                med_predictions[iCase] = math.exp(med_predictions[iCase])

    return med_predictions


# def to calculate aliatory sigma


def KBCG20_sigmaAleatory(period):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    thisOut = -999
    XinterpMin = 0
    XinterpMax = 10
    XinterpType = "log"
    YinterpType = "linear"
    extrapolateType = "extrapolate"

    Parameters = pd.read_csv("NGAsubGMM_KBCG20_coefficients.csv")
    TvalueList = Parameters["T"]
    InterpArray = interpolateFunction(
        period, TvalueList, XinterpType, extrapolateType, XinterpMin, XinterpMax
    )

    period0 = TvalueList[InterpArray[1]]
    y0 = KBCG20_sigmaAleatoryAtTlist(period0)

    if InterpArray[2] <= 0:
        thisOut = y0
    else:
        period1 = TvalueList[InterpArray[2], 1]
        y1 = KBCG20_sigmaAleatoryAtTlist(period1)
        if YinterpType.lower().startswith("log"):
            if y0 <= 0:
                y0 = 0.000000001
            if y1 <= 0:
                y1 = 0.000000001
            y0 = math.log(y0)
            y1 = math.log(y1)

        thisOut = y0 + (y1 - y0) * InterpArray[3]
        if YinterpType.lower().startswith("log"):
            thisOut = math.exp(thisOut)

    return thisOut


def KBCG20_sigmaAleatoryAtTlist(period):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    Parameters = pd.read_csv("NGAsubGMM_KBCG20_coefficients.csv")
    # calculate PSA
    period_used = period
    pars_period = getRangeRowT(Parameters, period_used)

    Phi = pars_period[84]
    Tau = pars_period[85]
    KBCG20_sigmaAleatoryAtTlistValue = math.sqrt(Phi * Phi + Tau * Tau)
    return KBCG20_sigmaAleatoryAtTlistValue


def KBCG20_sigmaPhi(period):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    Parameters = pd.read_csv("NGAsubGMM_KBCG20_coefficients.csv")

    # calculate PSA
    period_used = period
    pars_period = getRangeRowT(Parameters, period_used)

    Phi = pars_period[84]
    KBCG20_sigmaPhiValue = Phi
    return KBCG20_sigmaPhiValue


def KBCG20_sigmaTau(period):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    Parameters = pd.read_csv("NGAsubGMM_KBCG20_coefficients.csv")

    # calculate PSA
    period_used = getTvalue(period)
    pars_period = getRangeRowT(Parameters, period_used)
    Tau = pars_period[85]
    KBCG20_sigmaTauValue = Tau
    return KBCG20_sigmaTauValue


def KBCG20_SigmaEpistemic(
    period,
    M,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    EventType,
    Vs30,
    Z1pt0,
    Z2pt5,
    Mb,
    region,
    num_samples=100,
    Seattle_Basin=0,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    if num_samples == 0:
        KBCG20_SigmaEpistemicValue = 0
    elif num_samples < 50:
        KBCG20_SigmaEpistemicValue = (
            "not enough samples, please select a value =0 or between 50 and 800"
        )
    elif num_samples <= 800:
        thisKBCG20_posterior = KBCG20_posterior(
            period,
            M,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            Vs30,
            Z1pt0,
            Z2pt5,
            Mb,
            region,
            num_samples=100,
            Seattle_Basin=0,
        )
        KBCG20_SigmaEpistemicValue = thisKBCG20_posterior[3]
    else:
        KBCG20_SigmaEpistemicValue = "Please enter a value =0 or between 50 and 800"

    return KBCG20_SigmaEpistemicValue


def KBCG20_SigmaTotal(
    period,
    M,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    EventType,
    Vs30,
    Z1pt0,
    Z2pt5,
    Mb,
    region,
    num_samples=100,
    Seattle_Basin=0,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    SigmaAleatory = KBCG20_sigmaAleatory(period)
    if num_samples == 0:
        SigmaEpistemic = 0.0
    else:
        SigmaEpistemic = KBCG20_SigmaEpistemic(
            period,
            M,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            Vs30,
            Z1pt0,
            Z2pt5,
            Mb,
            region,
            num_samples=100,
            Seattle_Basin=0,
        )

    KBCG20_SigmaTotalValue = math.sqrt(
        SigmaAleatory * SigmaAleatory + SigmaEpistemic * SigmaEpistemic
    )
    return KBCG20_SigmaTotalValue


###################################################################

# GMM_at_760_slab_v3.R
## Grace Parker
## Modified February 26 to expand comments
## Modified April 25, 2020 to call coefficients from master table
#
##  Input Parameters --------------------------------------------------------
#
## Event type: 0 == interface, 1 == slab
#
## region corresponds to options in the DatabaseRegion column of the flatfile, plus global. Must be a string. if no matches, default will be global model:
##  "global", "Alaska", "Cascadia", "CAM", "Japan", "SA" or "Taiwan"
#
## Saturation Region corresponds to regions defined by R. Archuleta and C. Ji:
##  "global", "Aleutian","Alaska","Cascadia","Central_America_S", "Central_America_N", "Japan_Pac","Japan_Phi","South_America_N","South_America_S", "Taiwan_W","Taiwan_E"
#
##  Rrup is number in kilometers
#
## Hypocentral depth in km. : use Ztor value to estimate hypocentral depth, see Ch. 4.3.3 of Parker et al. PEER report
#
##  period can be: (-1,0,0.01,0.02,0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,2.5,3,4,5,7.5,10)
##  where -1 == PGV and 0 == PGA
#
##  Other pertinent information ---------------------------------------------
## Coefficient files must be in the active working directory
##  This def has no site term. Can only estimate ground motion at the reference condition VS30 = 760m/s
##  The output is the desired median model prediction in LN units
##  Take the exponential to get PGA, PSA in g or the PGV in cm/s


def PSHAB20_GMM_at_760_Slab(
    EventType, UserRegion, saturation_region, Rrup, M, hypocentral_depth, period
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    PSHAB20_GMM_at_760_SlabValue = -999
    if EventType == 0:
        return PSHAB20_GMM_at_760_SlabValue
        # ("This def is only for slab")
    region = getPSHABregion(UserRegion)

    ##import Coefficients

    ##  Import Master Coefficient Table -----------------------------------------
    #  CoefficientsTable = readtable('PSHAB20_Table_E2_Slab_Coefficients_OneRowHeader.csv','PreserveVariableNames',True);
    CoefficientsTable = pd.read_csv(
        "PSHAB20_Table_E2_Slab_Coefficients_OneRowHeader.csv"
    )

    ##Define mb based on Archuleta and Ji (2019)
    Mb = getMbDefault(EventType, UserRegion, saturation_region)

    ##  Constant ----------------------------------------------------------------
    #  isolate constant
    if region == "global" or region == "Cascadia":
        c0 = getRangeValueT(CoefficientsTable, period, "Global_c0")
    elif region == "Alaska" or region == "SA":
        c0 = getRangeValueT(CoefficientsTable, period, saturation_region + "_c0")
    else:
        c0 = getRangeValueT(CoefficientsTable, period, region + "_c0")

    # silviamazzoni, I added this:
    if checkEmptyNA999(c0):
        c0 = getRangeValueT(CoefficientsTable, period, "Global_c0")
    #  # Path Term ---------------------------------------------------------------

    #  #near-source saturation
    if M <= Mb:
        littleM = (Log10(35) - Log10(3.12)) / (Mb - 4)
        h = 10 ** (littleM * (M - Mb) + Log10(35))
    else:
        h = 35
    Rref = math.sqrt(1 + h ** 2)
    r = math.sqrt(Rrup ** 2 + h ** 2)
    LogR = math.log(r)
    R_Rref = math.log(r / Rref)

    #  #Need  to isolate regional anelastic coefficient, a0
    if region == "global":
        a0 = getRangeValueT(CoefficientsTable, period, "Global_a0")
    else:
        a0 = getRangeValueT(CoefficientsTable, period, region + "_a0")
    if checkEmptyNA999(a0):
        a0 = getRangeValueT(CoefficientsTable, period, "Global_a0")

    c1 = getRangeValueT(CoefficientsTable, period, "c1")
    b4 = getRangeValueT(CoefficientsTable, period, "b4")
    Fp = c1 * LogR + (b4 * M) * R_Rref + a0 * r

    #  # Magnitude Scaling -------------------------------------------------------
    c4 = getRangeValueT(CoefficientsTable, period, "c4")
    c5 = getRangeValueT(CoefficientsTable, period, "c5")
    c6 = getRangeValueT(CoefficientsTable, period, "c6")
    Fm = c4 * func1(M, Mb) + c6 * func2(M, Mb) + c5 * func3(M, Mb)

    #  # Source Depth Scaling ----------------------------------------------------
    Db = getRangeValueT(CoefficientsTable, period, "db_km")
    d = getRangeValueT(CoefficientsTable, period, "d")
    littleM = getRangeValueT(CoefficientsTable, period, "m")
    if hypocentral_depth >= Db:
        Fd = d
    elif hypocentral_depth <= 20:
        Fd = littleM * (20 - Db) + d
    else:
        Fd = littleM * (hypocentral_depth - Db) + d

    mu = c0 + Fp + Fm + Fd
    return mu


def func1(M, Mb):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    if M <= Mb:
        func1value = M - Mb
    else:
        func1value = 0

    return func1value


def func2(M, Mb):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    if M > Mb:
        func2value = M - Mb
    else:
        func2value = 0

    return func2value


def func3(M, Mb):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    if M <= Mb:
        func3value = (M - Mb) ** 2
    else:
        func3value = 0

    return func3value


def getPSHABregion(UserRegion):
    # ' CODE DEVELOPED/IMPLEMENTED BY
    # '          Silvia Mazzoni, 2020
    # '           smazzoni@ucla.edu

    return ["global", "Alaska", "Cascadia", "CAM", "Japan", "global", "SA", "Taiwan"][
        UserRegion
    ]


def getMbDefault(EventType, UserRegion, saturation_region):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    if EventType == 1 or EventType == 5:
        if UserRegion == 5:
            # the New-Zealand Mb model has not been tested by the model developers, yet
            Mb = 7.6
        elif saturation_region == "global":
            Mb = 7.6
        else:
            slab_saturation_regions_SBZ = [
                "Aleutian",
                "Alaska",
                "-999",
                "Cascadia",
                "Central_America_S",
                "Central_America_N",
                "Japan_Pac",
                "Japan_Phi",
                "New_Zealand_N",
                "New_Zealand_S",
                "South_America_N",
                "South_America_S",
                "Taiwan_W",
                "Taiwan_E",
            ]
            slab_saturation_regions_Mb = [
                7.98,
                7.2,
                -999,
                7.2,
                7.6,
                7.4,
                7.65,
                7.55,
                7.6,
                7.4,
                7.3,
                7.25,
                7.7,
                7.7,
            ]
            SaturationRegionList = [
                "Alaska",
                "Aleutian",
                "Cascadia",
                "Central_America_N",
                "Central_America_S",
                "Japan_Pac",
                "Japan_Phi",
                "Northern_Mariana",
                "New_Zealand_N",
                "New_Zealand_S",
                "South_America_N",
                "South_America_S",
                "Taiwan_E",
                "Taiwan_W",
            ]
            SaturationRegionListDbRegionArray = [
                1,
                1,
                2,
                3,
                3,
                4,
                4,
                4,
                5,
                5,
                6,
                6,
                7,
                7,
            ]
            thisSaturationRegionDbRegion = getArrayMap(
                saturation_region,
                SaturationRegionList,
                SaturationRegionListDbRegionArray,
            )

            if thisSaturationRegionDbRegion != UserRegion:
                saturation_region = "global"
                Mb = 7.6
            else:
                Mb = getArrayMap(
                    saturation_region,
                    slab_saturation_regions_SBZ,
                    slab_saturation_regions_Mb,
                )

        if checkEmptyNA999(Mb):
            Mb = 7.6

    else:
        if UserRegion == 5:
            # the New-Zealand Mb model has not been tested by the model developers, yet
            Mb = 7.8
        elif saturation_region == "global":
            Mb = 7.8
        else:
            IF_saturation_regions_SBZ = [
                "Aleutian",
                "Alaska",
                "-999",
                "Cascadia",
                "Central_America_S",
                "Central_America_N",
                "Japan_Pac",
                "Japan_Phi",
                "New_Zealand_N",
                "New_Zealand_S",
                "South_America_N",
                "South_America_S",
                "Taiwan_W",
                "Taiwan_E",
            ]
            IF_saturation_regions_Mb = [
                8,
                8,
                -999,
                7.56,
                7.5,
                7.45,
                8.31,
                7.28,
                -999,
                -999,
                8.45,
                8.45,
                8,
                8,
            ]

            SaturationRegionList = [
                "Alaska",
                "Aleutian",
                "Cascadia",
                "Central_America_N",
                "Central_America_S",
                "Japan_Pac",
                "Japan_Phi",
                "Northern_Mariana",
                "New_Zealand_N",
                "New_Zealand_S",
                "South_America_N",
                "South_America_S",
                "Taiwan_E",
                "Taiwan_W",
            ]
            SaturationRegionListDbRegionArray = [
                "1_Alaska",
                "1_Alaska",
                "2_Cascadia",
                "3_CentralAmerica&Mexico",
                "3_CentralAmerica&Mexico",
                "4_Japan",
                "4_Japan",
                "4_Japan",
                "5_NewZealand",
                "5_NewZealand",
                "6_SouthAmerica",
                "6_SouthAmerica",
                "7_Taiwan",
                "7_Taiwan",
            ]
            thisSaturationRegionDbRegion = getArrayMap(
                saturation_region,
                SaturationRegionList,
                SaturationRegionListDbRegionArray,
            )

            if thisSaturationRegionDbRegion.lower() != UserRegion.lower():
                saturation_region = "global"
                Mb = 7.8
            else:
                Mb = getArrayMap(
                    saturation_region,
                    IF_saturation_regions_SBZ,
                    IF_saturation_regions_Mb,
                )

        if checkEmptyNA999(Mb):
            Mb = 7.8

    getMbDefaultValue = Mb

    return getMbDefaultValue


def getArrayMap(Xvalue, Xarray, Yarray):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    getArrayMapValue = -999
    for (thisX, thisY) in zip(Xarray, Yarray):
        if thisX == Xvalue:
            getArrayMapValue = thisY

    return getArrayMapValue


def checkEmptyNA999(thisvalue):
    checkEmptyNA999value = False
    if str(thisvalue) == "NA":
        checkEmptyNA999value = True
    elif str(thisvalue) == "":
        checkEmptyNA999value = True
    elif str(thisvalue) == "nan":
        checkEmptyNA999value = True

    return checkEmptyNA999value


# GMM_at_VS30_slab_v3.R
## Grace Parker
## Modified February 26 to expand comments
## Modified April 25, 2020 to call consolidated coefficient table
#
##  Input Parameters --------------------------------------------------------
#
## Event type: 0 == interface, 1 == slab
#
## region corresponds to options in the DatabaseRegion column of the flatfile, plus global. Must be a string. if no matches, default will be global model:
##  "global", "Alaska", "Cascadia", "CAM", "Japan", "SA" or "Taiwan"
#
## Saturation Region corresponds to regions defined by R. Archuleta and C. Ji:
##  "global", "Aleutian","Alaska","Cascadia","Central_America_S", "Central_America_N", "Japan_Pac","Japan_Phi","South_America_N","South_America_S", "Taiwan_W","Taiwan_E"
#
##  Rrup is number in kilometers
#
## Hypocentral depth in km. : use Ztor value to estimate hypocentral depth, see Ch. XXX of Parker et al. PEER report
#
##  period can be: (-1,0,0.01,0.02,0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,2.5,3,4,5,7.5,10)
##  where -1 == PGV and 0 == PGA
#
## VS30 in units m/s
#
## Z2.5 in units m. Only used if DatabaseRegion == "Japan" or "Cascadia". Can also specify "default" to get no basin term
#
## basin is only used if DatabaseRegion == "Cascadia". Value can be 0, 1, or 2, where 0 == having an estimate of Z2.5 outside mapped basin, 1 == Seattle basin, and 0 == other mapped basin (Tacoma, Everett, Georgia, etc.)
#
##  Other pertinent information ---------------------------------------------
## Coefficient files must be in the active working directory
##  "GMM_at_VS30_Slab_v2.R" calls def "GMM_at_760_Slab_v2.R" to compute PGAr in the nonlinear site term. This def must be in the R environment else : an error will occur.
##  The output is the desired median model prediction in LN units
##  Take the exponential to get PGA, PSA in g or  PGV in cm/s
#
#
## def to compute GMM predictions at various VS30s for slab


def PSHAB20_GMM_at_VS30_Slab(
    EventType,
    UserRegion,
    saturation_region,
    Rrup,
    M,
    hypocentral_depth,
    period,
    Vs30,
    Z2pt5,
    basin,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    PSHAB20_GMM_at_VS30_SlabValue = -999
    if EventType == 0:
        # ("This def is only for slab")
        return PSHAB20_GMM_at_VS30_SlabValue

    region = getPSHABregion(UserRegion)
    CoefficientsTable = pd.read_csv(
        "PSHAB20_Table_E2_Slab_Coefficients_OneRowHeader.csv"
    )

    # Define mb based on Archuleta and Ji (2019)
    Mb = getMbDefault(EventType, UserRegion, saturation_region)

    # Constant ----------------------------------------------------------------
    # Isolate constant
    if region == "global" or region == "Cascadia":
        c0 = getRangeValueT(CoefficientsTable, period, "Global_c0")
    elif region == "Alaska" or region == "SA":
        c0 = getRangeValueT(CoefficientsTable, period, saturation_region + "_c0")
    else:
        c0 = getRangeValueT(CoefficientsTable, period, region + "_c0")

    if checkEmptyNA999(c0):
        c0 = getRangeValueT(CoefficientsTable, period, "Global_c0")

    # Path Term ---------------------------------------------------------------
    # near-source saturation
    if M <= Mb:
        littleM = (Log10(35) - Log10(3.12)) / (Mb - 4)
        h = 10 ** (littleM * (M - Mb) + Log10(35))
    else:
        h = 35

    Rref = math.sqrt(1 + h ** 2)
    r = math.sqrt(Rrup ** 2 + h ** 2)
    LogR = math.log(r)
    R_Rref = math.log(r / Rref)

    #  #Need  to isolate regional anelastic coefficient, a0
    if region == "global":
        a0 = getRangeValueT(CoefficientsTable, period, "Global_a0")
    else:
        a0 = getRangeValueT(CoefficientsTable, period, region + "_a0")

    if checkEmptyNA999(a0):
        a0 = getRangeValueT(CoefficientsTable, period, "Global_a0")

    c1 = getRangeValueT(CoefficientsTable, period, "c1")
    b4 = getRangeValueT(CoefficientsTable, period, "b4")
    Fp = c1 * LogR + (b4 * M) * R_Rref + a0 * r

    ##  Magnitude Scaling -------------------------------------------------------
    c4 = getRangeValueT(CoefficientsTable, period, "c4")
    c5 = getRangeValueT(CoefficientsTable, period, "c5")
    c6 = getRangeValueT(CoefficientsTable, period, "c6")

    Fm = c4 * func1(M, Mb) + c6 * func2(M, Mb) + c5 * func3(M, Mb)

    ##  Source Depth Scaling ----------------------------------------------------
    Db = getRangeValueT(CoefficientsTable, period, "db_km")
    d = getRangeValueT(CoefficientsTable, period, "d")
    littleM = getRangeValueT(CoefficientsTable, period, "m")

    # compute depth scaling term
    if hypocentral_depth >= Db:
        Fd = d
    elif hypocentral_depth <= 20:
        Fd = littleM * (20 - Db) + d
    else:
        Fd = littleM * (hypocentral_depth - Db) + d

    ##  Linear Site Amplification ----------------------------------------------

    ## Site Coefficients
    V1 = getRangeValueT(CoefficientsTable, period, "V1_m_s")
    V2 = getRangeValueT(CoefficientsTable, period, "V2_m_s")
    Vref = getRangeValueT(CoefficientsTable, period, "Vref_m_s")

    if region == "global" or region == "CAM":
        s2 = getRangeValueT(CoefficientsTable, period, "Global_s2")
        s1 = s2
    elif region == "Taiwan" or region == "Japan":
        s2 = getRangeValueT(CoefficientsTable, period, region + "_s2")
        s1 = getRangeValueT(CoefficientsTable, period, region + "_s1")
    else:
        s2 = getRangeValueT(CoefficientsTable, period, region + "_s2")
        s1 = s2

    # compute linear site term
    if Vs30 <= V1:
        Flin = s1 * math.log(Vs30 / V1) + s2 * math.log(V1 / Vref)
    elif Vs30 <= V2:
        Flin = s2 * math.log(Vs30 / Vref)
    else:
        Flin = s2 * math.log(V2 / Vref)

    # Nonlinear Site Term -----------------------------------------------------
    PGAr = math.exp(
        PSHAB20_GMM_at_760_Slab(
            EventType, UserRegion, saturation_region, Rrup, M, hypocentral_depth, 0
        )
    )
    f3 = 0.05
    Vb = 200
    Vref_Fnl = 900

    if period >= 3:
        Fnl = 0
    else:
        f4 = getRangeValueT(CoefficientsTable, period, "f4")
        f5 = getRangeValueT(CoefficientsTable, period, "f5")
        f2 = f4 * (
            math.exp(f5 * (min(Vs30, Vref_Fnl) - Vb)) - math.exp(f5 * (Vref_Fnl - Vb))
        )
        Fnl = 0 + f2 * math.log((PGAr + f3) / f3)

    ##  Basin Term --------------------------------------------------------------
    #     Z2pt5
    #     region
    if str(Z2pt5) == "default":
        Fb = 0
    elif Z2pt5 <= 0 or (region != "Japan" and region != "Cascadia"):
        Fb = 0
    else:
        if region == "Cascadia":
            theta0 = 3.94
            theta1 = -0.42
            vmu = 200
            vsig = 0.2
            e1 = getRangeValueT(CoefficientsTable, period, "C_e1")

            C_e3 = getRangeValueT(CoefficientsTable, period, "C_e3")
            C_e2 = getRangeValueT(CoefficientsTable, period, "C_e2")

            if basin == 0:
                del_None = getRangeValueT(CoefficientsTable, period, "del_None")
                e3 = C_e3 + del_None
                e2 = C_e2 + del_None
            elif basin == 1:
                del_Seattle = getRangeValueT(CoefficientsTable, period, "del_Seattle")
                e3 = C_e3 + del_Seattle
                e2 = C_e2 + del_Seattle
            else:
                e3 = C_e3
                e2 = C_e2

        elif region == "Japan":

            theta0 = 3.05
            theta1 = -0.8
            vmu = 500
            vsig = 0.33
            e3 = getRangeValueT(CoefficientsTable, period, "J_e3")
            e2 = getRangeValueT(CoefficientsTable, period, "J_e2")
            e1 = getRangeValueT(CoefficientsTable, period, "J_e1")

        Z2pt5_pred = 10 ** (
            theta0
            + theta1
            * (1 + math.erf((Log10(Vs30) - Log10(vmu)) / (vsig * math.sqrt(2))))
        )
        delZ2pt5 = math.log(Z2pt5) - math.log(Z2pt5_pred)

        if delZ2pt5 <= (e1 / e3):
            Fb = e1
        elif delZ2pt5 >= (e2 / e3):
            Fb = e2
        else:
            Fb = e3 * delZ2pt5

    mu = c0 + Fp + Fm + Fd + Fnl + Flin + Fb
    return mu


def Log10(x):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    Log10Value = math.log(x) / math.log(10.0)
    return Log10Value


# GMM_at_760_IF_v3.R

## Grace Parker
## Modified February 26 to expand comments
## Modified April 25, 2020 to call consolidated coefficient tables
#
##  Input Parameters --------------------------------------------------------
#
## Event type: 0 == interface, 1 == slab
#
## region corresponds to options in the DatabaseRegion column of the flatfile, plus global. Must be a string. if no matches, default will be global model:
##  "global", "Alaska", "Cascadia", "CAM", "Japan", "SA" or "Taiwan"
#
## Saturation Region corresponds to regions defined by R. Archuleta and C. Ji:
##  "global", "Aleutian","Alaska","Cascadia","Central_America_S", "Central_America_N", "Japan_Pac","Japan_Phi","South_America_N","South_America_S", "Taiwan_W","Taiwan_E"
#
##  Rrup is number in kilometers
#
##  period can be: (-1,0,0.01,0.02,0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,2.5,3,4,5,7.5,10)
##  where -1 == PGV and 0 == PGA
#
##  Other pertinent information ---------------------------------------------
##  Coefficient files must be in the active working directory
##  This def has no site term. Can only estimate ground motion at the reference condition VS30 = 760m/s
##  The output is the desired median model prediction in LN units
##  Take the exponential to get PGA, PSA in g or the PGV in cm/s


def PSHAB20_GMM_at_760_IF(EventType, UserRegion, saturation_region, Rrup, M, period):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu
    PSHAB20_GMM_at_760_IFValue = -999
    if EventType == 1 or EventType == 5:
        return PSHAB20_GMM_at_760_IFValue
        # ("This def is only for IF")

    region = getPSHABregion(UserRegion)

    ##  Import Master Coefficient Table -----------------------------------------
    ##import CoefficientsTable
    #  CoefficientsTable = readtable('PSHAB20_Table_E1_Interface_Coefficients_OneRowHeader.csv','PreserveVariableNames',True);
    CoefficientsTable = pd.read_csv(
        "PSHAB20_Table_E1_Interface_Coefficients_OneRowHeader.csv"
    )

    ##Define Mb
    Mb = getMbDefault(EventType, UserRegion, saturation_region)

    ##  Constant ----------------------------------------------------------------

    if region == "global" or region == "Cascadia":
        c0 = getRangeValueT(CoefficientsTable, period, "Global_c0")
    else:
        c0 = getRangeValueT(CoefficientsTable, period, region + "_c0")

    ##  Path Term ---------------------------------------------------------------
    h = 10 ** (-0.82 + 0.252 * M)
    Rref = math.sqrt(1 + h ** 2)
    r = math.sqrt(Rrup ** 2 + h ** 2)
    LogR = math.log(r)
    R_Rref = math.log(r / Rref)

    #  #Need  to isolate regional anelastic coefficient, a0
    if region == "global":
        a0 = getRangeValueT(CoefficientsTable, period, "Global_a0")
    else:
        a0 = getRangeValueT(CoefficientsTable, period, region + "_a0")

    if checkEmptyNA999(a0):
        a0 = getRangeValueT(CoefficientsTable, period, "Global_a0")

    c1 = getRangeValueT(CoefficientsTable, period, "c1")
    b4 = getRangeValueT(CoefficientsTable, period, "b4")

    Fp = c1 * LogR + (b4 * M) * R_Rref + a0 * r

    ##  Magnitude Scaling -------------------------------------------------------
    c4 = getRangeValueT(CoefficientsTable, period, "c4")
    c5 = getRangeValueT(CoefficientsTable, period, "c5")
    c6 = getRangeValueT(CoefficientsTable, period, "c6")
    Fm = c4 * func1(M, Mb) + c6 * func2(M, Mb) + c5 * func3(M, Mb)

    ##  Add it all up! ----------------------------------------------------------

    mu = c0 + Fp + Fm

    PSHAB20_GMM_at_760_IFValue = mu
    return PSHAB20_GMM_at_760_IFValue


# GMM_at_Vs30_IF_v3.R
## Grace Parker
## Modified February 26, 2020, to expand comments
## Modified April 25, 2020, to take coefficients from "Table_E1_Interface_Coefficients.csv"
#
##  Input Parameters --------------------------------------------------------
#
## Event type: 0 == interface, 1 == slab
#
## region corresponds to options in the DatabaseRegion column of the flatfile, plus global. Must be a string. if no matches, default will be global model:
#  # "global", "Alaska", "Cascadia", "CAM", "Japan", "SA" or "Taiwan"
#
## Saturation Region corresponds to regions defined by R. Archuleta and C. Ji:
#  # "global", "Aleutian","Alaska","Cascadia","Central_America_S", "Central_America_N", "Japan_Pac","Japan_Phi","South_America_N","South_America_S", "Taiwan_W","Taiwan_E"
#
##  Rrup is number in kilometers
#
##  period can be: (-1,0,0.01,0.02,0.025,0.03,0.04,0.05,0.075,0.1,0.15,0.2,0.25,0.3,0.4,0.5,0.75,1,1.5,2,2.5,3,4,5,7.5,10)
#  # where -1 == PGV and 0 == PGA
#
## VS30 in units m/s
#
## Z2.5 in units m. Only used if DatabaseRegion == "Japan" or "Cascadia". Can also specify "default" to get no basin term
#
## basin is only used if DatabaseRegion == "Cascadia". Value can be 0, 1, or 2, where 0 == having an estimate of Z2.5 outside mapped basin, 1 == Seattle basin, and 0 == other mapped basin (Tacoma, Everett, Georgia, etc.)
#
##  Other pertinent information ---------------------------------------------
## Coefficient files must be in the active working directory
##  "GMM_at_VS30_IF_v3.R" calls def "GMM_at_760_IF_v2.R" to compute PGAr in the nonlinear site term. This def must be in the R environment else : an error will occur.
##  The output is the desired median model prediction in LN units
##  Take the exponential to get PGA, PSA in g or the PGV in cm/s


def PSHAB20_GMM_at_VS30_IF(
    EventType, UserRegion, saturation_region, Rrup, M, period, Vs30, Z2pt5, basin
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    PSHAB20_GMM_at_VS30_IFValue = -999
    if EventType == 1 or EventType == 5:
        return PSHAB20_GMM_at_VS30_IFValue
        # ("This def is only for IF")
    region = getPSHABregion(UserRegion)

    ##  Import Master Coefficient Table -----------------------------------------
    #  CoefficientsTable = readtable('PSHAB20_Table_E1_Interface_Coefficients_OneRowHeader.csv','PreserveVariableNames',True);
    CoefficientsTable = pd.read_csv(
        "PSHAB20_Table_E1_Interface_Coefficients_OneRowHeader.csv"
    )

    ##Define Mb
    Mb = getMbDefault(EventType, UserRegion, saturation_region)

    ##  Constant ----------------------------------------------------------------
    if region == "global" or region == "Cascadia":
        c0 = getRangeValueT(CoefficientsTable, period, "Global_c0")
    else:
        c0 = getRangeValueT(CoefficientsTable, period, region + "_c0")

    if checkEmptyNA999(c0):
        c0 = getRangeValueT(CoefficientsTable, period, "Global_c0")

    ##  Path Term ---------------------------------------------------------------
    h = 10 ** (-0.82 + 0.252 * M)
    Rref = math.sqrt(1 + h ** 2)
    r = math.sqrt(Rrup ** 2 + h ** 2)
    LogR = math.log(r)
    R_Rref = math.log(r / Rref)

    #  #Need  to isolate regional anelastic coefficient, a0
    if region == "global" or region == "Cascadia":
        a0 = getRangeValueT(CoefficientsTable, period, "Global_a0")
    else:
        a0 = getRangeValueT(CoefficientsTable, period, region + "_a0")

    if checkEmptyNA999(a0):
        a0 = getRangeValueT(CoefficientsTable, period, "Global_a0")

    c1 = getRangeValueT(CoefficientsTable, period, "c1")
    b4 = getRangeValueT(CoefficientsTable, period, "b4")
    Fp = c1 * LogR + (b4 * M) * R_Rref + a0 * r

    ##  Magnitude Scaling -------------------------------------------------------

    c4 = getRangeValueT(CoefficientsTable, period, "c4")
    c5 = getRangeValueT(CoefficientsTable, period, "c5")
    c6 = getRangeValueT(CoefficientsTable, period, "c6")

    Fm = c4 * func1(M, Mb) + c6 * func2(M, Mb) + c5 * func3(M, Mb)

    # # Linear Site Amplification ----------------------------------------------

    ## Site Coefficients
    V1 = getRangeValueT(CoefficientsTable, period, "V1_m_s")
    V2 = getRangeValueT(CoefficientsTable, period, "V2_m_s")
    Vref = getRangeValueT(CoefficientsTable, period, "Vref_m_s")

    if region == "global" or region == "CAM":
        s2 = getRangeValueT(CoefficientsTable, period, "Global_s2")
        s1 = s2
    elif region == "Taiwan" or region == "Japan":
        s2 = getRangeValueT(CoefficientsTable, period, region + "_s2")
        s1 = getRangeValueT(CoefficientsTable, period, region + "_s1")
    else:
        s2 = getRangeValueT(CoefficientsTable, period, region + "_s2")
        s1 = s2

    ##Compute linear site term
    if Vs30 <= V1:
        Flin = s1 * math.log(Vs30 / V1) + s2 * math.log(V1 / Vref)
    elif Vs30 <= V2:
        Flin = s2 * math.log(Vs30 / Vref)
    else:
        Flin = s2 * math.log(V2 / Vref)

    # # Nonlinear Site Term -----------------------------------------------------
    PGAr = math.exp(
        PSHAB20_GMM_at_760_IF(EventType, UserRegion, saturation_region, Rrup, M, 0)
    )

    f3 = 0.05
    Vb = 200
    Vref_Fnl = 900
    if period >= 3:
        Fnl = 0
    else:
        f4 = getRangeValueT(CoefficientsTable, period, "f4")
        f5 = getRangeValueT(CoefficientsTable, period, "f5")
        f2 = f4 * (
            math.exp(f5 * (min(Vs30, Vref_Fnl) - Vb)) - math.exp(f5 * (Vref_Fnl - Vb))
        )
        Fnl = 0 + f2 * math.log((PGAr + f3) / f3)

    ##  Basin Term --------------------------------------------------------------
    if str(Z2pt5) == "default":
        Fb = 0
    elif Z2pt5 <= 0 or (region != "Japan" and region != "Cascadia"):
        Fb = 0
    else:
        if region == "Cascadia":
            theta0 = 3.94
            theta1 = -0.42
            vmu = 200
            vsig = 0.2
            e1 = getRangeValueT(CoefficientsTable, period, "C_e1")

            if basin == 0:
                C_e3 = getRangeValueT(CoefficientsTable, period, "C_e3")
                C_e2 = getRangeValueT(CoefficientsTable, period, "C_e2")
                del_None = getRangeValueT(CoefficientsTable, period, "del_None")
                e3 = C_e3 + del_None
                e2 = C_e2 + del_None
            elif basin == 1:
                C_e3 = getRangeValueT(CoefficientsTable, period, "C_e3")
                C_e2 = getRangeValueT(CoefficientsTable, period, "C_e2")
                del_Seattle = getRangeValueT(CoefficientsTable, period, "del_Seattle")
                e3 = C_e3 + del_Seattle
                e2 = C_e2 + del_Seattle
            else:
                C_e3 = getRangeValueT(CoefficientsTable, period, "C_e3")
                C_e2 = getRangeValueT(CoefficientsTable, period, "C_e2")
                e3 = C_e3
                e2 = C_e2

        elif region == "Japan":
            theta0 = 3.05
            theta1 = -0.8
            vmu = 500
            vsig = 0.33
            e3 = getRangeValueT(CoefficientsTable, period, "J_e3")
            e2 = getRangeValueT(CoefficientsTable, period, "J_e2")
            e1 = getRangeValueT(CoefficientsTable, period, "J_e1")

        Z2pt5_pred = 10 ** (
            theta0
            + theta1
            * (1 + math.erf((Log10(Vs30) - Log10(vmu)) / (vsig * math.sqrt(2))))
        )
        delZ2pt5 = math.log(Z2pt5) - math.log(Z2pt5_pred)

        if delZ2pt5 <= (e1 / e3):
            Fb = e1
        elif delZ2pt5 >= (e2 / e3):
            Fb = e2
        else:
            Fb = e3 * delZ2pt5

    mu = c0 + Fp + Fm + Fnl + Fb + Flin
    return mu


def PSHAB20_Median(
    EventType,
    UserRegion,
    SubductionSlab,
    Rrup,
    Magnitude,
    Zhypo,
    period,
    Vs30,
    Z2pt5,
    PNWbasinStrux,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    thisOut = -999
    XinterpMin = 0
    XinterpMax = 10
    XinterpType = "log"
    YinterpType = "log"
    extrapolateType = "extrapolate"

    # Parameters = readtable('PSHAB20_Table_E2_Slab_Coefficients_OneRowHeader.csv','PreserveVariableNames',True);
    Parameters = pd.read_csv("PSHAB20_Table_E2_Slab_Coefficients_OneRowHeader.csv")
    TvalueList = Parameters["T"]

    InterpArray = interpolateFunction(
        period, TvalueList, XinterpType, extrapolateType, XinterpMin, XinterpMax
    )

    period0 = TvalueList[InterpArray[1]]
    y0 = PSHAB20_Median_AtTlist(
        EventType,
        UserRegion,
        SubductionSlab,
        Rrup,
        Magnitude,
        Zhypo,
        period0,
        Vs30,
        Z2pt5,
        PNWbasinStrux,
    )

    if InterpArray[2] <= 0:
        thisOut = y0
    else:
        period1 = TvalueList[InterpArray[2]]
        y1 = PSHAB20_Median_AtTlist(
            EventType,
            UserRegion,
            SubductionSlab,
            Rrup,
            Magnitude,
            Zhypo,
            period1,
            Vs30,
            Z2pt5,
            PNWbasinStrux,
        )
        if YinterpType.lower().startswith("log"):
            if y0 <= 0:
                y0 = 0.000000001
            if y1 <= 0:
                y1 = 0.000000001
            y0 = math.log(y0)
            y1 = math.log(y1)

        thisOut = y0 + (y1 - y0) * InterpArray[3]
        if YinterpType.lower().startswith("log"):
            thisOut = math.exp(thisOut)

    return thisOut


def PSHAB20_Median_AtTlist(
    EventType,
    UserRegion,
    saturation_region,
    Rrup,
    M,
    hypocentral_depth,
    period,
    Vs30,
    Z2pt5,
    basin,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    if EventType == 1 or EventType == 5:
        this = PSHAB20_GMM_at_VS30_Slab(
            EventType,
            UserRegion,
            saturation_region,
            Rrup,
            M,
            hypocentral_depth,
            period,
            Vs30,
            Z2pt5,
            basin,
        )
        PSHAB20_Median_AtTlistValue = math.exp(
            PSHAB20_GMM_at_VS30_Slab(
                EventType,
                UserRegion,
                saturation_region,
                Rrup,
                M,
                hypocentral_depth,
                period,
                Vs30,
                Z2pt5,
                basin,
            )
        )
    else:
        PSHAB20_Median_AtTlistValue = math.exp(
            PSHAB20_GMM_at_VS30_IF(
                EventType,
                UserRegion,
                saturation_region,
                Rrup,
                M,
                period,
                Vs30,
                Z2pt5,
                basin,
            )
        )

    return PSHAB20_Median_AtTlistValue


def PSHAB20_SigmaAleatory(period, Rrup, Vs30):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    thisOut = -999
    XinterpMin = 0
    XinterpMax = 10
    XinterpType = "log"
    YinterpType = "linear"
    extrapolateType = "extrapolate"

    #    Parameters = readtable('PSHAB20_Table_E2_Slab_Coefficients_OneRowHeader.csv','PreserveVariableNames',True);
    Parameters = pd.read_csv("PSHAB20_Table_E2_Slab_Coefficients_OneRowHeader.csv")

    TvalueList = Parameters["T"]
    InterpArray = interpolateFunction(
        period, TvalueList, XinterpType, extrapolateType, XinterpMin, XinterpMax
    )

    period0 = TvalueList[InterpArray[1]]
    y0 = PSHAB20_SigmaAleatoryAtTlist(period0, Rrup, Vs30)

    if InterpArray[2] <= 0:
        thisOut = y0
    else:
        period1 = TvalueList[InterpArray[2]]
        y1 = PSHAB20_SigmaAleatoryAtTlist(period1, Rrup, Vs30)
        if YinterpType.lower().startswith("log"):
            if y0 <= 0:
                y0 = 0.000000001
            if y1 <= 0:
                y1 = 0.000000001
            y0 = math.log(y0)
            y1 = math.log(y1)

        thisOut = y0 + (y1 - y0) * InterpArray[3]
        if YinterpType.lower().startswith("log"):
            thisOut = math.exp(thisOut)

    return thisOut


def PSHAB20_SigmaAleatoryAtTlist(period, Rrup, Vs30):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    PSHAB20_SigmaAleatoryAtTlist = -999

    PhiTot = PSHAB20_PhiTot(period, Rrup, Vs30)
    Tau = PSHAB20_Tau(period, Rrup, Vs30)

    PSHAB20_SigmaAleatoryAtTlistValue = math.sqrt(PhiTot * PhiTot + Tau * Tau)

    return PSHAB20_SigmaAleatoryAtTlistValue


def PSHAB20_Tau(period, Rrup, Vs30):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    PSHAB20_TauValue = -999
    #    SigmacoefficientsTable = readtable("PSHAB20_Table_E3_AleatoryCoefficients_OneRowHeader.csv",'PreserveVariableNames',True);
    SigmacoefficientsTable = pd.read_csv(
        "PSHAB20_Table_E3_AleatoryCoefficients_OneRowHeader.csv"
    )
    Tau = getRangeValueT(SigmacoefficientsTable, period, "Tau")
    return Tau


def PSHAB20_PhiTot(period, Rrup, Vs30):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    PSHAB20_PhiTotValue = -999
    #    SigmacoefficientsTable = readtable("PSHAB20_Table_E3_AleatoryCoefficients_OneRowHeader.csv",'PreserveVariableNames',True);
    SigmacoefficientsTable = pd.read_csv(
        "PSHAB20_Table_E3_AleatoryCoefficients_OneRowHeader.csv"
    )

    PhiTot_Phi1squared = getRangeValueT(
        SigmacoefficientsTable, period, "PhiTot_Phi1squared"
    )
    PhiTot_Phi2squared = getRangeValueT(
        SigmacoefficientsTable, period, "PhiTot_Phi2squared"
    )
    PhiTot_PhiVsquared = getRangeValueT(
        SigmacoefficientsTable, period, "PhiTot_PhiVsquared"
    )

    # Corner Distances:
    R1 = 200
    # km
    R2 = 500
    # km

    V1 = 200
    # m/s
    V2 = 500
    # m/s

    if Vs30 <= V1:
        Rprime = max(R1, min(R2, Rrup))
        deltaVar = PhiTot_PhiVsquared * (math.log(R2 / Rprime)) / (math.log(R2 / R1))
    elif Vs30 < V2:
        Rprime = max(R1, min(R2, Rrup))
        deltaVar = (
            PhiTot_PhiVsquared
            * (math.log(R2 / Rprime))
            / (math.log(R2 / R1))
            * (math.log(V2 / Vs30))
            / (math.log(V2 / V1))
        )
    else:
        deltaVar = 0

    if Rrup < R1:
        phiSquared = PhiTot_Phi1squared
    elif Rrup < R2:
        phiSquared = (PhiTot_Phi2squared - PhiTot_Phi1squared) / (math.log(R2 / R1)) * (
            math.log(Rrup / R1)
        ) + PhiTot_Phi1squared
    else:
        phiSquared = PhiTot_Phi2squared

    PSHAB20_PhiTotValue = math.sqrt(phiSquared + deltaVar)
    return PSHAB20_PhiTotValue


def PSHAB20_SigmaS2S(period, Rrup, Vs30):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    PSHAB20_SigmaS2SValue = -999
    #    SigmacoefficientsTable = readtable("PSHAB20_Table_E3_AleatoryCoefficients_OneRowHeader.csv",'PreserveVariableNames',True);
    SigmacoefficientsTable = pd.read_csv(
        "PSHAB20_Table_E3_AleatoryCoefficients_OneRowHeader.csv"
    )

    PhiS2S_PhiS2S0squared = getRangeValueT(
        SigmacoefficientsTable, period, "PhiS2S_PhiS2S0squared"
    )
    PhiS2S_a1 = getRangeValueT(SigmacoefficientsTable, period, "PhiS2S_a1")
    PhiS2S_VM = getRangeValueT(SigmacoefficientsTable, period, "PhiS2S_VM")

    # Corner Distances:
    R3 = 200
    # km
    R4 = 500
    # km

    V3 = 200
    # m/s
    V4 = 800
    # m/s

    if Vs30 <= V3:
        Rprime = max(R3, min(R4, Rrup))
        deltaVarS2S = (
            PhiS2S_a1
            * math.log(V3 / PhiS2S_VM)
            * (math.log(R4 / Rprime))
            / (math.log(R4 / R3))
        )
    elif Vs30 < PhiS2S_VM:
        Rprime = max(R3, min(R4, Rrup))
        deltaVarS2S = (
            PhiS2S_a1
            * math.log(Vs30 / PhiS2S_VM)
            * (math.log(R4 / Rprime))
            / (math.log(R4 / R3))
        )
    elif Vs30 < V4:
        deltaVarS2S = PhiS2S_a1 * math.log(Vs30 / PhiS2S_VM)
    else:
        deltaVarS2S = PhiS2S_a1 * math.log(V4 / PhiS2S_VM)

    PSHAB20_SigmaS2SValue = math.sqrt(PhiS2S_PhiS2S0squared + deltaVarS2S)
    return PSHAB20_SigmaS2SValue


def PSHAB20_SigmaSS(period, Rrup, Vs30):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    PSHAB20_SigmaSSValue = -999
    #    SigmacoefficientsTable = readtable("PSHAB20_Table_E3_AleatoryCoefficients_OneRowHeader.csv",'PreserveVariableNames',True);
    SigmacoefficientsTable = pd.read_csv(
        "PSHAB20_Table_E3_AleatoryCoefficients_OneRowHeader.csv"
    )

    PhiSS_PhiSS1squared = getRangeValueT(
        SigmacoefficientsTable, period, "PhiSS_PhiSS1squared"
    )
    PhiSS_PhiSS2squared = getRangeValueT(
        SigmacoefficientsTable, period, "PhiSS_PhiSS2squared"
    )
    PhiSS_a2 = getRangeValueT(SigmacoefficientsTable, period, "PhiSS_a2")
    PhiSS_VM = getRangeValueT(SigmacoefficientsTable, period, "PhiSS_VM")

    # Corner Distances:
    R3 = 200
    # km
    R4 = 500
    # km
    R5 = 500
    # km
    R6 = 800
    # km

    V3 = 200
    # m/s
    V4 = 800
    # m/s

    if Rrup < R5:
        phiSquaredSS = PhiSS_PhiSS1squared
    elif Rrup < R6:
        phiSquaredSS = (PhiSS_PhiSS2squared - PhiSS_PhiSS1squared) / (
            math.log(R6 / R5)
        ) * (math.log(Rrup / R5)) + PhiSS_PhiSS1squared
    else:
        phiSquaredSS = PhiSS_PhiSS2squared

    if Vs30 <= V3:
        Rprime = max(R3, min(R4, Rrup))
        deltaVarSS = (
            PhiSS_a2
            * math.log(V3 / PhiSS_VM)
            * (math.log(R4 / Rprime))
            / (math.log(R4 / R3))
        )
    elif Vs30 < PhiSS_VM:
        Rprime = max(R3, min(R4, Rrup))
        deltaVarSS = (
            PhiSS_a2
            * math.log(Vs30 / PhiSS_VM)
            * (math.log(R4 / Rprime))
            / (math.log(R4 / R3))
        )
    elif Vs30 < V4:
        deltaVarSS = PhiSS_a2 * math.log(Vs30 / PhiSS_VM)
    else:
        deltaVarSS = PhiSS_a2 * math.log(V4 / PhiSS_VM)

    PSHAB20_SigmaSSValue = math.sqrt(phiSquaredSS + deltaVarSS)
    return PSHAB20_SigmaSSValue


def NGAsubGMM_Median(
    UserPeriod,
    UserRegion,
    Magnitude,
    Vs30,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    Zhypo,
    EventType,
    Z1pt0,
    Z2pt5,
    MbUser,
    SubductionSlab,
    PNWbasinStrux,
    Weight_KBCG20,
    Weight_PSHAB20,
    User_MeanType="geometric",
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    NGAsubGMM_MedianValue = 0

    if str(MbUser).lower() == "default".lower():
        Mb = getMbDefault(EventType, UserRegion, SubductionSlab)
    else:
        Mb = MbUser

    period = getTvalue(UserPeriod)
    WeightedSum = 0
    WeightSum = 0

    if Weight_KBCG20 > 0:
        KBCG20 = KBCG20_medPSA(
            period,
            Magnitude,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            Vs30,
            Z1pt0,
            Z2pt5,
            Mb,
            UserRegion,
            PNWbasinStrux,
        )
        WeightSum = WeightSum + Weight_KBCG20
        if User_MeanType.lower() == "arithmetic".lower():
            WeightedSum = WeightedSum + KBCG20 * Weight_KBCG20
        else:
            WeightedSum = WeightedSum + math.log(KBCG20) * Weight_KBCG20

    if Weight_PSHAB20 > 0:
        PSHAB20 = PSHAB20_Median(
            EventType,
            UserRegion,
            SubductionSlab,
            Rrup,
            Magnitude,
            Zhypo,
            period,
            Vs30,
            Z2pt5,
            PNWbasinStrux,
        )
        WeightSum = WeightSum + Weight_PSHAB20
        if User_MeanType.lower() == "arithmetic".lower():
            WeightedSum = WeightedSum + PSHAB20 * Weight_PSHAB20
        else:
            WeightedSum = WeightedSum + math.log(PSHAB20) * Weight_PSHAB20

    if WeightSum > 0:
        if User_MeanType.lower() == "arithmetic".lower():
            NGAsubGMM_MedianValue = WeightedSum / WeightSum
        else:
            NGAsubGMM_MedianValue = math.exp(WeightedSum / WeightSum)

    return NGAsubGMM_MedianValue


def NGAsubGMM_SigmaTotal(
    UserPeriod,
    UserRegion,
    Magnitude,
    Vs30,
    Rrup,
    AlphaBackarc,
    AlphaNankai,
    Ztor,
    Zhypo,
    EventType,
    Z1pt0,
    Z2pt5,
    MbUser,
    SubductionSlab,
    PNWbasinStrux,
    Weight_KBCG20,
    Weight_PSHAB20,
    User_AleInSigmaModels,
    User_EpiInSigmaModels,
    User_NsampleEpi,
):
    # CODE DEVELOPED/IMPLEMENTED BY
    #          Silvia Mazzoni, 2020
    #           smazzoni@ucla.edu

    NGAsubGMM_SigmaTotalValue = -999

    if str(MbUser).lower() == "default".lower():
        Mb = getMbDefault(EventType, UserRegion, SubductionSlab)
    else:
        Mb = MbUser

    period = getTvalue(UserPeriod)

    if User_EpiInSigmaModels == 0:
        SigmaEpistemicKBCG20 = 0
        SigmaEpistemicPSHAB20 = 0
    elif User_EpiInSigmaModels == 1:
        # all models
        SigmaEpistemicKBCG20 = KBCG20_SigmaEpistemic(
            period,
            Magnitude,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            Vs30,
            Z1pt0,
            Z2pt5,
            Mb,
            UserRegion,
            User_NsampleEpi,
            PNWbasinStrux,
        )
        SigmaEpistemicPSHAB20 = SigmaEpistemicKBCG20
    elif User_EpiInSigmaModels == 2:
        ## KBCG20 only
        SigmaEpistemicKBCG20 = KBCG20_SigmaEpistemic(
            period,
            Magnitude,
            Rrup,
            AlphaBackarc,
            AlphaNankai,
            Ztor,
            EventType,
            Vs30,
            Z1pt0,
            Z2pt5,
            Mb,
            UserRegion,
            User_NsampleEpi,
            PNWbasinStrux,
        )
        SigmaEpistemicPSHAB20 = 0

    if User_AleInSigmaModels == 1:
        SigmaAleatoryKBCG20 = KBCG20_sigmaAleatory(period)
        SigmaAleatoryPSHAB20 = PSHAB20_SigmaAleatory(period, Rrup, Vs30)
    else:
        SigmaAleatoryKBCG20 = 0
        SigmaAleatoryPSHAB20 = 0

    KBCG20SigmaTotal = math.sqrt(
        SigmaAleatoryKBCG20 * SigmaAleatoryKBCG20
        + SigmaEpistemicKBCG20 * SigmaEpistemicKBCG20
    )
    PSHAB20SigmaTotal = math.sqrt(
        SigmaAleatoryPSHAB20 * SigmaAleatoryPSHAB20
        + SigmaEpistemicPSHAB20 * SigmaEpistemicPSHAB20
    )
    NGAsubGMM_SigmaTotalValue = math.sqrt(
        Weight_PSHAB20 * PSHAB20SigmaTotal * PSHAB20SigmaTotal
        + Weight_KBCG20 * KBCG20SigmaTotal * KBCG20SigmaTotal
    ) / (Weight_PSHAB20 + Weight_KBCG20)
    return NGAsubGMM_SigmaTotalValue


# #############################################################
# the following commands compute the output values at each period and generate a figure of the data
# The user may edit these lines, if needed
# #############################################################
# Compute Values at each Period
def runScenarioStudy(
    User_Region,
    User_SubductionSlab,
    User_Magnitude,
    User_Vs30,
    User_Rrup,
    User_AlphaBackarc,
    User_AlphaNankai,
    User_Ztor,
    User_Zhypo,
    User_EventType,
    User_Z1pt0,
    User_Z2pt5,
    User_Mb,
    User_PNWbasinStrux,
    User_RelativeWeight_KBCG20,
    User_RelativeWeight_PSHAB20,
    User_AleInSigmaModels,
    User_EpiInSigmaModels,
    User_NsampleEpi,
    User_Nsigma,
    User_PeriodList,
):
    UserInputArray = [
        User_Region,
        User_SubductionSlab,
        User_Magnitude,
        User_Vs30,
        User_Rrup,
        User_AlphaBackarc,
        User_AlphaNankai,
        User_Ztor,
        User_Zhypo,
        User_EventType,
        User_Z1pt0,
        User_Z2pt5,
        User_Mb,
        User_PNWbasinStrux,
        User_RelativeWeight_KBCG20,
        User_RelativeWeight_PSHAB20,
        User_AleInSigmaModels,
        User_EpiInSigmaModels,
        User_NsampleEpi,
        User_Nsigma,
    ]
    outTlist = []
    outMedianList = []
    outSigmaList = []
    outMedianMinusNSigmaList = []
    outMedianPlusNSigmaList = []
    outWeightedAvgData = []
    outIndivModelData = []

    # print('Period_(s)','PSA_median_(g)','Sigma','PSA_median-N*Sigma_(g)','PSA_median+N*Sigma_(g)')
    for irow in range(len(User_PeriodList)):
        thisT = getTvalue(User_PeriodList[irow])
        outTlist.append(thisT)

        PSA = NGAsubGMM_Median(
            thisT,
            User_Region,
            User_Magnitude,
            User_Vs30,
            User_Rrup,
            User_AlphaBackarc,
            User_AlphaNankai,
            User_Ztor,
            User_Zhypo,
            User_EventType,
            User_Z1pt0,
            User_Z2pt5,
            User_Mb,
            User_SubductionSlab,
            User_PNWbasinStrux,
            User_RelativeWeight_KBCG20,
            User_RelativeWeight_PSHAB20,
        )
        Sigma = NGAsubGMM_SigmaTotal(
            thisT,
            User_Region,
            User_Magnitude,
            User_Vs30,
            User_Rrup,
            User_AlphaBackarc,
            User_AlphaNankai,
            User_Ztor,
            User_Zhypo,
            User_EventType,
            User_Z1pt0,
            User_Z2pt5,
            User_Mb,
            User_SubductionSlab,
            User_PNWbasinStrux,
            User_RelativeWeight_KBCG20,
            User_RelativeWeight_PSHAB20,
            User_AleInSigmaModels,
            User_EpiInSigmaModels,
            User_NsampleEpi,
        )
        outMedianMinusNSigma = math.exp(math.log(PSA) - User_Nsigma * Sigma)
        outMedianPlusNSigma = math.exp(math.log(PSA) + User_Nsigma * Sigma)
        outWeightedAvgData.append(
            [thisT, PSA, Sigma, outMedianMinusNSigma, outMedianPlusNSigma]
        )
        # print(thisT,PSA,Sigma,outMedianMinusNSigma,outMedianPlusNSigma)

        #### Individual-Model Data:
        Prog_MbDefault = getMbDefault(User_EventType, User_Region, User_SubductionSlab)
        if str(User_Mb).lower() == "default".lower():
            inUser_Mb = Prog_MbDefault
        else:
            inUser_Mb = User_Mb

        outKBCG20_MedianPSA = KBCG20_medPSA(
            thisT,
            User_Magnitude,
            User_Rrup,
            User_AlphaBackarc,
            User_AlphaNankai,
            User_Ztor,
            User_EventType,
            User_Vs30,
            User_Z1pt0,
            User_Z2pt5,
            inUser_Mb,
            User_Region,
            User_PNWbasinStrux,
        )
        outKBCG20_Tau = KBCG20_sigmaTau(thisT)
        outKBCG20_Phi = KBCG20_sigmaPhi(thisT)
        outKBCG20_SigmaAleatory = KBCG20_sigmaAleatory(thisT)
        outKBCG20_SigmaEpistemic = KBCG20_SigmaEpistemic(
            thisT,
            User_Magnitude,
            User_Rrup,
            User_AlphaBackarc,
            User_AlphaNankai,
            User_Ztor,
            User_EventType,
            User_Vs30,
            User_Z1pt0,
            User_Z2pt5,
            inUser_Mb,
            User_Region,
            User_NsampleEpi,
            User_PNWbasinStrux,
        )
        outKBCG20_SigmaTotal = KBCG20_SigmaTotal(
            thisT,
            User_Magnitude,
            User_Rrup,
            User_AlphaBackarc,
            User_AlphaNankai,
            User_Ztor,
            User_EventType,
            User_Vs30,
            User_Z1pt0,
            User_Z2pt5,
            inUser_Mb,
            User_Region,
            User_NsampleEpi,
            User_PNWbasinStrux,
        )
        outKBCG20_Median_minus_1_SigmaTotal = math.exp(
            math.log(outKBCG20_MedianPSA) - User_Nsigma * outKBCG20_SigmaTotal
        )
        outKBCG20_Median_plus_1_SigmaTotal = math.exp(
            math.log(outKBCG20_MedianPSA) + User_Nsigma * outKBCG20_SigmaTotal
        )
        outKBCG20_Median_minus_1_SigmaAleatory = math.exp(
            math.log(outKBCG20_MedianPSA) - User_Nsigma * outKBCG20_SigmaAleatory
        )
        outKBCG20_Median_plus_1_SigmaAleatory = math.exp(
            math.log(outKBCG20_MedianPSA) + User_Nsigma * outKBCG20_SigmaAleatory
        )
        outKBCG20_Median_minus_1_SigmaEpistemic = math.exp(
            math.log(outKBCG20_MedianPSA) - User_Nsigma * outKBCG20_SigmaEpistemic
        )
        outKBCG20_Median_plus_1_SigmaEpistemic = math.exp(
            math.log(outKBCG20_MedianPSA) + User_Nsigma * outKBCG20_SigmaEpistemic
        )
        outPSHAB20_MedianPSA = PSHAB20_Median(
            User_EventType,
            User_Region,
            User_SubductionSlab,
            User_Rrup,
            User_Magnitude,
            User_Zhypo,
            thisT,
            User_Vs30,
            User_Z2pt5,
            User_PNWbasinStrux,
        )
        outPSHAB20_Tau = PSHAB20_Tau(thisT, User_Rrup, User_Vs30)
        outPSHAB20_PhiTot = PSHAB20_PhiTot(thisT, User_Rrup, User_Vs30)
        outPSHAB20_PhiS2S = PSHAB20_SigmaS2S(thisT, User_Rrup, User_Vs30)
        outPSHAB20_PhiSS = PSHAB20_SigmaSS(thisT, User_Rrup, User_Vs30)
        outPSHAB20_SigmaAleatory = PSHAB20_SigmaAleatory(thisT, User_Rrup, User_Vs30)
        outPSHAB20_SigmaTotal = math.sqrt(
            outKBCG20_SigmaEpistemic ** 2 + outPSHAB20_SigmaAleatory ** 2
        )
        outPSHAB20_Median_minus_1_SigmaTotal = math.exp(
            math.log(outPSHAB20_MedianPSA) - User_Nsigma * outPSHAB20_SigmaTotal
        )
        outPSHAB20_Median_plus_1_SigmaTotal = math.exp(
            math.log(outPSHAB20_MedianPSA) + User_Nsigma * outPSHAB20_SigmaTotal
        )
        outPSHAB20_Median_minus_1_SigmaAleatory = math.exp(
            math.log(outPSHAB20_MedianPSA) - User_Nsigma * outPSHAB20_SigmaAleatory
        )
        outPSHAB20_Median_plus_1_SigmaAleatory = math.exp(
            math.log(outPSHAB20_MedianPSA) + User_Nsigma * outPSHAB20_SigmaAleatory
        )
        outPSHAB20_Median_minus_1_SigmaEpistemic = math.exp(
            math.log(outPSHAB20_MedianPSA) - User_Nsigma * outKBCG20_SigmaEpistemic
        )
        outPSHAB20_Median_plus_1_SigmaEpistemic = math.exp(
            math.log(outPSHAB20_MedianPSA) + User_Nsigma * outKBCG20_SigmaEpistemic
        )
        outIndivModelData.append(
            [
                thisT,
                outKBCG20_MedianPSA,
                outKBCG20_Tau,
                outKBCG20_Phi,
                outKBCG20_SigmaAleatory,
                outKBCG20_SigmaEpistemic,
                outKBCG20_SigmaTotal,
                outKBCG20_Median_minus_1_SigmaTotal,
                outKBCG20_Median_plus_1_SigmaTotal,
                outKBCG20_Median_minus_1_SigmaAleatory,
                outKBCG20_Median_plus_1_SigmaAleatory,
                outKBCG20_Median_minus_1_SigmaEpistemic,
                outKBCG20_Median_plus_1_SigmaEpistemic,
                outPSHAB20_MedianPSA,
                outPSHAB20_Tau,
                outPSHAB20_PhiTot,
                outPSHAB20_PhiS2S,
                outPSHAB20_PhiSS,
                outPSHAB20_SigmaAleatory,
                outPSHAB20_SigmaTotal,
                outPSHAB20_Median_minus_1_SigmaTotal,
                outPSHAB20_Median_plus_1_SigmaTotal,
                outPSHAB20_Median_minus_1_SigmaAleatory,
                outPSHAB20_Median_plus_1_SigmaAleatory,
                outPSHAB20_Median_minus_1_SigmaEpistemic,
                outPSHAB20_Median_plus_1_SigmaEpistemic,
            ]
        )

    print("Scenario Input Data:")
    UserInputDataRows = [
        "Region",
        "SubductionSlab",
        "Magnitude",
        "Vs30",
        "Rrup",
        "AlphaBackarc",
        "AlphaNankai",
        "Ztor",
        "Zhypo",
        "EventType",
        "Z1pt0",
        "Z2pt5",
        "Mb",
        "PNWbasinStrux",
        "RelativeWeight_KBCG20",
        "RelativeWeight_PSHAB20",
        "AleInSigmaModels",
        "EpiInSigmaModels",
        "NsampleEpi",
        "Nsigma",
    ]
    UserInputTable = pd.DataFrame(UserInputArray, index=UserInputDataRows)
    UserInputTable.columns = ["User Input"]
    UserInputTable.index = UserInputDataRows
    print(UserInputTable)
    print("Scenario Weighted-Average Data:")
    outWeightedAvgTable = pd.DataFrame(outWeightedAvgData)
    outWeightedAvgTable.columns = [
        "Period_(s)",
        "PSA_median_(g)",
        "Sigma",
        "PSA_median-N*Sigma_(g)",
        "PSA_median+N*Sigma_(g)",
    ]
    blankIndex = [""] * len(outWeightedAvgTable)
    outWeightedAvgTable.index = blankIndex
    print(outWeightedAvgTable)
    print("Scenario Individual-Model Data:")
    outIndivModelTable = pd.DataFrame(outIndivModelData)
    outIndivModelTable.columns = [
        "Period_s",
        "KBCG20_MedianPSA",
        "KBCG20_Tau",
        "KBCG20_Phi",
        "KBCG20_SigmaAleatory",
        "KBCG20_SigmaEpistemic",
        "KBCG20_SigmaTotal",
        "KBCG20_Median_minus_1*SigmaTotal",
        "KBCG20_Median_plus_1*SigmaTotal",
        "KBCG20_Median_minus_1*SigmaAleatory",
        "KBCG20_Median_plus_1*SigmaAleatory",
        "KBCG20_Median_minus_1*SigmaEpistemic",
        "KBCG20_Median_plus_1*SigmaEpistemic",
        "PSHAB20_MedianPSA",
        "PSHAB20_Tau",
        "PSHAB20_PhiTot",
        "PSHAB20_PhiS2S",
        "PSHAB20_PhiSS",
        "PSHAB20_SigmaAleatory",
        "PSHAB20_SigmaTotal",
        "PSHAB20_Median_minus_1*SigmaTotal",
        "PSHAB20_Median_plus_1*SigmaTotal",
        "PSHAB20_Median_minus_1*SigmaAleatory",
        "PSHAB20_Median_plus_1*SigmaAleatory",
        "PSHAB20_Median_minus_1*SigmaEpistemic",
        "PSHAB20_Median_plus_1*SigmaEpistemic",
    ]
    blankIndex = [""] * len(outIndivModelTable)
    outIndivModelTable.index = blankIndex
    print(outIndivModelTable)


# #############################################################
# #############################################################
# #############################################################
############### USER INPUT
# #############################################################
# #############################################################
# #############################################################

# Region (Options: 0_global, 1_Alaska, 2_Cascadia, 3_CentralAmerica&Mexico, 4_Japan, 5_NewZealand (PSHAB20 uses global model), 6_SouthAmerica, 7_Taiwan)
# Subducting Slab (used to compute Mb) (global,Alaska,Aleutian,Cascadia,Central_America_N,Central_America_S,Japan_Pac,Japan_Phi,New_Zealand_N,New_Zealand_S,South_America_N,South_America_S,Taiwan_E,Taiwan_W)
# Moment Magnitude
# VS30 (m/sec)
# Rupture Distance (km)
# Fraction of Rrup in Backarc (Range: 0-1)
# Fraction of Rrup in Nankai Region (Range: 0-(1-AlphaBackArc))(Japan only)
# Z_tor (km) (KBCG20 only)
# Hypocentral Depth (km) (PSHAB20 only)
# Event Type, (Options: 0_Interface, 1_Intraslab)
# Z1.0 input (m) (KBCG20 only)
# Z2.5 input (m)
# Mb (KBCG20 only, can set = "default" or value [units?])
# PNW Basin Structure (Options: False (NoBasin), True (InSeattleBasin, Cascadia Only)
# Relative Weight -- KBCG20
# Relative Weight -- PSHAB20
# Include Aleatory Variability (Options: 0_None, 1_AllModels)
# Apply KBCG20 Epistemic to Sigma Models (Options: 0_None, 1_AllModels, 2_KBCG20_only)
# Number of Samples in Epistemic-Uncertainty Calculation (0=none, Range: 100-800)
# Number of Sigma away from Median
# Periods range:0.01-10sec (PGA: T=0, PGV: T=-1)
User_Region = 4
User_SubductionSlab = "Japan_Pac"
User_Magnitude = 7.2
User_Vs30 = 760
User_Rrup = 200
User_AlphaBackarc = 0.5
User_AlphaNankai = 0
User_Ztor = 10
User_Zhypo = 55
User_EventType = 1
User_Z1pt0 = 550
User_Z2pt5 = 2000
User_Mb = "default"
User_PNWbasinStrux = False
User_RelativeWeight_KBCG20 = 1
User_RelativeWeight_PSHAB20 = 1
User_AleInSigmaModels = 1
User_EpiInSigmaModels = 1
User_NsampleEpi = 100
User_Nsigma = 1
User_PeriodList = [
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
    "PGA",
    "PGV",
]

runScenarioStudy(
    User_Region,
    User_SubductionSlab,
    User_Magnitude,
    User_Vs30,
    User_Rrup,
    User_AlphaBackarc,
    User_AlphaNankai,
    User_Ztor,
    User_Zhypo,
    User_EventType,
    User_Z1pt0,
    User_Z2pt5,
    User_Mb,
    User_PNWbasinStrux,
    User_RelativeWeight_KBCG20,
    User_RelativeWeight_PSHAB20,
    User_AleInSigmaModels,
    User_EpiInSigmaModels,
    User_NsampleEpi,
    User_Nsigma,
    User_PeriodList,
)
