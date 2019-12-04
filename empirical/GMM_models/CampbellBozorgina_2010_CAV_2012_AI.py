import numpy as np
import copy
from empirical.util.classdef import estimate_z1p0, estimate_z2p5, Orientation

"""
Provides the attenuation relation for AI in units of cm/s and CAV in gs

Translated from CampbellBorzorgina_2012_AI.m &
Translated from CampbellBorzorgina_2010_CAV.m

Input Variables:
 M             = Moment magnitude (Mw)
 Rrup          = 'closest distance coseismic rupture (km)
 siteprop      = properties of site (soil etc)
                 siteprop.Rjb -Source-to-site distance (km) (Joyner Boore
                 distance)
                 siteprop.vs30   -'(any real variable)' shear wave velocity(m/s)
                 siteprop.Zvs -'depth to the 2.5km/s shear wave velocity
                                  horizon (Uses the V30 ~ Z2p5 estimation)
                 siteprop.orientation -'random'
                                      -'average'

 faultprop     = properties of fault (strikeslip etc)
                 faultprop.Ztor -'depth to top of coseismic rupture (km)
                 faultprop.rake -'rake angle in degrees
                 faultprop.dip -'avg dip angle in degrees
 im_name       = if = 0 or 'AI'  then calculates AI value
                 if = 1 or 'CAV' then calculates CAV value
                 if = 2 then calculates A1100 value
                

Output Variables:
 IM_Value           = median AI / CAV
 sigma        = lognormal standard deviation of IM
                 sigma_AI(1) = total std
                 sigma_AI(2) = interevent std
                 sigma_AI(3) = intraevent std

"""


def CampbellBozorgina_2012(siteprop, faultprop, im_name):
    if im_name == "AI":
        i = 0
    elif im_name == "CAV":
        i = 1
    else:
        i = im_name

    # Coefficients
    #    AI           CAV       PGA
    c0 = [1.056,      -4.354,   -1.715]
    c1 = [1.215,      0.942,     0.500]
    c2 = [-0.403,     -0.178,   -0.530]
    c3 = [-1.003,     -0.346,   -0.262]
    c4 = [-3.566,     -1.309,   -2.118]
    c5 = [0.295,      0.087,     0.170]
    c6 = [5.87,       7.24,       5.60]
    c7 = [0.486,      0.111,     0.280]
    c8 = [-0.219,     -0.108,   -0.120]
    c9 = [0.837,      0.362,     0.490]
    c10 = [5.404,     2.549,     1.058]
    c11 = [0.068,     0.090,     0.040]
    c12 = [2.311,     1.277,     0.610]
    k1 = [349.0,      400.0,     865.0]
    k2 = [-5.312,     -2.690,   -1.186]
    k3 = [1.0,        1.0,       1.839]
    sAF = [0.623,     0.3,         0.3]
    slny = [0.771,    0.371,     0.478]
    tlny = [0.309,    0.196,     0.219]
    sigmac = [0.214,  0.089,     0.166]
    rho = [0.891,     0.735,     1.000]

    c = 1.88
    n = 1.18

    M = faultprop.Mw * 1.0
    Rrup = siteprop.Rrup

    Rjb = siteprop.Rjb
    Ztor = faultprop.ztor
    rake = faultprop.rake
    delta = faultprop.dip
    V30 = siteprop.vs30
    Z_2p5 = siteprop.z2p5
    siteprop.orientation = Orientation.AVERAGE

    # Magnitude dependence
    if M <= 5.5:
        fmag = c0[i] + c1[i] * M
    elif M <= 6.5:
        fmag = c0[i] + c1[i] * M + c2[i] * (M - 5.5)
    else:
        fmag = c0[i] + c1[i] * M + c2[i] * (M - 5.5) + c3[i] * (M - 6.5)

    # distance dependance
    fdis = (c4[i] + c5[i] * M) * np.log(np.sqrt(Rrup ** 2 + c6[i] ** 2))

    # faulting style
    if Ztor < 1:
        ffltz = Ztor
    else:
        ffltz = 1.0

    Frv = 30 < rake < 150
    Fnm = -150 < rake < -30

    fflt = c7[i] * Frv * ffltz + c8[i] * Fnm

    # Hanging-wall effects

    if Rjb == 0:
        fhngr = 1
    else:
        if Ztor < 1:
            fhngr = (max(Rrup, np.sqrt(Rjb ** 2 + 1)) - Rjb) / max(
                Rrup, np.sqrt(Rjb ** 2 + 1)
            )
        else:
            fhngr = (Rrup - Rjb) / Rrup

    if M <= 6:
        fhngm = 0.0
    else:
        if M < 6.5:
            fhngm = 2.0 * (M - 6)
        else:
            fhngm = 1.0

    fhngz = ((20.0 - Ztor) / 20.0) * (0 <= Ztor < 20)
    fhngdelta = (delta <= 70) + ((90.0 - delta) / 20) * (delta > 70)
    fhng = c9[i] * fhngr * fhngm * fhngz * fhngdelta

    # Site conditions

    if V30 < k1[i]:
        # get A1100
        rock_site = copy.deepcopy(siteprop)
        rock_site.vs30 = 1100
        rock_site.z1p0 = estimate_z1p0(rock_site.vs30)
        rock_site.z2p5 = estimate_z2p5(rock_site.z1p0)
        A1100 = CampbellBozorgina_2012(rock_site, faultprop, 2)[0]

        fsite = c10[i] * np.log(V30 / k1[i]) + k2[i] * (
            np.log(A1100 + c * (V30 / k1[i]) ** n) - np.log(A1100 + c)
        )
    else:
        fsite = (c10[i] + k2[i] * n) * np.log(min(V30, 1100.0) / k1[i])

    # Sediment effects
    if Z_2p5 < 1:
        fsed = c11[i] * (Z_2p5 - 1)
    elif Z_2p5 <= 3:
        fsed = 0.0
    else:
        fsed = c12[i] * k3[i] * np.exp(-0.75) * (1 - np.exp(-0.25 * (Z_2p5 - 3.0)))

    # Median value
    IM_value = np.exp(fmag + fdis + fflt + fhng + fsite + fsed)

    # Standard deviation computations
    if V30 < k1[i]:
        alpha1 = (
            k2[i]
            * A1100
            * ((A1100 + c * (V30 / k1[i]) ** n) ** (-1) - (A1100 + c) ** (-1))
        )
    else:
        alpha1 = 0

    sigma = np.sqrt(
        (slny[i] ** 2 - sAF[i] ** 2)
        + sAF[i] ** 2
        + alpha1 ** 2 * slny[1] ** 2
        + 2 * alpha1 * rho[i] * np.sqrt(slny[i] ** 2 - sAF[i] ** 2) * slny[1]
    )

    sigma_IM = [None] * 3
    if siteprop.orientation == Orientation.RANDOM:  # random/arbitrary component
        sigma_IM[0] = np.sqrt(tlny[i] ** 2 + sigma ** 2 + sigmac[i] ** 2)
        sigma_IM[1] = tlny[i]
        sigma_IM[2] = np.sqrt(sigma ** 2 + sigmac[i] ** 2)
    else:
        sigma_IM[0] = np.sqrt(tlny[i] ** 2 + sigma ** 2)
        sigma_IM[1] = tlny[i]
        sigma_IM[2] = sigma

    return IM_value, sigma_IM
