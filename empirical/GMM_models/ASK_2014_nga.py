
import math

#import numba
import numpy as np

from empirical.util.classdef import interpolate_to_closest

# fmt: off
periods = np.array([0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6, 7.5, 10, 0, -1])
# coefficients
Vlin = np.array([660.0000, 680.0000, 770.0000, 915.0000, 960.0000, 910.0000, 740.0000, 590.0000, 495.0000, 430.0000, 360.0000, 340.0000, 330.0000, 330.0000, 330.0000, 330.0000, 330.0000, 330.0000, 330.0000, 330.0000, 330.0000, 330.0000, 660.0000, 330.0000])
b = np.array([-1.4700, -1.4590, -1.3900, -1.2190, -1.1520, -1.2300, -1.5870, -2.0120, -2.4110, -2.7570, -3.2780, -3.5990, -3.8000, -3.5000, -2.4000, -1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -1.4700, -2.0200])
n = np.array([1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000, 1.5000])
m1 = np.array([6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.7500, 6.8200, 6.9200, 7.0000, 7.0600, 7.1450, 7.2500, 6.7500, 6.7500])
c = np.array([2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2.4000, 2400.0000])
c4 = np.array([4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000, 4.5000])
a1 = np.array([0.5870, 0.5980, 0.6020, 0.7070, 0.9730, 1.1690, 1.4420, 1.6370, 1.7010, 1.7120, 1.6620, 1.5710, 1.2990, 1.0430, 0.6650, 0.3290, -0.0600, -0.2990, -0.5620, -0.8750, -1.3030, -1.9280, 0.5870, 5.9750])
a2 = np.array([-0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7900, -0.7650, -0.7110, -0.6340, -0.5290, -0.7900, -0.9190])
a3 = np.array([0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750, 0.2750])
a4 = np.array([-0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000])
a5 = np.array([-0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100, -0.4100])
a6 = np.array([2.1541, 2.1461, 2.1566, 2.0845, 2.0285, 2.0408, 2.1208, 2.2241, 2.3124, 2.3383, 2.4688, 2.5586, 2.6821, 2.7630, 2.8355, 2.8973, 2.9061, 2.8888, 2.8984, 2.8955, 2.8700, 2.8431, 2.1541, 2.3657])
a7 = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
a8 = np.array([-0.0150, -0.0150, -0.0150, -0.0150, -0.0150, -0.0150, -0.0220, -0.0300, -0.0380, -0.0450, -0.0550, -0.0650, -0.0950, -0.1100, -0.1240, -0.1380, -0.1720, -0.1970, -0.2180, -0.2350, -0.2550, -0.2850, -0.0150, -0.0940])
a10 = np.array([1.7350, 1.7180, 1.6150, 1.3580, 1.2580, 1.3100, 1.6600, 2.2200, 2.7700, 3.2500, 3.9900, 4.4500, 4.7500, 4.3000, 2.6000, 0.5500, -0.9500, -0.9500, -0.9300, -0.9100, -0.8700, -0.8000, 1.7350, 2.3600])
a11 = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
a12 = np.array([-0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.1000, -0.2000, -0.2000, -0.2000, -0.1000, -0.1000])
a13 = np.array([0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.6000, 0.5800, 0.5600, 0.5300, 0.5000, 0.4200, 0.3500, 0.2000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6000, 0.2500])
a14 = np.array([-0.3000, -0.3000, -0.3000, -0.3000, -0.3000, -0.3000, -0.3000, -0.3000, -0.2400, -0.1900, -0.1100, -0.0400, 0.0700, 0.1500, 0.2700, 0.3500, 0.4600, 0.5400, 0.6100, 0.6500, 0.7200, 0.8000, -0.3000, 0.2200])
a15 = np.array([1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.1000, 1.0300, 0.9200, 0.8400, 0.6800, 0.5700, 0.4200, 0.3100, 0.1600, 0.0500, -0.0400, -0.1100, -0.1900, -0.3000, 1.1000, 0.3000])
a17 = np.array([-0.0072, -0.0073, -0.0075, -0.0080, -0.0089, -0.0095, -0.0095, -0.0086, -0.0074, -0.0064, -0.0043, -0.0032, -0.0025, -0.0025, -0.0022, -0.0019, -0.0015, -0.0010, -0.0010, -0.0010, -0.0010, -0.0010, -0.0072, -0.0005])
a43 = np.array([0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1000, 0.1400, 0.1700, 0.2200, 0.2600, 0.3400, 0.4100, 0.5100, 0.5500, 0.4900, 0.4200, 0.1000, 0.2800])
a44 = np.array([0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0700, 0.1000, 0.1400, 0.1700, 0.2100, 0.2500, 0.3000, 0.3200, 0.3200, 0.3200, 0.2750, 0.2200, 0.0500, 0.1500])
a45 = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0300, 0.0600, 0.1000, 0.1400, 0.1700, 0.2000, 0.2200, 0.2300, 0.2300, 0.2200, 0.2000, 0.1700, 0.1400, 0.0000, 0.0900])
a46 = np.array([-0.0500, -0.0500, -0.0500, -0.0500, -0.0500, -0.0500, -0.0500, -0.0300, 0.0000, 0.0300, 0.0600, 0.0900, 0.1300, 0.1400, 0.1600, 0.1600, 0.1600, 0.1400, 0.1300, 0.1000, 0.0900, 0.0800, -0.0500, 0.0700])
a25 = np.array([-0.0015, -0.0015, -0.0016, -0.0020, -0.0027, -0.0033, -0.0035, -0.0033, -0.0029, -0.0027, -0.0023, -0.0020, -0.0010, -0.0005, -0.0004, -0.0002, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, -0.0015, -0.0001])
a28 = np.array([0.0025, 0.0024, 0.0023, 0.0027, 0.0032, 0.0036, 0.0033, 0.0027, 0.0024, 0.0020, 0.0010, 0.0008, 0.0007, 0.0007, 0.0006, 0.0003, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0025, 0.0005])
a29 = np.array([-0.0034, -0.0033, -0.0034, -0.0033, -0.0029, -0.0025, -0.0025, -0.0031, -0.0036, -0.0039, -0.0048, -0.0050, -0.0041, -0.0032, -0.0020, -0.0017, -0.0020, -0.0020, -0.0020, -0.0020, -0.0020, -0.0020, -0.0034, -0.0037])
a31 = np.array([-0.1503, -0.1479, -0.1447, -0.1326, -0.1353, -0.1128, 0.0383, 0.0775, 0.0741, 0.2548, 0.2136, 0.1542, 0.0787, 0.0476, -0.0163, -0.1203, -0.2719, -0.2958, -0.2718, -0.2517, -0.1400, -0.0216, -0.1503, -0.1462])
a36 = np.array([0.2650, 0.2550, 0.2490, 0.2020, 0.1260, 0.0220, -0.1360, -0.0780, 0.0370, -0.0910, 0.1290, 0.3100, 0.5050, 0.3580, 0.1310, 0.1230, 0.1090, 0.1350, 0.1890, 0.2150, 0.1500, 0.0920, 0.2650, 0.3770])
a37 = np.array([0.3370, 0.3280, 0.3200, 0.2890, 0.2750, 0.2560, 0.1620, 0.2240, 0.2480, 0.2030, 0.2320, 0.2520, 0.2080, 0.2080, 0.1080, 0.0680, -0.0230, 0.0280, 0.0310, 0.0240, -0.0700, -0.1590, 0.3370, 0.2120])
a38 = np.array([0.1880, 0.1840, 0.1800, 0.1670, 0.1730, 0.1890, 0.1080, 0.1150, 0.1220, 0.0960, 0.1230, 0.1340, 0.1290, 0.1520, 0.1180, 0.1190, 0.0930, 0.0840, 0.0580, 0.0650, 0.0000, -0.0500, 0.1880, 0.1570])
a39 = np.array([0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000])
a40 = np.array([0.0880, 0.0880, 0.0930, 0.1330, 0.1860, 0.1600, 0.0680, 0.0480, 0.0550, 0.0730, 0.1430, 0.1600, 0.1580, 0.1450, 0.1310, 0.0830, 0.0700, 0.1010, 0.0950, 0.1330, 0.1510, 0.1240, 0.0880, 0.0950])
a41 = np.array([-0.1960, -0.1940, -0.1750, -0.0900, 0.0900, 0.0060, -0.1560, -0.2740, -0.2480, -0.2030, -0.1540, -0.1590, -0.1410, -0.1440, -0.1260, -0.0750, -0.0210, 0.0720, 0.2050, 0.2850, 0.3290, 0.3010, -0.1960, -0.0380])
a42 = np.array([0.0440, 0.0610, 0.1620, 0.4510, 0.5060, 0.3350, -0.0840, -0.1780, -0.1870, -0.1590, -0.0230, -0.0290, 0.0610, 0.0620, 0.0370, -0.1430, -0.0280, -0.0970, 0.0150, 0.1040, 0.2990, 0.2430, 0.0440, 0.0650])
s1 = np.array([0.7540, 0.7600, 0.7810, 0.8100, 0.8100, 0.8100, 0.8010, 0.7890, 0.7700, 0.7400, 0.6990, 0.6760, 0.6310, 0.6090, 0.5780, 0.5550, 0.5480, 0.5270, 0.5050, 0.4770, 0.4570, 0.4290, 0.7540, 0.6620])
s2 = np.array([0.5200, 0.5200, 0.5200, 0.5300, 0.5400, 0.5500, 0.5600, 0.5650, 0.5700, 0.5800, 0.5900, 0.6000, 0.6150, 0.6300, 0.6400, 0.6500, 0.6400, 0.6300, 0.6300, 0.6300, 0.6300, 0.6300, 0.5200, 0.5100])
s3 = np.array([0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.4700, 0.3800])
s4 = np.array([0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3600, 0.3800])
s1_m = np.array([0.7410, 0.7470, 0.7690, 0.7980, 0.7980, 0.7950, 0.7730, 0.7530, 0.7290, 0.6930, 0.6440, 0.6160, 0.5660, 0.5410, 0.5060, 0.4800, 0.4720, 0.4470, 0.4250, 0.3950, 0.3780, 0.3590, 0.7410, 0.6600])
s2_m = np.array([0.5010, 0.5010, 0.5010, 0.5120, 0.5220, 0.5270, 0.5190, 0.5140, 0.5130, 0.5190, 0.5240, 0.5320, 0.5480, 0.5650, 0.5760, 0.5870, 0.5760, 0.5650, 0.5680, 0.5710, 0.5750, 0.5850, 0.5010, 0.5100])
s5_JP = np.array([0.5400, 0.5400, 0.5500, 0.5600, 0.5700, 0.5700, 0.5800, 0.5900, 0.6100, 0.6300, 0.6600, 0.6900, 0.7300, 0.7700, 0.8000, 0.8000, 0.8000, 0.7600, 0.7200, 0.7000, 0.6700, 0.6400, 0.5400, 0.5800])
s6_JP = np.array([0.6300, 0.6300, 0.6300, 0.6500, 0.6900, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.7000, 0.6900, 0.6800, 0.6600, 0.6200, 0.5500, 0.5200, 0.5000, 0.5000, 0.5000, 0.5000, 0.6300, 0.5300])
# fmt: on


def ASK_2014_nga(
    siteprop, faultprop, im=None, period=None, region=0, f_hw=None, f_as=0
):
    """
    Matlab coded by Yue Hua, 5/19/10
                  Stanford University
                  yuehua@stanford.edu
    edited by Jack Baker, 9/22/2017 to fix an error in f_vs30 variable
    definition in the comments (no change to the function of the code).
    
    Abrahamson, N. A., Silva, W. J., and Kamai, R. (2014). ?Summary of the 
    ASK14 Ground Motion Relation for Active Crustal Regions.? Earthquake 
    Spectra, 30(3), 1025?1055.
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Input Variables
    siteprop.Rrup         = closest distance (km) to the ruptured plane
    siteprop.Rjb          = Joyner-Boore distance (km); closest distance (km) to surface
                            projection of rupture plane
    siteprop.Rx           = site coordinate (km) measured perpendicular to the fault strike
                            from the fault line with down-dip direction to be positive
    siteprop.Ry          = horizontal distance off the end of the rupture measured parallel
                            to strike
    siteprop.vs30         = shear wave velocity averaged over top 30 m in m/s
                            ref: 1130
    siteprop.vs30measured = True for measured vs30
                            False for vs30 inferred from geology or Japan
    siteprop.z1p0         = Basin depth (km); depth from the groundsurface to the
                            1km/s shear-wave horizon.

    faultprop.dip   = fault dip angle (in degrees)
    faultprop.Mw    = moment magnitude
    faultprop.rake  = Rake angle (in degrees)
    faultprop.width = down-dip rupture width (km)
    faultprop.ztor  = depth (km) to the top of ruptured plane

    period = period (sec); period = -1 for PGV computation
                    None for output the array of Sa with original period (no interpolation)

    region        = 0 for global
                  = 1 for California
                  = 2 for Japan
                  = 3 for China 
                  = 4 for Italy 
                  = 5 for Turkey
                  = 6 for Taiwan

    f_hw   = flag for hanging wall sites

    f_as   = flag for aftershocks

    crjb = centroid crjb, hardcoded to assume no aftershock

    Output Variables
    Sa: Median spectral acceleration prediction
    sigma: logarithmic standard deviation of spectral acceleration
             prediction
    """
    mag = faultprop.Mw
    if im == "PGV":
        period = -1
    if im == "PGA":
        period = 0
    if f_hw is None:
        f_hw = int(siteprop.Rx >= 0)
    T = period
    w = faultprop.width
    z10 = siteprop.z1p0
    ztor = faultprop.ztor

    f_rv = int(30 <= faultprop.rake <= 150)
    f_nm = int(-150 <= faultprop.rake <= -30)

    if ztor is None:
        if f_rv == 1:
            ztor = max(2.704 - 1.226 * max(mag - 5.849, 0), 0) ** 2
        else:
            ztor = max(2.673 - 1.136 * max(mag - 4.970, 0), 0) ** 2

    if w is None:
        w = min(18 / math.sin(math.radians(faultprop.dip)), 10 ** (-1.75 + 0.45 * mag))

    if z10 is None:
        if region == 2:
            # Japan
            z10 = (
                math.exp(
                    -5.23
                    / 2
                    * math.log((siteprop.vs30 ** 2 + 412 ** 2) / (1360 ** 2 + 412 ** 2))
                )
                / 1000
            )
        else:
            z10 = (
                math.exp(
                    (-7.67 / 4)
                    * math.log((siteprop.vs30 ** 4 + 610 ** 4) / (1360 ** 4 + 610 ** 4))
                )
                / 1000
            )

    def sa_sigma(period_i):
        """
        Return Sa and sigma for given period index.
        """
        return ASK_2014_sub_1(
            mag,
            period_i,
            siteprop.Rrup,
            siteprop.Rjb,
            siteprop.Rx,
            siteprop.Ry,
            ztor,
            faultprop.dip,
            f_rv,
            f_nm,
            f_as,
            f_hw,
            w,
            z10,
            siteprop.vs30,
            siteprop.vs30measured,
            region,
        )

    if T is None:
        # compute Sa and sigma with pre-defined periods
        periods_out = periods[:-2]
        Sa = np.zeros(len(periods_out))
        sigma = np.zeros(len(periods_out))
        for ip in range(len(periods_out)):
            Sa[ip], sigma[ip] = sa_sigma(ip)
        return Sa, sigma, periods_out

    # compute Sa and sigma with user-defined period
    try:
        T[0]
    except TypeError:
        T = [T]
    Sa = np.zeros(len(T))
    sigma = np.zeros([len(T), 3])
    for i, Ti in enumerate(T):
        if not np.isclose(periods, Ti, atol=0.0001).any():
            # user defined period requires interpolation
            ip_high = np.argmin(periods < Ti)
            ip_low = ip_high - 1

            y_low = sa_sigma(ip_low)
            y_high = sa_sigma(ip_high)
            Sa[i], sigma[i] = interpolate_to_closest(Ti, periods[ip_high], periods[ip_low], y_high, y_low)
        else:
            ip_T = np.argmin(np.abs(periods - Ti))
            Sa[i], sigma[i, :] = sa_sigma(ip_T)

    if len(T) == 1:
        return Sa[0], sigma[0]
    else:
        return list(zip(Sa, sigma))


#@numba.jit(nopython=True)
def ASK_2014_sub_1(
    mag,
    ip,
    rrup,
    rjb,
    rx,
    ry0,
    ztor,
    delta,
    f_rv,
    f_nm,
    f_as,
    f_hw,
    w,
    z10,
    vs30,
    f_vs30,
    region,
):
    m2 = 5
    crjb = None

    # f1 - basic form
    if mag > 5:
        c4m = c4[ip]
    elif 4 < mag <= 5:
        c4m = c4[ip] - (c4[ip] - 1) * (5 - mag)
    else:
        c4m = 1

    R = math.sqrt(rrup ** 2 + c4m ** 2)

    if mag > m1[ip]:
        f1 = (
            a1[ip]
            + a5[ip] * (mag - m1[ip])
            + a8[ip] * (8.5 - mag) ** 2
            + (a2[ip] + a3[ip] * (mag - m1[ip])) * math.log(R)
            + a17[ip] * rrup
        )
    elif m2 <= mag <= m1[ip]:
        f1 = (
            a1[ip]
            + a4[ip] * (mag - m1[ip])
            + a8[ip] * (8.5 - mag) ** 2
            + (a2[ip] + a3[ip] * (mag - m1[ip])) * math.log(R)
            + a17[ip] * rrup
        )
    else:
        f1 = (
            a1[ip]
            + a4[ip] * (m2 - m1[ip])
            + a8[ip] * (8.5 - m2) ** 2
            + a6[ip] * (mag - m2)
            + a7[ip] * (mag - m2) ** 2
            + (a2[ip] + a3[ip] * (m2 - m1[ip])) * math.log(R)
            + a17[ip] * rrup
        )

    # term f4 - hanging wall model
    def hanging_wall():
        if not f_hw:
            return 0

        r1 = w * math.cos(math.radians(delta))
        r2 = 3 * r1
        t1 = (90 - max(30, delta)) / 45
        a2hw = 0.2

        if mag > 6.5:
            t2 = 1 + a2hw * (mag - 6.5)
        elif mag > 5.5:
            t2 = 1 + a2hw * (mag - 6.5) - (1 - a2hw) * (mag - 6.5) ** 2
        else:
            t2 = 0

        if rx <= r1:
            # h1 = 0.25; h2 = 1.5; h3 = -0.75
            t3 = 0.25 + 1.5 * (rx / r1) - 0.75 * (rx / r1) ** 2
        elif rx < r2:
            t3 = 1 - (rx - r1) / (r2 - r1)
        else:
            t3 = 0

        t4 = 1 - ztor ** 2 / 100 if ztor < 10 else 0

        if ry0 == 999 or ry0 == 0:
            if rjb == 0:
                t5 = 1
            elif rjb < 30:
                t5 = 1 - rjb / 30
            else:
                t5 = 0
        else:
            Ry1 = rx * math.tan(math.radians(20))
            if ry0 - Ry1 <= 0:
                t5 = 1
            elif ry0 - Ry1 < 5:
                t5 = 1 - (ry0 - Ry1) / 5
            else:
                t5 = 0

        return a13[ip] * t1 * t2 * t3 * t4 * t5

    f4 = hanging_wall()

    # f6 - depth to top rupture model
    f6 = a15[ip]
    if ztor < 20:
        f6 *= ztor / 20

    # f7 and f8 - style of faulting
    if mag > 5:
        f7 = a11[ip]
        f8 = a12[ip]
    elif 4 <= mag <= 5:
        f7 = a11[ip] * (mag - 4)
        f8 = a12[ip] * (mag - 4)
    else:
        # mag < 4
        f7 = 0
        f8 = 0

    if periods[ip] <= 0.5:
        V1 = 1500
    elif periods[ip] < 3:
        V1 = math.exp(-0.35 * math.log(periods[ip] / 0.5) + math.log(1500))
    else:
        V1 = 800

    vs30s = min(V1, vs30)
    vs30star1180 = min(V1, 1180)

    # regional terms
    if region == 2:
        # Japan
        if vs30 < 150:
            y1 = a36[ip]
            y2 = a36[ip]
            x1 = 50
            x2 = 150
        elif vs30 < 250:
            y1 = a36[ip]
            y2 = a37[ip]
            x1 = 150
            x2 = 250
        elif vs30 < 350:
            y1 = a37[ip]
            y2 = a38[ip]
            x1 = 250
            x2 = 350
        elif vs30 < 450:
            y1 = a38[ip]
            y2 = a39[ip]
            x1 = 350
            x2 = 450
        elif vs30 < 600:
            y1 = a39[ip]
            y2 = a40[ip]
            x1 = 450
            x2 = 600
        elif vs30 < 850:
            y1 = a40[ip]
            y2 = a41[ip]
            x1 = 600
            x2 = 850
        elif vs30 < 1150:
            y1 = a41[ip]
            y2 = a42[ip]
            x1 = 850
            x2 = 1150
        else:
            y1 = a42[ip]
            y2 = a42[ip]
            x1 = 1150
            x2 = 3000
        f13vs30 = y1 + (y2 - y1) / (x2 - x1) * (vs30 - x1)
        regional = f13vs30 + a29[ip] * rrup
        regional_1180 = regional
    elif region == 6:
        # Taiwan
        f12vs30 = a31[ip] * math.log(vs30s / Vlin[ip])
        f12vs30_1180 = a31[ip] * math.log(vs30star1180 / Vlin[ip])
        regional = f12vs30 + a25[ip] * rrup
        regional_1180 = f12vs30_1180 + a25[ip] * rrup
    elif region == 3:
        # China
        regional = a28[ip] * rrup
        regional_1180 = regional
    else:
        # rest of the world
        regional = 0
        regional_1180 = 0

    # f5 - site response model

    # Sa 1180
    f5_1180 = (a10[ip] + b[ip] * n[ip]) * math.log(vs30star1180 / Vlin[ip])

    Sa1180 = math.exp(
        f1 + f6 + f_rv * f7 + f_nm * f8 + f_hw * f4 + f5_1180 + regional_1180
    )

    if vs30 >= Vlin[ip]:
        f5 = (a10[ip] + b[ip] * n[ip]) * math.log(vs30s / Vlin[ip])
    else:
        f5 = (
            a10[ip] * math.log(vs30s / Vlin[ip])
            - b[ip] * math.log(Sa1180 + c[ip])
            + b[ip] * math.log(Sa1180 + c[ip] * (vs30s / Vlin[ip]) ** n[ip])
        )

    # f10 - soil depth model

    if region == 2:
        # Japan
        Z1ref = (
            1
            / 1000
            * math.exp(
                -5.23 / 2 * math.log((vs30 ** 2 + 412 ** 2) / (1360 ** 2 + 412 ** 2))
            )
        )
    else:
        Z1ref = (
            1
            / 1000
            * math.exp(
                -7.67 / 4 * math.log((vs30 ** 4 + 610 ** 4) / (1360 ** 4 + 610 ** 4))
            )
        )

    if vs30 <= 150:
        y1z = a43[ip]
        y2z = a43[ip]
        x1z = 50
        x2z = 150
    elif vs30 <= 250:
        y1z = a43[ip]
        y2z = a44[ip]
        x1z = 150
        x2z = 250
    elif vs30 <= 400:
        y1z = a44[ip]
        y2z = a45[ip]
        x1z = 250
        x2z = 400
    elif vs30 <= 700:
        y1z = a45[ip]
        y2z = a46[ip]
        x1z = 400
        x2z = 700
    else:
        y1z = a46[ip]
        y2z = a46[ip]
        x1z = 700
        x2z = 1000

    # f10 term goes to zero at 1180 m/s (reference)
    f10 = (y1z + (vs30 - x1z) * (y2z - y1z) / (x2z - x1z)) * math.log(
        (z10 + 0.01) / (Z1ref + 0.01)
    )

    # f11 - aftershock scaling
    if crjb is None or crjb >= 15 or f_as == 0:
        f11 = 0
    elif crjb <= 5:
        f11 = a14[ip]
    elif crjb < 15:
        f11 = a14[ip] * (1 - (crjb - 5) / 10)

    # Sa
    lnSa = (
        f1 + f6 + f_rv * f7 + f_nm * f8 + f_hw * f4 + f_as * f11 + f5 + f10 + regional
    )
    Sa = np.exp(lnSa)
    sigma_SA = compute_stdev(ip, mag, rrup, Sa1180, vs30, f_vs30, region=region)

    return Sa, sigma_SA


#@numba.jit(nopython=True)
def compute_stdev(ip, mag, rrup, Sa1180, vs30, f_vs30, region=0):
    # Standard deviation

    if region == 2:
        # Japan
        if rrup < 30:
            phi_AL = s5_JP[ip]
        elif rrup <= 80:
            phi_AL = s5_JP[ip] + (s6_JP[ip] - s5_JP[ip]) / 50 * (rrup - 30)
        else:
            phi_AL = s6_JP[ip]
    else:
        if f_vs30:
            # measured
            s1l = s1_m
            s2l = s2_m
        else:
            s1l = s1
            s2l = s2
        if mag < 4:
            phi_AL = s1l[ip]
        elif mag <= 6:
            phi_AL = s1l[ip] + (s2l[ip] - s1l[ip]) / 2 * (mag - 4)
        else:
            phi_AL = s2l[ip]
    phi_amp = 0.4
    phi_B = math.sqrt(phi_AL ** 2 - phi_amp ** 2)

    if mag < 5:
        tau_B = s3[ip]
    elif mag <= 7:
        tau_B = s3[ip] + (s4[ip] - s3[ip]) / 2 * (mag - 5)
    else:
        tau_B = s4[ip]

    if vs30 >= Vlin[ip]:
        dln = 0
    else:
        dln = -b[ip] * Sa1180 / (Sa1180 + c[ip]) + b[ip] * Sa1180 / (
            Sa1180 + c[ip] * (vs30 / Vlin[ip]) ** n[ip]
        )

    phi = math.sqrt(phi_B ** 2 * (1 + dln) ** 2 + phi_amp ** 2)
    tau = tau_B * (1 + dln)

    return math.sqrt(phi ** 2 + tau ** 2), tau, phi
