from enum import Enum
import numpy as np


class Site:  # Class of site properties. initialize all attributes to None
    def __init__(self):
        self.period = None  # '(-1),(0),(real variable)' period of vibration =-1->PGV; =0->PGA; >0->SA
        self.Rrup = None  # closest distance coseismic rupture (km)
        self.Rjb = None  # closest horizontal distance coseismic rupture (km)
        self.Rx = None  # distance measured perpendicular to fault strike from surface projection of
                        #  updip edge of the fault rupture (+ve in downdip dir) (km)
        self.Rtvz = None  # source-to-site distance in the Taupo volcanic zone (TVZ) (km)
        self.vs30measured = None  # yes =True (i.e. from Vs tests); no=False (i.e. estimated from geology)
        self.vs30 = None  # shear wave velocity at 30m depth (m/s)
        self.z1p0 = None  # depth to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise
        self.z2p5 = None
        self.orientation = 'average'


class Fault:  # Class of fault properties. initialize all attributes to None
    def __init__(self):
        self.Mw = None  # moment tensor magnitude
        self.rake = None  # rake angle (degrees)
        self.dip = None  # dip angle (degrees)
        self.ztor = None  # depth to top of coseismic rupture (km)
        self.rupture_type = None  # Valid values are: N, R, SS and None which correlate to Normal, Reverse, Strike-Slip and Unknown
        self.tect_type = None
        self.faultstyle = None


class TectType(Enum):
    ACTIVE_SHALLOW = 1
    VOLCANIC = 2
    SUBDUCTION_INTERFACE = 3
    SUBDUCTION_SLAB = 4


class GMM(Enum):
    ZA_06 = 1
    Br_13 = 2
    AS_16 = 3
    CB_12 = 4


class SiteClass(Enum):
    HARDROCK = 1
    ROCK = 2
    HARDSOIL = 3
    MEDIUMSOIL = 4
    SOFTSOIL = 5


class FaultStyle(Enum):
    REVERSE = 1
    NORMAL = 2
    STRIKESLIP = 3
    OBLIQUE = 4
    UNKNOWN = 5


def interpolate_to_closest(T, T_hi, T_low, y_high, y_low):
    [SA_low, sigma_SA_low] = y_low
    [SA_high, sigma_SA_high] = y_high
    SA_sigma = np.array([sigma_SA_low, sigma_SA_high])
    if T_low > 0:  # log interpolation
        x = [np.log(T_low), np.log(T_hi)]
        Y_sa = [np.log(SA_low), np.log(SA_high)]
        SA = np.exp(np.interp(T, x, Y_sa))
        sigma_SA = np.interp(np.log(T), x, SA_sigma)
    else:  # linear interpolation
        x = [T_low, T_hi]
        Y_sa = [SA_low, SA_high]
        SA = np.interp(T, x, Y_sa)
        sigma_total = np.interp(T, x, SA_sigma[:, 0])
        sigma_inter = np.interp(T, x, SA_sigma[:, 1])
        sigma_intra = np.interp(T, x, SA_sigma[:, 2])
        sigma_SA = [sigma_total, sigma_inter, sigma_intra]
    return SA, sigma_SA