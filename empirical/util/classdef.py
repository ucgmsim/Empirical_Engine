from enum import Enum
import numpy as np


VS30_DEFAULT = 250


class Site:  # Class of site properties. initialize all attributes to None
    def __init__(self):
        self.name = None
        self.Rrup = None  # closest distance coseismic rupture (km)
        self.Rjb = None  # closest horizontal distance coseismic rupture (km)
        self.Rx = None  # distance measured perpendicular to fault strike from surface projection of
#                       # updip edge of the fault rupture (+ve in downdip dir) (km)
        self.Rtvz = None  # source-to-site distance in the Taupo volcanic zone (TVZ) (km)
        self.vs30measured = None  # yes =True (i.e. from Vs tests); no=False (i.e. estimated from geology)
        self.vs30 = None  # shear wave velocity at 30m depth (m/s)
        self.z1p0 = None  # depth (km) to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise)
        self.z1p5 = None  # (km)
        self.z2p5 = None  # (km)
        self.siteclass = None
        self.orientation = 'average'
        self.rvol = 0 # length in km of the part of the source to site distance in volcanic zone
        self.backarc = False # forearc/unknown = False, backarc = True



class Fault:  # Class of fault properties. initialize all attributes to None
    def __init__(self):
        self.Mw = None  # moment tensor magnitude
        self.rake = None  # rake angle (degrees)
        self.dip = None  # dip angle (degrees)
        self.ztor = None  # depth to top of coseismic rupture (km)
        self.rupture_type = None  # Valid values are: N, R, SS and None which correlate to Normal, Reverse, Strike-Slip and Unknown
        self.tect_type = None
        self.faultstyle = None
        self.hdepth = None


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
    BSSA_14 = 5
    MV_06 = 6
    ASK_14 = 7
    BC_16 = 8
    CB_14 = 9
    CY_14 = 10


class SiteClass(Enum):
    # as per NZS1170.5
    HARDROCK = "A"
    ROCK = "B"
    HARDSOIL = "C"
    MEDIUMSOIL = "D"
    SOFTSOIL = "E"


class FaultStyle(Enum):
    REVERSE = 1
    NORMAL = 2
    STRIKESLIP = 3
    OBLIQUE = 4
    UNKNOWN = 5
    SLAB = 6
    INTERFACE = 7


def interpolate_to_closest(T, T_hi, T_low, y_high, y_low):
    [SA_low, sigma_SA_low] = y_low
    [SA_high, sigma_SA_high] = y_high
    SA_sigma = np.array([sigma_SA_low, sigma_SA_high])
    if T_low > 0:  # log interpolation
        x = [np.log(T_low), np.log(T_hi)]
        Y_sa = [np.log(SA_low), np.log(SA_high)]
        SA = np.exp(np.interp(np.log(T), x, Y_sa))
        sigma_total = np.interp(np.log(T), x, SA_sigma[:, 0])
        sigma_inter = np.interp(np.log(T), x, SA_sigma[:, 1])
        sigma_intra = np.interp(np.log(T), x, SA_sigma[:, 2])
        sigma_SA = [sigma_total, sigma_inter, sigma_intra]
    else:  # linear interpolation
        x = [T_low, T_hi]
        Y_sa = [SA_low, SA_high]
        SA = np.interp(T, x, Y_sa)
        sigma_total = np.interp(T, x, SA_sigma[:, 0])
        sigma_inter = np.interp(T, x, SA_sigma[:, 1])
        sigma_intra = np.interp(T, x, SA_sigma[:, 2])
        sigma_SA = [sigma_total, sigma_inter, sigma_intra]
    return SA, sigma_SA


# CB08 estimate of Z2p5
def estimate_z2p5(z1p0=None, z1p5=None):
    if z1p5 is not None:
        return 0.636 + 1.549 * z1p5
    elif z1p0 is not None:
        return 0.519 + 3.595 * z1p0
    else:
        print('no z2p5 able to be estimated')
        exit()


def estimate_z1p0(vs30):
    return np.exp(28.5 - 3.82 / 8.0 * np.log(vs30 ** 8 + 378.7 ** 8)) / 1000.0  # CY08 estimate in KM
