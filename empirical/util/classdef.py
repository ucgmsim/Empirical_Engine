from enum import Enum, IntEnum

import numba
import numpy as np


VS30_DEFAULT = 250


class GMM(IntEnum):
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


class TectType(IntEnum):
    ACTIVE_SHALLOW = 1
    VOLCANIC = 2
    SUBDUCTION_INTERFACE = 3
    SUBDUCTION_SLAB = 4


class SiteClass(Enum):
    # as per NZS1170.5
    HARDROCK = 0
    ROCK = 1
    HARDSOIL = 2
    MEDIUMSOIL = 3
    SOFTSOIL = 4


class FaultStyle(IntEnum):
    REVERSE = 1
    NORMAL = 2
    STRIKESLIP = 3
    OBLIQUE = 4
    UNKNOWN = 5
    SLAB = 6
    INTERFACE = 7


class Orientation(IntEnum):
    AVERAGE = 1
    RANDOM = 2


site_spec = [
    ("Rrup", numba.float32),
    ("Rjb", numba.optional(numba.float32)),
    ("Rx", numba.optional(numba.float32)),
    ("Ry", numba.optional(numba.float32)),
    ("Rtvz", numba.optional(numba.float32)),
    ("vs30measured", numba.boolean),
    ("vs30", numba.optional(numba.float32)),
    ("z1p0", numba.optional(numba.float32)),
    ("z1p5", numba.optional(numba.float32)),
    ("z2p5", numba.optional(numba.float32)),
    ("siteclass", numba.optional(numba.uint8)),
    ("orientation", numba.int8),
    ("backarc", numba.boolean),
]


@numba.jitclass(site_spec)
class Site:  # Class of site properties. initialize all attributes to None
    def __init__(
        self,
        rrup=0,
        rjb=None,
        rx=None,
        ry=None,
        rtvz=None,
        vs30measured=False,
        vs30=None,
        z1p0=None,
        z1p5=None,
        z2p5=None,
        siteclass=None,
        orientation=Orientation.AVERAGE,
        backarc=False,
    ):
        self.Rrup = rrup  # closest distance coseismic rupture (km)
        self.Rjb = rjb  # closest horizontal distance coseismic rupture (km)
        self.Rx = (
            rx
        )  # distance measured perpendicular to fault strike from surface projection of
        #                       # updip edge of the fault rupture (+ve in downdip dir) (km)
        self.Ry = (
            ry
        )  # horizontal distance off the end of the rupture measured parallel
        self.Rtvz = (
            rtvz
        )  # source-to-site distance in the Taupo volcanic zone (TVZ) (km)
        self.vs30measured = (
            vs30measured
        )  # yes =True (i.e. from Vs tests); no=False (i.e. estimated from geology)
        self.vs30 = vs30  # shear wave velocity at 30m depth (m/s)
        self.z1p0 = (
            z1p0
        )  # depth (km) to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise)
        self.z1p5 = z1p5  # (km)
        self.z2p5 = z2p5  # (km)
        self.siteclass = siteclass
        self.orientation = orientation
        self.backarc = backarc  # forearc/unknown = False, backarc = True


fault_spec = [
    ("dip", numba.optional(numba.float32)),
    ("faultstyle", numba.optional(numba.uint8)),
    ("hdepth", numba.optional(numba.float32)),
    ("Mw", numba.optional(numba.float32)),
    ("rake", numba.optional(numba.float32)),
    ("tect_type", numba.optional(numba.int8)),
    ("width", numba.optional(numba.float32)),
    ("zbot", numba.optional(numba.float32)),
    ("ztor", numba.float32),
]


@numba.jitclass(fault_spec)
class Fault:  # Class of fault properties. initialize all attributes to None
    def __init__(
        self,
        dip=None,
        faultstyle=None,
        hdepth=None,
        Mw=None,
        rake=None,
        tect_type=None,
        width=None,
        zbot=None,
        ztor=None,
    ):
        self.dip = dip  # dip angle (degrees)
        self.faultstyle = faultstyle  # Faultstyle (options described in enum above)
        self.hdepth = hdepth  # hypocentre depth
        self.Mw = Mw  # moment tensor magnitude
        self.rake = rake  # rake angle (degrees)
        self.tect_type = tect_type  # tectonic type of the rupture (options described in the enum below)
        self.width = width  # down-dip width of the fault rupture plane
        self.zbot = zbot  # depth to the bottom of the seismogenic crust
        self.ztor = ztor  # depth to top of coseismic rupture (km)


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
        print("no z2p5 able to be estimated")
        exit()


def estimate_z1p0(vs30):
    return (
        np.exp(28.5 - 3.82 / 8.0 * np.log(vs30 ** 8 + 378.7 ** 8)) / 1000.0
    )  # CY08 estimate in KM
