from enum import Enum
import numpy as np

from qcore.constants import ExtendedEnum, ExtendedStrEnum


VS30_DEFAULT = 250


class Site:  # Class of site properties. initialize all attributes to None
    def __init__(self, **kwargs):
        self.name = kwargs.get("name")  # station name
        self.Rrup = kwargs.get("rrup")  # closest distance coseismic rupture (km)
        self.Rjb = kwargs.get(
            "rjb"
        )  # closest horizontal distance coseismic rupture (km)
        self.Rx = kwargs.get(
            "rx", -1.0
        )  # distance measured perpendicular to fault strike from surface projection of
        #                       # updip edge of the fault rupture (+ve in downdip dir) (km)
        self.Ry = kwargs.get(
            "ry", -1.0
        )  # horizontal distance off the end of the rupture measured parallel
        self.Rtvz = kwargs.get(
            "rtvz"
        )  # source-to-site distance in the Taupo volcanic zone (TVZ) (km)
        self.vs30measured = kwargs.get(
            "vs30measured", False
        )  # yes =True (i.e. from Vs tests); no=False (i.e. estimated from geology)
        self.vs30 = kwargs.get("vs30")  # shear wave velocity at 30m depth (m/s)
        self.z1p0 = kwargs.get(
            "z1p0"
        )  # depth (km) to the 1.0km/s shear wave velocity horizon (optional, uses default relationship otherwise)
        self.z1p5 = kwargs.get("z1p5")  # (km)
        self.z2p5 = kwargs.get("z2p5")  # (km)
        self.siteclass = kwargs.get("siteclass")
        self.orientation = kwargs.get("orientation", "average")
        self.backarc = kwargs.get(
            "backarc", False
        )  # forearc/unknown = False, backarc = True
        self.fpeak = kwargs.get("fpeak", 0)

    def __str__(self):
        return f"rrup: {self.Rrup}, rjb: {self.Rjb}"

    def __repr__(self):
        return self.__str__()


class Fault:  # Class of fault properties. initialize all attributes to None
    def __init__(self, **kwargs):
        self.dip = kwargs.get("dip")  # dip angle (degrees)
        self.faultstyle = kwargs.get(
            "faultstyle"
        )  # Faultstyle (options described in enum below)
        self.hdepth = kwargs.get("hdepth")  # hypocentre depth
        self.Mw = kwargs.get("Mw")  # moment tensor magnitude
        self.rake = kwargs.get("rake")  # rake angle (degrees)
        self.tect_type = kwargs.get(
            "tect_type"
        )  # tectonic type of the rupture (options described in the enum below)
        self.width = kwargs.get("width")  # down-dip width of the fault rupture plane
        self.zbot = kwargs.get("zbot")  # depth to the bottom of the seismogenic crust
        self.ztor = kwargs.get("ztor")  # depth to top of coseismic rupture (km)

    def __str__(self):
        return f"dip: {self.dip}, faultstyle: {self.faultstyle}, hdepth: {self.hdepth}, mw: {self.Mw}, rake: {self.rake},  tect_type: {self.tect_type}, width: {self.width}, zbot: {self.zbot}, ztor: {self.ztor}"

    def __repr__(self):
        return self.__str__()


class TectType(ExtendedEnum):
    ACTIVE_SHALLOW = 1
    VOLCANIC = 2
    SUBDUCTION_INTERFACE = 3
    SUBDUCTION_SLAB = 4


class GMM(ExtendedStrEnum):
    META = 0, "Meta model"
    ZA_06 = 1, "Z06"
    Br_10 = 2, "B10"
    AS_16 = 3, "AS16"
    CB_12 = 4, "CB12"
    BSSA_14 = 5, "BSSA14"
    MV_06 = 6, "M06"
    ASK_14 = 7, "ASK14"
    BCH_16 = 8, "A16"
    CB_14 = 9, "CB14"
    CY_14 = 10, "CY14"
    CB_10 = 11, "CB10"
    A_18 = 12, "A18"
    SB_13 = 13, "SB13"
    BB_13 = 14, "BB13"
    # openquake models below
    P_20 = 101, "P20"
    HA_20 = 102, "HA20"
    G_17 = 103, "G17"
    BC_16 = 104, "BC16"
    S_16 = 105, "S16"
    Ph_20 = 106, "Ph20"
    Ch_20 = 1071, "C20"
    AG_20 = 108, "AG20 (global)"
    K_20 = 109, "K20 (global)"
    K_20_NZ = 209, "K20 (NZ)"
    AG_20_NZ = 208, "AG20 (NZ)"
    Si_20 = 110, "S20"
    Z_16 = 111, "Z16"


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


class Orientation(Enum):
    AVERAGE = 1
    RANDOM = 2


class StdDevType(Enum):
    TOTAL = 1
    INTER_EVENT = 2
    INTRA_EVENT = 3


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
        np.exp(28.5 - 3.82 / 8.0 * np.log(vs30**8 + 378.7**8)) / 1000.0
    )  # CY08 estimate in KM
