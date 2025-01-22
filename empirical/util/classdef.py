from qcore.constants import ExtendedEnum, ExtendedStrEnum

VS30_DEFAULT = 250


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
    S_22 = 112, "S22"
    A_22 = 113, "A22"
    Br_13 = 114, "B13"
    P_21 = 115, "P21"
    GA_11 = 116, "GA11"
