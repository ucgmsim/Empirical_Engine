from qcore.constants import ExtendedEnum, ExtendedStrEnum

VS30_DEFAULT = 250


class TectType(ExtendedEnum):
    """Fault tectonic type."""
    ACTIVE_SHALLOW = 1
    VOLCANIC = 2
    SUBDUCTION_INTERFACE = 3
    SUBDUCTION_SLAB = 4


class GMM(ExtendedStrEnum):
    """Ground motion models."""
    META = 0, "Meta model"
    ZA_06 = 1, "Z06"
    Br_10 = 2, "B10"
    AS_16 = 3, "AS16"
    BSSA_14 = 5, "BSSA14"
    ASK_14 = 7, "ASK14"
    BCH_16 = 8, "A16"
    CB_14 = 9, "CB14"
    CY_14 = 10, "CY14"
    CB_10 = 11, "CB10"
    A_18 = 12, "A18"
    P_20 = 101, "P20"
    AG_20 = 108, "AG20 (global)"
    K_20 = 109, "K20 (global)"
    K_20_NZ = 209, "K20 (NZ)"
    AG_20_NZ = 208, "AG20 (NZ)"
    S_22 = 112, "S22"
    A_22 = 113, "A22"
    Br_13 = 114, "B13"
    P_21 = 115, "P21"
    GA_11 = 116, "GA11"
