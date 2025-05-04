from pathlib import Path

from openquake.hazardlib import const as oq_const

from qcore.constants import ExtendedEnum, ExtendedStrEnum

# Different OQ GMM model standard deviation types of interest
SPT_STD_DEVS = [
    oq_const.StdDev.TOTAL,
    oq_const.StdDev.INTER_EVENT,
    oq_const.StdDev.INTRA_EVENT,
]

class TectType(ExtendedEnum):
    """Fault tectonic type."""
    ACTIVE_SHALLOW = 1
    VOLCANIC = 2
    SUBDUCTION_INTERFACE = 3
    SUBDUCTION_SLAB = 4

OQ_TECT_TYPE_MAPPING = {
    oq_const.TRT.ACTIVE_SHALLOW_CRUST: TectType.ACTIVE_SHALLOW,
    oq_const.TRT.SUBDUCTION_INTERFACE: TectType.SUBDUCTION_INTERFACE,
    oq_const.TRT.SUBDUCTION_INTRASLAB: TectType.SUBDUCTION_SLAB,
    oq_const.TRT.VOLCANIC: TectType.VOLCANIC,
}

class GMM(ExtendedStrEnum):
    """Ground motion models."""
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

class GMMLogicTree(ExtendedStrEnum):
    """Logic tree for GMMs."""
    NHM2010_BB = 1, "NHM2010_BB"
    NSHM2022 = 2, "NSHM2022"

GMM_LT_CONFIG_DIR = Path(__file__).parent / "gmm_lt_configs"

GMM_LT_CONFIG_MAPPING = {
    GMMLogicTree.NHM2010_BB: GMM_LT_CONFIG_DIR / "nhm_2010_bb_gmm_lt_config.yaml",
    GMMLogicTree.NSHM2022: GMM_LT_CONFIG_DIR / "nshm_2022_gmm_lt_config.yaml",
}


NZ_GMDB_SOURCE_COLUMNS = [
    "mag",
    "tect_class",
    "z_tor",
    "z_bor",
    "rake",
    "dip",
    "depth",
    "ev_depth",
    "r_rup",
    "r_jb",
    "r_x",
    "r_y",
    "r_hyp",
    "Vs30",
    "Z1.0",
    "Z2.5",
] 

OQ_RUPTURE_COLUMNS = [
    "mag",
    "tect_class",
    "ztor",
    "zbot",
    "rake",
    "dip",
    "hypo_depth",
    "hypo_depth",
    "rrup",
    "rjb",
    "rx",
    "ry",
    "rhypo",
    "vs30",
    "z1pt0",
    "z2pt5",
]

NZGMDB_OQ_COL_MAPPING = dict(zip(NZ_GMDB_SOURCE_COLUMNS, OQ_RUPTURE_COLUMNS))