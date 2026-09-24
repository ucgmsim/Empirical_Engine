"""Constants used in the OpenQuake wrapper package."""

from enum import StrEnum, auto
from pathlib import Path

from openquake.hazardlib import const as oq_const

# Different OQ GMM model standard deviation types of interest

SPT_STD_DEVS = [
    oq_const.StdDev.TOTAL,
    oq_const.StdDev.INTER_EVENT,
    oq_const.StdDev.INTRA_EVENT,
]


class TectType(StrEnum):
    """Fault tectonic type."""

    ACTIVE_SHALLOW = auto()
    VOLCANIC = auto()
    SUBDUCTION_INTERFACE = auto()
    SUBDUCTION_SLAB = auto()


OQ_TECT_TYPE_MAPPING = {
    oq_const.TRT.ACTIVE_SHALLOW_CRUST: TectType.ACTIVE_SHALLOW,
    oq_const.TRT.SUBDUCTION_INTERFACE: TectType.SUBDUCTION_INTERFACE,
    oq_const.TRT.SUBDUCTION_INTRASLAB: TectType.SUBDUCTION_SLAB,
    oq_const.TRT.VOLCANIC: TectType.VOLCANIC,
}


class GMM(StrEnum):
    """Ground motion models."""

    ZA_06 = "ZA_06"
    AS_16 = "AS_16"
    BSSA_14 = "BSSA_14"
    ASK_14 = "ASK_14"
    BCH_16 = "BCH_16"
    CB_14 = "CB_14"
    CY_14 = "CY_14"
    CB_10 = "CB_10"
    A_18 = "A_18"
    P_20 = "P_20"
    AG_20 = "AG_20"  # (global)
    K_20 = "K_20"  # (global)
    K_20_NZ = "K_20_NZ"  # (NZ)
    AG_20_NZ = "AG_20_NZ"  # (NZ)
    S_22 = "S_22"
    A_22 = "A_22"
    Br_13 = "Br_13"
    P_21 = "P_21"
    GA_11 = "GA_11"
    BA_18 = "BA_18"


class GMMLogicTree(StrEnum):
    """Logic tree for GMMs."""

    NHM2010_BB = "NHM2010_BB"
    NSHM2022 = "NSHM2022"


class EpistemicBranch(StrEnum):
    """Epistemic uncertainty for GMMs"""

    LOWER = "LOWER"
    CENTRAL = "CENTRAL"
    UPPER = "UPPER"


GMM_EPISTEMIC_BRANCH_KWARGS_MAPPING = {
    GMM.S_22: {
        EpistemicBranch.LOWER: {"mu_branch": "Lower", "sigma_branch": "Lower"},
        EpistemicBranch.UPPER: {"mu_branch": "Upper", "sigma_branch": "Upper"},
    },
    GMM.A_22: {
        EpistemicBranch.LOWER: {"epistemic": "lower"},
        EpistemicBranch.UPPER: {"epistemic": "upper"},
    },
    GMM.AG_20: {
        EpistemicBranch.LOWER: {"sigma_mu_epsilon": -1.2815},
        EpistemicBranch.UPPER: {"sigma_mu_epsilon": 1.2815},
    },
    GMM.K_20: {
        EpistemicBranch.LOWER: {"sigma_mu_epsilon": -1.2815},
        EpistemicBranch.UPPER: {"sigma_mu_epsilon": 1.2815},
    },
    GMM.P_21: {
        EpistemicBranch.LOWER: {"sigma_mu_epsilon": -1.2815},
        EpistemicBranch.UPPER: {"sigma_mu_epsilon": 1.2815},
    },
    # ASK_14/CY_14/CB_14/BSSA_14 don't have a native epistemic-in-the-mean model,
    # so they're routed through OQ's NSHMP2014 wrapper (Rezaeian et al., 2014, as
    # used in the 2014 US NSHM), which shifts the median by a magnitude/distance-
    # binned amount (see OQ_MODEL_MAPPING).
    #
    # Rezaeian et al. (2014)'s tabulated value is the 5th/95th-percentile offset
    # (i.e. z=1.645, the extended Pearson-Tukey convention implied by its
    # 0.185/0.63/0.185 branch weights -- Keefer and Bodily, 1983), i.e.
    # sigma_ept = 1.645 * Std[mu]. NZ NSHM 2022 (Bradley, 2024, Eq. 1) instead
    # uses the 10th/90th percentile (z=1.2815) uniformly across the logic tree,
    # matching the z=1.2815 already used natively below for S_22/A_22/AG_20/
    # K_20/P_21/Br_13 and the 0.3/0.4/0.3 weights baked into
    # nshm_2022_config.yaml. NSHMP2014's `sgn` is a free scalar multiplier (not
    # restricted to {-1, 0, 1}), so it's rescaled here to reproduce z=1.2815
    # from Rezaeian's z=1.645 table: 1.2815 / 1.645.
    GMM.ASK_14: {
        EpistemicBranch.LOWER: {"sgn": -1.2815 / 1.645},
        EpistemicBranch.UPPER: {"sgn": 1.2815 / 1.645},
    },
    GMM.CY_14: {
        EpistemicBranch.LOWER: {"sgn": -1.2815 / 1.645},
        EpistemicBranch.UPPER: {"sgn": 1.2815 / 1.645},
    },
    GMM.CB_14: {
        EpistemicBranch.LOWER: {"sgn": -1.2815 / 1.645},
        EpistemicBranch.UPPER: {"sgn": 1.2815 / 1.645},
    },
    GMM.BSSA_14: {
        EpistemicBranch.LOWER: {"sgn": -1.2815 / 1.645},
        EpistemicBranch.UPPER: {"sgn": 1.2815 / 1.645},
    },
    # Br_13 (NZ-specific, CY-lineage refit) has its own native epistemic-in-the-
    # mean model in OQ (Bradley2013's `sigma_mu_epsilon` kwarg, computed from NZ
    # SMDB v1.0 earthquake counts via the same USGS-2014 square-root rule as
    # Rezaeian et al. 2014) -- same convention as AG_20/K_20/P_21 above, used
    # directly with z=1.2815 (no rescaling needed, unlike ASK_14/CY_14/CB_14/
    # BSSA_14's borrowed US table).
    GMM.Br_13: {
        EpistemicBranch.LOWER: {"sigma_mu_epsilon": -1.2815},
        EpistemicBranch.UPPER: {"sigma_mu_epsilon": 1.2815},
    },
}


GMM_LT_CONFIG_DIR = Path(__file__).parent / "gmm_logic_tree_configs"

GMM_LT_CONFIG_MAPPING = {
    GMMLogicTree.NHM2010_BB: GMM_LT_CONFIG_DIR / "nhm_2010_bb_config.yaml",
    GMMLogicTree.NSHM2022: GMM_LT_CONFIG_DIR / "nshm_2022_config.yaml",
}


NZGMDB_SITE_COLUMNS = [
    "Vs30",
    "Z1.0",
    "Z2.5",
]
OQ_SITE_COLUMNS = [
    "vs30",
    "z1pt0",
    "z2pt5",
]

NZGMDB_SITE_TO_SOURCE_COLUMNS = [
    "r_rup",
    "r_jb",
    "r_x",
    "r_y",
    "r_hyp",
]
OQ_SITE_TO_SOURCE_COLUMNS = [
    "rrup",
    "rjb",
    "rx",
    "ry",
    "rhypo",
]

NZ_GMDB_SOURCE_COLUMNS = [
    "mag",
    "tect_class",
    "z_tor",
    "z_bor",
    "rake",
    "dip",
    "ev_depth",
]
OQ_SOURCE_COLUMNS = [
    "mag",
    "tect_class",
    "ztor",
    "zbot",
    "rake",
    "dip",
    "hypo_depth",
]


NZGMDB_COLUMNS = (
    NZGMDB_SITE_COLUMNS + NZGMDB_SITE_TO_SOURCE_COLUMNS + NZ_GMDB_SOURCE_COLUMNS
)
OQ_RUPTURE_COLUMNS = OQ_SITE_COLUMNS + OQ_SITE_TO_SOURCE_COLUMNS + OQ_SOURCE_COLUMNS

NZGMDB_OQ_COL_MAPPING = dict(zip(NZGMDB_COLUMNS, OQ_RUPTURE_COLUMNS))
