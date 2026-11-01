"""Constants shared across the ground motion models."""

from enum import StrEnum


class TectType(StrEnum):
    """Tectonic type."""

    SUBDUCTION_INTERFACE = "SUBDUCTION_INTERFACE"
    SUBDUCTION_SLAB = "SUBDUCTION_SLAB"


class EpistemicBranch(StrEnum):
    """Epistemic uncertainty branch, mirroring `oq_wrapper.constants.EpistemicBranch`."""

    LOWER = "LOWER"
    CENTRAL = "CENTRAL"
    UPPER = "UPPER"


# Matches oq_wrapper.constants.GMM_EPISTEMIC_BRANCH_KWARGS_MAPPING[GMM.P_21].
P21_SIGMA_MU_EPSILON = {
    EpistemicBranch.LOWER: -1.2815,
    EpistemicBranch.CENTRAL: 0.0,
    EpistemicBranch.UPPER: 1.2815,
}
