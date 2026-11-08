"""Constants shared across the ground motion models."""

from enum import StrEnum


class TectType(StrEnum):
    """Tectonic type."""

    ACTIVE_SHALLOW = "ACTIVE_SHALLOW"
    VOLCANIC = "VOLCANIC"
    SUBDUCTION_INTERFACE = "SUBDUCTION_INTERFACE"
    SUBDUCTION_SLAB = "SUBDUCTION_SLAB"


class EpistemicBranch(StrEnum):
    """Epistemic uncertainty branch, mirroring `oq_wrapper.constants.EpistemicBranch`."""

    LOWER = "LOWER"
    CENTRAL = "CENTRAL"
    UPPER = "UPPER"


# `sigma_mu_epsilon` per branch, shared by every model that has it. Matches
# oq_wrapper.constants.GMM_EPISTEMIC_BRANCH_KWARGS_MAPPING, where CENTRAL is
# omitted and so takes OpenQuake's default of 0.
SIGMA_MU_EPSILON = {
    EpistemicBranch.LOWER: -1.2815,
    EpistemicBranch.CENTRAL: 0.0,
    EpistemicBranch.UPPER: 1.2815,
}
