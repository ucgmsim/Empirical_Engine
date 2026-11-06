"""Type stubs for the compiled `gm_models._core` Rust extension."""

import numpy as np
import numpy.typing as npt

class IM:
    """Intensity measure."""

    class PGA(IM):
        """Peak ground acceleration."""

        def __init__(self) -> None: ...

    class PGV(IM):
        """Peak ground velocity."""

        def __init__(self) -> None: ...

    class SA(IM):
        """Pseudo-spectral acceleration at `period` seconds."""

        period: float
        def __init__(self, period: float) -> None: ...

class P21Inputs:
    """P21 input columns, one value per rupture-site pair."""

    @staticmethod
    def interface(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
    ) -> P21Inputs:
        """Subduction interface inputs."""

    @staticmethod
    def slab(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        hypo_depth: npt.NDArray[np.float64],
        backarc: npt.NDArray[np.bool_],
    ) -> P21Inputs:
        """Subduction slab inputs."""

class GMMResult:
    """Mean (ln space) and standard deviations, one array per quantity."""

    @property
    def ln_mean(self) -> npt.NDArray[np.float64]: ...
    @property
    def std_total(self) -> npt.NDArray[np.float64]: ...
    @property
    def std_inter(self) -> npt.NDArray[np.float64]: ...
    @property
    def std_intra(self) -> npt.NDArray[np.float64]: ...

def p21(
    inputs: P21Inputs,
    im: IM,
    sigma_mu_epsilon: float,
    modified_sigma: bool,
) -> GMMResult: ...
