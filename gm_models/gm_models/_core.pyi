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

def _p21_interface(
    mag: npt.NDArray[np.float64],
    rrup: npt.NDArray[np.float64],
    vs30: npt.NDArray[np.float64],
    im: IM,
    sigma_mu_epsilon: float,
    modified_sigma: bool,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...
def _p21_slab(
    mag: npt.NDArray[np.float64],
    rrup: npt.NDArray[np.float64],
    vs30: npt.NDArray[np.float64],
    hypo_depth: npt.NDArray[np.float64],
    backarc: npt.NDArray[np.bool_],
    im: IM,
    sigma_mu_epsilon: float,
    modified_sigma: bool,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...
