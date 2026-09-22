"""Type stubs for the compiled `gm_models._utils` Rust extension."""

import numpy as np
import numpy.typing as npt

def _p21_interface(
    mag: npt.NDArray[np.float64],
    rrup: npt.NDArray[np.float64],
    vs30: npt.NDArray[np.float64],
    im: str,
    period: float,
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
    im: str,
    period: float,
    sigma_mu_epsilon: float,
    modified_sigma: bool,
) -> tuple[
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
    npt.NDArray[np.float64],
]: ...
