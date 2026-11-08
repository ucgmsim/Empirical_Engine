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

    class CAV(IM):
        """Cumulative absolute velocity."""

        def __init__(self) -> None: ...

    class Ds575(IM):
        """5-75% significant duration."""

        def __init__(self) -> None: ...

    class Ds595(IM):
        """5-95% significant duration."""

        def __init__(self) -> None: ...

    class Ds2080(IM):
        """20-80% significant duration."""

        def __init__(self) -> None: ...

    class EAS(IM):
        """Effective amplitude spectrum at `frequency` Hz."""

        frequency: float
        def __init__(self, frequency: float) -> None: ...

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

class P20Inputs:
    """P20 input columns, one value per rupture-site pair."""

    @staticmethod
    def interface(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
    ) -> P20Inputs:
        """Subduction interface inputs."""

    @staticmethod
    def slab(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        hypo_depth: npt.NDArray[np.float64],
    ) -> P20Inputs:
        """Subduction slab inputs."""

def p20(inputs: P20Inputs, im: IM) -> GMMResult: ...

class K20Inputs:
    """K20 input columns, one value per rupture-site pair (z1pt0 in metres)."""

    @staticmethod
    def interface(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        z1pt0: npt.NDArray[np.float64],
    ) -> K20Inputs:
        """Subduction interface inputs."""

    @staticmethod
    def slab(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        z1pt0: npt.NDArray[np.float64],
        backarc: npt.NDArray[np.bool_],
    ) -> K20Inputs:
        """Subduction slab inputs."""

def k20(
    inputs: K20Inputs,
    im: IM,
    sigma_mu_epsilon: float,
    modified_sigma: bool,
) -> GMMResult: ...
def k20_nz(inputs: K20Inputs, im: IM) -> GMMResult: ...

class AG20Inputs:
    """AG20 input columns, one value per rupture-site pair."""

    @staticmethod
    def interface(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
    ) -> AG20Inputs:
        """Subduction interface inputs."""

    @staticmethod
    def slab(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        backarc: npt.NDArray[np.bool_],
    ) -> AG20Inputs:
        """Subduction slab inputs."""

def ag20(
    inputs: AG20Inputs,
    im: IM,
    new_zealand_region: bool,
    sigma_mu_epsilon: float,
) -> GMMResult: ...

class Branch:
    """Epistemic mean branch."""

    Lower: Branch
    Central: Branch
    Upper: Branch

class A22Inputs:
    """A22 input columns, one value per rupture-site pair."""

    @staticmethod
    def crust(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
    ) -> A22Inputs:
        """Active shallow crust (and volcanic) inputs."""

    @staticmethod
    def interface(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
    ) -> A22Inputs:
        """Subduction interface inputs."""

    @staticmethod
    def slab(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        backarc: npt.NDArray[np.bool_],
    ) -> A22Inputs:
        """Subduction slab inputs."""

class S22Inputs:
    """S22 input columns, one value per rupture-site pair (z1pt0 in metres)."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        dip: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        rjb: npt.NDArray[np.float64],
        rx: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        z1pt0: npt.NDArray[np.float64],
    ) -> None: ...

def a22(
    inputs: A22Inputs,
    im: IM,
    branch: Branch,
    modified_sigma: bool,
) -> GMMResult: ...
def s22(
    inputs: S22Inputs,
    im: IM,
    branch: Branch,
) -> GMMResult: ...

class Ask14Inputs:
    """ASK14 input columns, one value per rupture-site pair (z1pt0 in metres)."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        rx: npt.NDArray[np.float64],
        ry0: npt.NDArray[np.float64],
        dip: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        width: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        z1pt0: npt.NDArray[np.float64],
        vs30measured: npt.NDArray[np.bool_],
    ) -> None: ...

class Bssa14Inputs:
    """BSSA14 input columns, one value per rupture-site pair."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        rjb: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
    ) -> None: ...

def ask14(
    inputs: Ask14Inputs,
    im: IM,
    sigma_mu_epsilon: float,
) -> GMMResult: ...
def bssa14(
    inputs: Bssa14Inputs,
    im: IM,
    sigma_mu_epsilon: float,
) -> GMMResult: ...

class CY14Inputs:
    """CY14 input columns, one value per rupture-site pair (z1pt0 in metres)."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        dip: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        rjb: npt.NDArray[np.float64],
        rx: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        vs30measured: npt.NDArray[np.bool_],
        z1pt0: npt.NDArray[np.float64],
    ) -> None: ...

def cy14(
    inputs: CY14Inputs,
    im: IM,
    sigma_mu_epsilon: float,
) -> GMMResult: ...

class Br13Inputs:
    """Br13 input columns, one value per rupture-site pair (z1pt0 in metres)."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        dip: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        rjb: npt.NDArray[np.float64],
        rx: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        vs30measured: npt.NDArray[np.bool_],
        z1pt0: npt.NDArray[np.float64],
        is_aftershock: npt.NDArray[np.bool_],
    ) -> None: ...

def br13(
    inputs: Br13Inputs,
    im: IM,
    sigma_mu_epsilon: float,
    volcanic: bool,
) -> GMMResult: ...

class CB14Inputs:
    """CB14 input columns, one value per rupture-site pair (degrees, km)."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        dip: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        zbot: npt.NDArray[np.float64],
        hypo_depth: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        rjb: npt.NDArray[np.float64],
        rx: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        z2pt5: npt.NDArray[np.float64],
    ) -> None: ...

def cb14(inputs: CB14Inputs, im: IM, sigma_mu_epsilon: float) -> GMMResult: ...

class CB10Inputs:
    """CB10 input columns, one value per rupture-site pair (degrees, km)."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        dip: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        rjb: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        z2pt5: npt.NDArray[np.float64],
    ) -> None: ...

def cb10(inputs: CB10Inputs, im: IM) -> GMMResult: ...

class Za06Inputs:
    """ZA_06 input columns, one value per rupture-site pair."""

    @staticmethod
    def asc(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        hypo_depth: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
    ) -> Za06Inputs:
        """Active shallow crust inputs."""

    @staticmethod
    def interface(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        hypo_depth: npt.NDArray[np.float64],
    ) -> Za06Inputs:
        """Subduction interface inputs."""

    @staticmethod
    def slab(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        hypo_depth: npt.NDArray[np.float64],
    ) -> Za06Inputs:
        """Subduction slab inputs."""

def za06(inputs: Za06Inputs, im: IM, extrapolate: bool) -> GMMResult: ...

class A18Inputs:
    """A18 input columns, one value per rupture-site pair."""

    @staticmethod
    def interface(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
    ) -> A18Inputs:
        """Subduction interface inputs."""

    @staticmethod
    def slab(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
    ) -> A18Inputs:
        """Subduction slab inputs."""

def a18(inputs: A18Inputs, im: IM) -> GMMResult: ...

class Bch16Inputs:
    """BCH_16 input columns, one value per rupture-site pair."""

    @staticmethod
    def interface(
        mag: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        xvf: npt.NDArray[np.float64],
    ) -> Bch16Inputs:
        """Subduction interface inputs."""

    @staticmethod
    def slab(
        mag: npt.NDArray[np.float64],
        rhypo: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        xvf: npt.NDArray[np.float64],
        hypo_depth: npt.NDArray[np.float64],
    ) -> Bch16Inputs:
        """Subduction slab inputs."""

def bch16(inputs: Bch16Inputs, im: IM) -> GMMResult: ...

class AS16Inputs:
    """AS16 input columns, one value per rupture-site pair (z1pt0 in metres)."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        z1pt0: npt.NDArray[np.float64],
    ) -> None: ...

def as16(inputs: AS16Inputs, im: IM) -> GMMResult: ...

class BA18Inputs:
    """BA18 input columns, one value per rupture-site pair (z1pt0 in metres)."""

    def __init__(
        self,
        mag: npt.NDArray[np.float64],
        rake: npt.NDArray[np.float64],
        ztor: npt.NDArray[np.float64],
        rrup: npt.NDArray[np.float64],
        vs30: npt.NDArray[np.float64],
        z1pt0: npt.NDArray[np.float64],
    ) -> None: ...

def ba18(inputs: BA18Inputs, im: IM) -> GMMResult: ...
