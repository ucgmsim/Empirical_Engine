"""The ground motion models, each a `Model` around its Rust implementation.

A model's GMPE math lives in the compiled `gm_models._core` extension (see
`src-rust/`). Its inputs constructors' signatures name the columns it needs,
which `Model.run` takes from the rupture dataframe (OpenQuake column names,
Z1.0 in km, as for `oq_wrapper.run_gmm`).
"""

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

from . import _core, utils
from .constants import CORE_BRANCHES, SIGMA_MU_EPSILON, EpistemicBranch, TectType


def _z1pt0_m(rupture_df: pd.DataFrame) -> np.ndarray:
    """Return ``z1pt0`` in metres, as the Rust models (and OpenQuake) take it.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        Rupture/site dataframe, ``z1pt0`` in km.

    Returns
    -------
    np.ndarray
        Z1.0 in metres.

    Raises
    ------
    ValueError
        If ``z1pt0`` looks like it is already in metres.
    """
    z1pt0 = rupture_df["z1pt0"].to_numpy(dtype=np.float64)
    if (z1pt0 >= 10).any():
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )
    return z1pt0 * 1000


# Optional columns and their default.
_DEFAULTS = {"vs30measured": False, "is_aftershock": False, "xvf": 0.0}
_BOOL_COLUMNS = {"backarc", "vs30measured", "is_aftershock"}


@dataclass(frozen=True, eq=False)
class Model:
    """A ground motion model.

    Attributes
    ----------
    name : str
        Model name, as in `oq_wrapper` (e.g. ``"ASK_14"``).
    core : Callable[..., _core.GMMResult]
        Rust model function, e.g. `_core.ask14`. Called with the inputs, the
        IM, the epistemic argument (if `epistemic`), ``modified_sigma`` (if
        `modified_sigma`) and `options` as keywords.
    inputs : Mapping[TectType, Callable[..., object]]
        Rust inputs constructor per supported tectonic type, e.g.
        `_core.A18Inputs.slab`. Its parameter names are the columns it takes.
    epistemic : Mapping[EpistemicBranch, object] or None
        `core`'s epistemic argument per branch (`SIGMA_MU_EPSILON` or
        `CORE_BRANCHES`), None if the model has no epistemic branches.
    modified_sigma : bool
        Whether `core` takes ``modified_sigma``.
    options : Mapping[str, bool]
        Further keyword options of `core` and their defaults.
    fixed : Mapping[str, float or bool]
        Inputs the model fixes rather than reading from the rupture dataframe.
    check : Callable[[pd.DataFrame], None] or None
        Extra validation of the rupture dataframe, raising ``ValueError``.
    """

    name: str
    core: Callable[..., _core.GMMResult]
    inputs: Mapping[TectType, Callable[..., object]]
    epistemic: Mapping[EpistemicBranch, object] | None = None
    modified_sigma: bool = False
    options: Mapping[str, bool] = field(default_factory=dict)
    fixed: Mapping[str, float | bool] = field(default_factory=dict)
    check: Callable[[pd.DataFrame], None] | None = None

    def run(
        self,
        tect_type: TectType,
        rupture_df: pd.DataFrame,
        im: str,
        periods: Sequence[float] | np.ndarray | None = None,
        *,
        frequencies: Sequence[float] | np.ndarray | None = None,
        epistemic_branch: EpistemicBranch | None = None,
        modified_sigma: bool | None = None,
        **options: bool,
    ) -> xr.Dataset:
        """Run the model.

        Parameters
        ----------
        tect_type : TectType
            Tectonic type, one of `inputs`.
        rupture_df : pd.DataFrame
            Rupture/site dataframe with the columns the model needs for
            `tect_type` (a missing one is reported).
        im : str
            Intensity measure, see `gm_models.utils.evaluate_ims`.
        periods : Sequence[float] or np.ndarray, optional
            Periods to compute, required if `im` is ``"pSA"``.
        frequencies : Sequence[float] or np.ndarray, optional
            Frequencies (Hz) to compute, required if `im` is ``"EAS"``.
        epistemic_branch : EpistemicBranch, optional
            Epistemic branch, `EpistemicBranch.CENTRAL` by default. Only for
            models with `epistemic` branches.
        modified_sigma : bool, optional
            Whether to use the NZ NSHM 2022 "modified sigma" model, True by
            default as in the logic tree (Bradley et al. 2024) and
            `oq_wrapper`. Only for models with `modified_sigma`.
        **options : bool
            Model `options`, e.g. ``extrapolate`` for ZA_06.

        Returns
        -------
        xr.Dataset
            Variables ``ln_mean`` (natural-log space), ``std_total``,
            ``std_inter`` and ``std_intra`` over `rupture_df`'s index (plus
            ``period`` or ``frequency``), with the model settings as
            attributes. See `gm_models.utils.evaluate_ims`.

        Raises
        ------
        ValueError
            If `tect_type` is unsupported, an argument does not apply to the
            model, `rupture_df` is missing columns or fails `check`, or the
            model rejects `im`, `periods` or `frequencies`.
        """
        if tect_type not in self.inputs:
            raise ValueError(f"{self.name} does not support tectonic type {tect_type}")
        if unknown := options.keys() - self.options.keys():
            raise ValueError(f"{self.name} has no options {sorted(unknown)}")
        if self.epistemic is None and epistemic_branch is not None:
            raise ValueError(f"{self.name} has no epistemic branches")
        if not self.modified_sigma and modified_sigma is not None:
            raise ValueError(f"{self.name} has no modified sigma model")
        if self.check is not None:
            self.check(rupture_df)

        ctor = self.inputs[tect_type]
        columns = inspect.signature(ctor).parameters.keys()
        utils.validate_columns(
            rupture_df, columns - self.fixed.keys() - _DEFAULTS.keys()
        )
        inputs = ctor(
            *(
                np.full(len(rupture_df), self.fixed[name])
                if name in self.fixed
                else _z1pt0_m(rupture_df)
                if name == "z1pt0"
                else rupture_df.get(
                    name, pd.Series(_DEFAULTS.get(name), index=rupture_df.index)
                ).to_numpy(dtype=np.bool_ if name in _BOOL_COLUMNS else np.float64)
                for name in columns
            )
        )

        # Attributes as int where bool: netCDF attributes can't be bool.
        args: list[object] = []
        attrs: dict[str, str | int | float] = {
            "model": self.name,
            "tect_type": str(tect_type),
        }
        if self.epistemic is not None:
            epistemic_branch = epistemic_branch or EpistemicBranch.CENTRAL
            args.append(self.epistemic[epistemic_branch])
            attrs["epistemic_branch"] = str(epistemic_branch)
        if self.modified_sigma:
            modified_sigma = True if modified_sigma is None else modified_sigma
            args.append(modified_sigma)
            attrs["modified_sigma"] = int(modified_sigma)
        options = {**self.options, **options}
        attrs |= {key: int(value) for key, value in options.items()}
        return utils.evaluate_ims(
            lambda core_im: self.core(inputs, core_im, *args, **options),
            rupture_df.index,
            im,
            periods,
            attrs=attrs,
            frequencies=frequencies,
        )


_CRUSTAL = (TectType.ACTIVE_SHALLOW, TectType.VOLCANIC)


def _reject_aftershocks(rupture_df: pd.DataFrame) -> None:
    """Raise ``ValueError`` if `rupture_df` flags aftershocks (ASK_14 has no aftershock term).

    Parameters
    ----------
    rupture_df : pd.DataFrame
        Rupture/site dataframe, with an optional ``is_aftershock`` column.
    """
    if rupture_df.get("is_aftershock", pd.Series(False)).any():
        raise ValueError("ASK_14 aftershock scaling is not supported")


# NZ NSHM2022 modification of Kuehn et al. (2020), global (GLO) region: backarc
# term, modified sigma, epistemic branches.
K_20 = Model(
    "K_20",
    _core.k20,
    {
        TectType.SUBDUCTION_INTERFACE: _core.K20Inputs.interface,
        TectType.SUBDUCTION_SLAB: _core.K20Inputs.slab,
    },
    epistemic=SIGMA_MU_EPSILON,
    modified_sigma=True,
    # No basin term in the global region: NaN depths are ignored.
    fixed={"z1pt0": np.nan},
)
# Kuehn et al. (2020) with the New Zealand regional coefficients and basin term.
K_20_NZ = Model("K_20_NZ", _core.k20_nz, K_20.inputs, fixed={"backarc": False})
# Parker et al. (2020) NGA-Subduction global model, no regional adjustments.
P_20 = Model(
    "P_20",
    _core.p20,
    {
        TectType.SUBDUCTION_INTERFACE: _core.P20Inputs.interface,
        TectType.SUBDUCTION_SLAB: _core.P20Inputs.slab,
    },
)
# NZ NSHM2022 modification of Parker et al. (2020), global (GLO) region.
P_21 = Model(
    "P_21",
    _core.p21,
    {
        TectType.SUBDUCTION_INTERFACE: _core.P21Inputs.interface,
        TectType.SUBDUCTION_SLAB: _core.P21Inputs.slab,
    },
    epistemic=SIGMA_MU_EPSILON,
    modified_sigma=True,
)
# NZ NSHM2022 modification of Abrahamson and Gulerce (2020), global (GLO)
# region with the slab backarc term and epistemic branches.
AG_20 = Model(
    "AG_20",
    _core.ag20,
    {
        TectType.SUBDUCTION_INTERFACE: _core.AG20Inputs.interface,
        TectType.SUBDUCTION_SLAB: _core.AG20Inputs.slab,
    },
    epistemic=SIGMA_MU_EPSILON,
)
# Abrahamson and Gulerce (2020) with the New Zealand (NZL) regional
# adjustment, no backarc term.
AG_20_NZ = Model("AG_20_NZ", _core.ag20_nz, AG_20.inputs, fixed={"backarc": False})
# Atkinson (2022) NZ NSHM backbone model. Volcanic uses the crustal model, as
# in oq_wrapper.
A_22 = Model(
    "A_22",
    _core.a22,
    {
        **dict.fromkeys(_CRUSTAL, _core.A22Inputs.crust),
        TectType.SUBDUCTION_INTERFACE: _core.A22Inputs.interface,
        TectType.SUBDUCTION_SLAB: _core.A22Inputs.slab,
    },
    epistemic=CORE_BRANCHES,
    modified_sigma=True,
)
# Stafford (2022) NZ NSHM backbone model (modified Chiou and Youngs 2014).
S_22 = Model(
    "S_22",
    _core.s22,
    dict.fromkeys(_CRUSTAL, _core.S22Inputs),
    epistemic=CORE_BRANCHES,
)
# Abrahamson, Silva and Kamai (2014), default California model for mainshocks.
# ``vs30measured`` defaults to False.
ASK_14 = Model(
    "ASK_14",
    _core.ask14,
    dict.fromkeys(_CRUSTAL, _core.Ask14Inputs),
    epistemic=SIGMA_MU_EPSILON,
    check=_reject_aftershocks,
)
# Boore, Stewart, Seyhan and Atkinson (2014), global model without basin term.
BSSA_14 = Model(
    "BSSA_14",
    _core.bssa14,
    dict.fromkeys(_CRUSTAL, _core.Bssa14Inputs),
    epistemic=SIGMA_MU_EPSILON,
)
# Chiou and Youngs (2014), default OpenQuake configuration (California region,
# hanging-wall term on).
CY_14 = Model(
    "CY_14",
    _core.cy14,
    dict.fromkeys(_CRUSTAL, _core.CY14Inputs),
    epistemic=SIGMA_MU_EPSILON,
)
# Bradley (2013) NZ model, `Bradley2013` (active shallow) and `Bradley2013Volc`
# (volcanic).
Br_13 = Model(
    "Br_13",
    _core.br13,
    {
        TectType.ACTIVE_SHALLOW: _core.Br13Inputs.crust,
        TectType.VOLCANIC: _core.Br13Inputs.volcanic,
    },
    epistemic=SIGMA_MU_EPSILON,
)
# Campbell and Bozorgnia (2014), rupture width always estimated (Eq. 39).
CB_14 = Model(
    "CB_14",
    _core.cb14,
    dict.fromkeys(_CRUSTAL, _core.CB14Inputs),
    epistemic=SIGMA_MU_EPSILON,
)
# Campbell and Bozorgnia (2008) with the 2010 CAV model.
CB_10 = Model("CB_10", _core.cb10, {TectType.ACTIVE_SHALLOW: _core.CB10Inputs})
# Zhao et al. (2006). `extrapolate` is oq_wrapper's constant-displacement
# extrapolation above 5 s.
ZA_06 = Model(
    "ZA_06",
    _core.za06,
    {
        TectType.ACTIVE_SHALLOW: _core.Za06Inputs.asc,
        TectType.SUBDUCTION_INTERFACE: _core.Za06Inputs.interface,
        TectType.SUBDUCTION_SLAB: _core.Za06Inputs.slab,
    },
    options={"extrapolate": False},
)
# Abrahamson et al. (2018) BC Hydro update (Cascadia adjustment, central branch).
A_18 = Model(
    "A_18",
    _core.a18,
    {
        TectType.SUBDUCTION_INTERFACE: _core.A18Inputs.interface,
        TectType.SUBDUCTION_SLAB: _core.A18Inputs.slab,
    },
)
# BC Hydro (Abrahamson et al. 2016) with the ESHM20 theta6 calibration and the
# "Step" forearc/backarc taper. ``xvf`` (signed distance to the volcanic front
# in km, positive on the forearc side, negative on the backarc side) defaults
# to 0, as in oq_wrapper.
BCH_16 = Model(
    "BCH_16",
    _core.bch16,
    {
        TectType.SUBDUCTION_INTERFACE: _core.Bch16Inputs.interface,
        TectType.SUBDUCTION_SLAB: _core.Bch16Inputs.slab,
    },
)
# Afshari and Stewart (2016) significant durations (Ds575, Ds595, Ds2080).
AS_16 = Model("AS_16", _core.as16, {TectType.ACTIVE_SHALLOW: _core.AS16Inputs})
# Bayless and Abrahamson (2018) effective amplitude spectrum (EAS only).
BA_18 = Model("BA_18", _core.ba18, {TectType.ACTIVE_SHALLOW: _core.BA18Inputs})
