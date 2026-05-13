"""Xarray interface for running OpenQuake Ground Motion Models and Logic Trees."""

import re
from collections.abc import Sequence
from typing import Any

import pandas as pd
import xarray as xr

from . import constants, wrapper

GENERAL_RX = re.compile(
    r"""
    (?P<im>[^_]+)               # Capture group 'im': Matches one or more characters 
                                # that are NOT an underscore (stops at first '_')
    
    (?:                         # Start of a non-capturing group for the optional value
        _                       # A literal underscore separator
        (?P<val>[\d\.eE\-\+]+)  # Capture group 'val': Matches digits, periods, or scientific notation. 
    )?                          # The '?' makes the entire '_value' portion optional
    
    _                           # A literal underscore separator before the statistic
    
    (?P<statistic>.+)           # Capture group 'statistic': Matches everything else 
                                # to the end of the string
    """,
    re.VERBOSE,
)


def _pack_dataset(
    results: pd.DataFrame,
) -> xr.Dataset:
    """Pack the results of run_gmm(_logic_tree) into a multi-dimensional dataset.

    Parameters
    ----------
    results : pd.DataFrame
        The results of run_gmm(_logic_tree).

    Returns
    -------
    Dataset
        A multi-dimensional dataset extracted the dataframe, analysing the
        columns to extract the intensity measure, period and frequency.
    """
    test_column = results.columns[0]
    extracted = results.columns.str.extract(GENERAL_RX)
    levels = ["im"]

    match (match := GENERAL_RX.match(test_column)) and match.groupdict():
        case {"im": "pSA"}:
            extracted = extracted.rename(columns=dict(val="period"))
            extracted["period"] = pd.to_numeric(extracted["period"])
            levels.append("period")
        case {"im": "EAS"}:
            extracted = extracted.rename(columns=dict(val="frequency"))
            extracted["frequency"] = pd.to_numeric(extracted["frequency"])
            levels.append("frequency")
        case None:
            raise ValueError(
                f"Could not extract dimensions for test column: {test_column!r}"
            )
        case _:
            extracted = extracted.drop("val", axis="columns")

    results.columns = pd.MultiIndex.from_frame(extracted)
    results = results.stack(level=levels)  # ty: ignore[invalid-assignment]
    # match is None is impossible because a ValueError would have already been
    # thrown at this point.
    assert match is not None
    im_name = match.group("im")

    dset = results.to_xarray()
    dset = dset.sel(im=im_name, drop=True)
    dset.attrs["intensity_measure"] = im_name

    return dset


def run_gmm_xarray(
    model: constants.GMM,
    tect_type: constants.TectType,
    inputs: xr.Dataset,
    im: str,
    periods: Sequence[float | int] | None = None,
    frequencies: Sequence[float | int] | None = None,
    epistemic_branch: constants.EpistemicBranch = constants.EpistemicBranch.CENTRAL,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Run OpenQuake GMM for the given xarray dataset and intensity measure.

    Parameters
    ----------
    model : constants.GMM
        Ground motion model identifier
    tect_type : constants.TectType
        Tectonic type
    inputs : xr.Dataset
        Xarray dataset containing variables like vs30, rrup, mag.
        Dimensions do not need to match natively; they will be broadcast together.
    im : str
        Intensity measure (e.g., 'PGA', 'pSA', 'EAS')
    periods : Sequence[float | int], optional
        Periods to compute for pSA
    frequencies : Sequence[float | int], optional
        Frequencies to compute for EAS
    epistemic_branch : constants.EpistemicBranch
        Epistemic branch mapping
    **kwargs : Any
        Options passed to `wrapper.run_gmm`

    Returns
    -------
    xr.Dataset
        A Dataset containing the GMM results. Statistics (mean, std_Total, etc.)
        are stored as data variables. The intensity measure type is stored
        in the global attributes.

    Examples
    --------
    >>> import oq_wrapper as oqw
    >>> import oq_wrapper.xarray as oqwx
    >>> import xarray as xr
    >>> import numpy as np

    >>> # Create a dataset with heterogeneous dimensions:
    >>> # rrup (station), vs30 (model, station), and mag (scalar)
    >>> rng = np.random.default_rng(seed=0)
    >>> inputs = xr.Dataset(
    ...     data_vars=dict(
    ...         rrup=('station', np.linspace(1, 100, num=10)),
    ...         vs30=(('model', 'station'), rng.uniform(500, 1500, size=(5, 10))),
    ...         mag=8.3
    ...     ),
    ...     coords=dict(
    ...         station=[f'a_{i}' for i in range(1, 11)],
    ...         model=['a', 'b', 'c', 'd', 'e']
    ...     )
    ... )

    >>> # Run the GMM across the entire dimension grid
    >>> psa = oqwx.run_gmm_xarray(
    ...     oqw.constants.GMM.A_22,
    ...     oqw.constants.TectType.ACTIVE_SHALLOW,
    ...     inputs,
    ...     'pSA',
    ...     periods=[1.0, 5.0]
    ... )

    >>> # Statistics become data variables
    >>> sorted(psa)
    ['mean', 'std_Inter', 'std_Intra', 'std_Total']

    >>> # Input dimensions are broadcast with periods (or frequencies).
    >>> sorted(psa.coords.keys())
    ['model', 'period', 'station']

    >>> # Get the mean for model 'd' and 'a_10' at 1.0s.
    >>> psa['mean'].sel(station='a_10', model='d').sel(period=1.0, method='nearest').item()
    ...

    See Also
    --------
    run_gmm : The lower level wrapper function for pandas dataframes.
    """
    rupture_df = inputs.to_dataframe()

    df_result = wrapper.run_gmm(
        model=model,
        tect_type=tect_type,
        rupture_df=rupture_df,
        im=im,
        periods=periods,
        frequencies=frequencies,
        epistemic_branch=epistemic_branch,
        **kwargs,
    )

    return _pack_dataset(df_result)


def run_gmm_logic_tree_xarray(
    gmm_lt: constants.GMMLogicTree,
    tect_type: constants.TectType,
    inputs: xr.Dataset,
    im: str,
    periods: Sequence[float | int] | None = None,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Run OpenQuake GMM Logic Tree for the given xarray dataset and intensity measure.

    Parameters
    ----------
    gmm_lt : constants.GMMLogicTree
        Ground motion model logic tree identifier
    tect_type : constants.TectType
        Tectonic type
    inputs : xr.Dataset
        Xarray dataset containing variables like vs30, rrup, mag.
        Dimensions do not need to match natively; they will be broadcast together.
    im : str
        Intensity measure (e.g., 'PGA', 'pSA', 'EAS')
    periods : Sequence[float | int], optional
        Periods to compute for pSA
    **kwargs : Any
        Keyword arguments passed to `run_gmm_logic_tree`

    Returns
    -------
    xr.Dataset
        A Dataset containing the GMM results. Statistics (mean, std_Total)
        are stored as data variables. The intensity measure type is stored
        in the global attributes.

    Examples
    --------
    >>> import oq_wrapper as oqw
    >>> import oq_wrapper.xarray as oqwx
    >>> import xarray as xr
    >>> import numpy as np

    >>> rng = np.random.default_rng(seed=0)
    >>> inputs = xr.Dataset(
    ...     dict(
    ...         rrup=("station", np.linspace(1, 100, num=10)),
    ...         rjb=("station", rng.uniform(1, 100, size=(10,))),
    ...         rx=("station", rng.uniform(1, 100, size=(10,))),
    ...         ry=("station", rng.uniform(1, 100, size=(10,))),
    ...         hyp=("station", rng.uniform(1, 100, size=(10,))),
    ...         epi=("station", rng.uniform(1, 100, size=(10,))),
    ...         ztor=0.0,
    ...         dip=60.0,
    ...         rake=15.0,
    ...         hypo_depth=5.0,
    ...         zbot=10.0,
    ...         vs30measured=False,
    ...         vs30=(("model", "station"), rng.uniform(500, 1500, size=(5, 10))),
    ...         z1pt0=(("model", "station"), rng.uniform(0.5, 1.5, size=(5, 10))),
    ...         z2pt5=(("model", "station"), rng.uniform(0.5, 1.5, size=(5, 10))),
    ...         mag=8.3,
    ...     ),
    ...     coords=dict(
    ...         station=[f"a_{i}" for i in range(1, 11)], model=["a", "b", "c", "d", "e"]
    ...     ),
    ... )

    >>> # Run a Logic Tree
    >>> psa = oqwx.run_gmm_logic_tree_xarray(
    ...     oqw.constants.GMMLogicTree.NSHM2022,
    ...     oqw.constants.TectType.ACTIVE_SHALLOW,
    ...     inputs,
    ...     'pSA',
    ...     periods=list(np.linspace(1, 10, 50))
    ... )

    >>> # Statistics become data variables
    >>> sorted(psa)
    ['mean', 'std_Total']

    >>> # Input dimensions are broadcast with periods (or frequencies).
    >>> sorted(psa.coords.keys())
    ['model', 'period', 'station']

    >>> # Get the mean for model 'd' and 'a_10' at 1.0s.
    >>> psa['mean'].sel(station='a_10', model='d').sel(period=1.0, method='nearest').item()
    ...

    See Also
    --------
    run_gmm_logic_tree : The lower level wrapper function for pandas dataframes.
    """
    rupture_df = inputs.to_dataframe()

    df_result = wrapper.run_gmm_logic_tree(
        gmm_lt,
        tect_type=tect_type,
        rupture_df=rupture_df,
        im=im,
        periods=periods,
        **kwargs,
    )

    return _pack_dataset(df_result)
