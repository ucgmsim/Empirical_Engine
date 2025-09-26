"""
Module containing functions of the OQ GMM wrapper.
The two key functions for running GMMs are:
- `run_gmm`: Runs a single GMM for the given rupture dataframe and IM.
- `run_gmm_logic_tree`: Runs a logic tree of GMMs for the given rupture dataframe and IM.
"""

import functools
import logging
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
import yaml
from openquake.hazardlib import const as oq_const
from openquake.hazardlib import contexts, gsim, imt

from . import constants, estimations


def _oq_model(model: gsim.base.GMPE, **kwargs: dict[str, Any]) -> gsim.base.GMPE:
    """
    Create a partial function to simplify model instantiation.

    Parameters
    ----------
    model : gsim.base.GMPE
        OpenQuake ground motion model class
    **kwargs : dict
        Extra model-specific parameters, e.g.:
        - region="NZL" for models ending with _NZ
        - estimate_width=True for CB_14 to estimate fault width

    Returns
    -------
    gsim.base.GMPE
        Instantiated OpenQuake ground motion model
    """
    return model(**kwargs)


OQ_MODEL_MAPPING = {
    constants.GMM.AS_16: {
        constants.TectType.ACTIVE_SHALLOW: gsim.afshari_stewart_2016.AfshariStewart2016
    },
    constants.GMM.A_18: {
        constants.TectType.SUBDUCTION_SLAB: gsim.abrahamson_2018.AbrahamsonEtAl2018SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_2018.AbrahamsonEtAl2018SInter,
    },
    # OQ's CB_08 includes the CB_10's CAV
    constants.GMM.CB_10: {
        constants.TectType.ACTIVE_SHALLOW: gsim.campbell_bozorgnia_2008.CampbellBozorgnia2008,
    },
    constants.GMM.BCH_16: {
        constants.TectType.SUBDUCTION_SLAB: gsim.bchydro_2016_epistemic.BCHydroESHM20SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.bchydro_2016_epistemic.BCHydroESHM20SInter,
    },
    constants.GMM.ZA_06: {
        constants.TectType.ACTIVE_SHALLOW: gsim.zhao_2006.ZhaoEtAl2006Asc,
        constants.TectType.SUBDUCTION_SLAB: gsim.zhao_2006.ZhaoEtAl2006SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.zhao_2006.ZhaoEtAl2006SInter,
    },
    constants.GMM.K_20_NZ: {
        constants.TectType.SUBDUCTION_SLAB: functools.partial(
            _oq_model, model=gsim.kuehn_2020.KuehnEtAl2020SSlab, region="NZL"
        ),
        constants.TectType.SUBDUCTION_INTERFACE: functools.partial(
            _oq_model, model=gsim.kuehn_2020.KuehnEtAl2020SInter, region="NZL"
        ),
    },
    constants.GMM.AG_20_NZ: {
        constants.TectType.SUBDUCTION_SLAB: functools.partial(
            _oq_model,
            model=gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
            region="NZL",
        ),
        constants.TectType.SUBDUCTION_INTERFACE: functools.partial(
            _oq_model,
            model=gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
            region="NZL",
        ),
    },
    constants.GMM.S_22: {
        constants.TectType.ACTIVE_SHALLOW: gsim.nz22.stafford_2022.Stafford2022,
        constants.TectType.VOLCANIC: gsim.nz22.stafford_2022.Stafford2022,
    },
    constants.GMM.A_22: {
        constants.TectType.ACTIVE_SHALLOW: gsim.nz22.atkinson_2022.Atkinson2022Crust,
        constants.TectType.VOLCANIC: gsim.nz22.atkinson_2022.Atkinson2022Crust,
        constants.TectType.SUBDUCTION_SLAB: gsim.nz22.atkinson_2022.Atkinson2022SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.nz22.atkinson_2022.Atkinson2022SInter,
    },
    constants.GMM.ASK_14: {
        constants.TectType.ACTIVE_SHALLOW: gsim.abrahamson_2014.AbrahamsonEtAl2014,
        constants.TectType.VOLCANIC: gsim.abrahamson_2014.AbrahamsonEtAl2014,
    },
    constants.GMM.CY_14: {
        constants.TectType.ACTIVE_SHALLOW: gsim.chiou_youngs_2014.ChiouYoungs2014,
        constants.TectType.VOLCANIC: gsim.chiou_youngs_2014.ChiouYoungs2014,
    },
    constants.GMM.CB_14: {
        constants.TectType.ACTIVE_SHALLOW: functools.partial(
            _oq_model,
            model=gsim.campbell_bozorgnia_2014.CampbellBozorgnia2014,
            estimate_width=True,
        ),
        constants.TectType.VOLCANIC: functools.partial(
            _oq_model,
            model=gsim.campbell_bozorgnia_2014.CampbellBozorgnia2014,
            estimate_width=True,
        ),
    },
    constants.GMM.BSSA_14: {
        constants.TectType.ACTIVE_SHALLOW: gsim.boore_2014.BooreEtAl2014,
        constants.TectType.VOLCANIC: gsim.boore_2014.BooreEtAl2014,
    },
    constants.GMM.Br_13: {
        constants.TectType.ACTIVE_SHALLOW: gsim.bradley_2013.Bradley2013,
        constants.TectType.VOLCANIC: gsim.bradley_2013.Bradley2013Volc,
    },
    constants.GMM.AG_20: {
        constants.TectType.SUBDUCTION_SLAB: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
    },
    constants.GMM.P_20: {
        constants.TectType.SUBDUCTION_SLAB: gsim.parker_2020.ParkerEtAl2020SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.parker_2020.ParkerEtAl2020SInter,
    },
    constants.GMM.K_20: {
        constants.TectType.SUBDUCTION_SLAB: gsim.kuehn_2020.KuehnEtAl2020SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.kuehn_2020.KuehnEtAl2020SInter,
    },
    constants.GMM.P_21: {
        constants.TectType.SUBDUCTION_SLAB: gsim.nz22.nz_nshm2022_parker.NZNSHM2022_ParkerEtAl2020SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.nz22.nz_nshm2022_parker.NZNSHM2022_ParkerEtAl2020SInter,
    },
    constants.GMM.GA_11: {
        constants.TectType.ACTIVE_SHALLOW: gsim.gulerce_abrahamson_2011.GulerceAbrahamson2011
    },
    constants.GMM.BA_18: {
        constants.TectType.ACTIVE_SHALLOW: gsim.bayless_abrahamson_2018.BaylessAbrahamson2018
    },
}


def run_gmm(
    model: constants.GMM,
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[int | float] | None = None,
    frequencies: Sequence[int | float] | None = None,
    epistemic_branch: constants.EpistemicBranch = constants.EpistemicBranch.CENTRAL,
    **kwargs: dict[str, Any],
) -> pd.DataFrame:
    """
    Run OpenQuake GMM for the given rupture dataframe and intensity measure.

    Parameters
    ----------
    model : constants.GMM
        Ground motion model identifier
    tect_type : constants.TectType
        Tectonic type (ACTIVE_SHALLOW, SUBDUCTION_SLAB or SUBDUCTION_INTERFACE)
    rupture_df : pd.DataFrame
        DataFrame containing rupture and site parameters.
        Columns should include properties like vs30, z1pt0, rrup, rjb, mag, rake, dip, etc.
        Note: z1pt0 needs to be in km.
        Each row represents a separate site-fault pair.
    im : str
        Intensity measure (e.g., 'PGA', 'PGV', 'pSA', 'Ds575', 'Ds595')
    periods : sequence of ints or floats, optional
        Periods to compute for pSA, required if im is 'pSA'.
        Ignored for other IMs.
    frequencies : sequence of ints or floats, optional
        Frequencies to compute for EAS, required if im is 'EAS'.
        Ignored for other IMs.
    epistemic_branch : constants.EpistemicBranch
        Epistemic uncertainty branch to use for the model.
        Defaults to constants.EpistemicBranch.CENTRAL.
    **kwargs
        Extra parameters passed to the model constructor

    Returns
    -------
    pd.DataFrame
        DataFrame containing IM mean and standard deviation for each entry in rupture_df.
        Following columns per IM/pSA period:
        - mean: {IM}_mean               - Mean lnIM value
        - std_total: {IM}_std_total     - Total standard deviation
        - std_intra: {IM}_std_intra     - Within-event standard deviation
        - std_inter: {IM}_std_inter     - Between-event standard deviation

    Raises
    ------
    ValueError
        If required periods for pSA, required frequencies for EAS are
        not specified, or if the IM is not supported.
    """
    # Get the OQ model
    oq_model, stddev_types = get_oq_model(
        model, tect_type, epistemic_branch=epistemic_branch, **kwargs
    )

    # Prepare inputs
    im = _convert_to_oq_im(im)
    rupture_df = prepare_model_inputs(
        model,
        oq_model,
        rupture_df,
        im,
    )

    # Convert Z1.0 from km to m
    if np.any(rupture_df["z1pt0"] >= 10):
        raise ValueError(
            "Z1.0 values are too high. Did you pass in metre values instead of kilometres?"
        )
    rupture_df["z1pt0"] *= 1000  # this is ok as we are not editing the original df

    # OQ's single new-style context which contains all site, distance and rupture's information
    rupture_ctx = contexts.RuptureContext(
        tuple(
            [
                # Openquake requiring occurrence_rate attribute to exist
                ("occurrence_rate", None),
                # sids is the number of sites provided (OQ term)
                # This term needs to be repeated for the number of rows in the df
                ("sids", [1] * rupture_df.shape[0]),
                *(
                    (
                        column,
                        rupture_df.loc[:, column].values,
                    )
                    for column in rupture_df.columns.values
                ),
            ]
        )
    )

    # Special handling for pSA
    if im.startswith("pSA"):
        if periods is None:
            raise ValueError("Periods must be specified for pSA.")

        result_df = _oq_run_pSA(
            model,
            oq_model,
            rupture_ctx,
            periods,
            stddev_types,
        )
    elif im.startswith("EAS"):
        if frequencies is None:
            raise ValueError("Frequencies must be specified for EAS.")

        result_df = _oq_run_EAS(model, oq_model, rupture_ctx, frequencies, stddev_types)
    else:
        # Check that the IM is supported
        # Supported IMs are specified as functions per model, hence the getattr
        # Have to use the IM type function, as that is what is used to
        # define the allowed IM types for a model.
        oq_im_type_fn = getattr(imt, im)
        if oq_im_type_fn not in oq_model.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
            raise ValueError(
                f"Model {model.name} does not support {im}. Supported types are {oq_model.DEFINED_FOR_INTENSITY_MEASURE_TYPES}"
            )
        result_df = _run_oq_model(oq_model, rupture_ctx, oq_im_type_fn(), stddev_types)

    result_df.index = rupture_df.index

    # Median prediction adjustment using sigma factor
    # Used by 2022 NZ NSHM
    if (
        epistemic_branch is not constants.EpistemicBranch.CENTRAL
        and model in constants.GMM_EPISTEMIC_BRANCH_SIGMA_FACTOR_MAPPING
    ):
        sigma_factor_mapping = constants.GMM_EPISTEMIC_BRANCH_SIGMA_FACTOR_MAPPING[
            model
        ]
        mean_cols = [col for col in result_df.columns if col.endswith("_mean")]
        std_total_cols = [
            col for col in result_df.columns if col.endswith("_std_Total")
        ]
        assert len(mean_cols) == len(std_total_cols)
        assert all(
            [
                c1.rstrip("_mean") == c2.rstrip("_std_Total")
                for c1, c2 in zip(mean_cols, std_total_cols)
            ]
        )

        result_df[mean_cols] = (
            result_df[mean_cols].values
            + sigma_factor_mapping[epistemic_branch] * result_df[std_total_cols].values
        )

    return result_df


def run_gmm_logic_tree(
    gmm_lt: constants.GMMLogicTree,
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[int | float] | None = None,
    return_ind_results: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict]:
    """
    Run a logic tree of GMMs with the specified weights.

    Parameters
    ----------
    gmm_lt : constants.GMMLogicTree
        The logic tree of GMMs to run, available logic trees
        are defined in constants.py
    tect_type : constants.TectType
        Tectonic type (ACTIVE_SHALLOW, SUBDUCTION_SLAB or SUBDUCTION_INTERFACE)
    rupture_df : pd.DataFrame
        DataFrame containing rupture and site parameters
    im : str
        Intensity measure (e.g., 'PGA', 'PGV', 'pSA', 'Ds575', 'Ds595')
    periods : sequence of ints or floats, optional
        Periods to compute for pSA, required if im is 'pSA'
    return_ind_results : bool, optional
        If True, returns individual model results and their weights.
        Defaults to False.

    Returns
    -------
    pd.DataFrame
        DataFrame containing weighted combination of IM results from all models
    dict
        Dictionary containing individual model results
        and their weights if `return_ind_results` is True
    """
    gmm_lt_config = load_gmm_logic_tree_config(
        gmm_lt,
        tect_type,
        im,
    )
    if gmm_lt_config is None:
        raise ValueError(
            f"IM {im} is not supported for GMM logic tree {gmm_lt.name} and tectonic type {tect_type.name}"
        )

    results = []
    ind_results = {}
    for cur_model_name, cur_value in gmm_lt_config.items():
        # Epistemic uncertainy branches per GMM
        cur_model = constants.GMM[cur_model_name]
        if isinstance(cur_value, dict):
            for cur_epistemic_branch, cur_weight in cur_value.items():
                cur_result_df = run_gmm(
                    cur_model,
                    tect_type,
                    rupture_df,
                    im,
                    periods=periods,
                    epistemic_branch=constants.EpistemicBranch(cur_epistemic_branch),
                )
                results.append(cur_result_df * cur_weight)
                ind_results[f"{cur_model}_{cur_epistemic_branch}"] = (
                    cur_weight,
                    cur_result_df,
                )
        # No epistemic uncertainty branches
        else:
            cur_weight = cur_value
            cur_result_df = run_gmm(
                cur_model,
                tect_type,
                rupture_df,
                im,
                periods=periods,
            )
            results.append(cur_result_df * cur_weight)
            ind_results[str(cur_model)] = (cur_weight, cur_result_df)

    im_keys = [im] if not im.startswith("pSA") else [f"pSA_{p}" for p in periods]
    mean_im_keys = [f"{key}_mean" for key in im_keys]
    std_im_keys = [f"{key}_std_Total" for key in im_keys]

    # Compute weighted mean and standard deviation
    lt_mean = sum([w * df[mean_im_keys] for (w, df) in ind_results.values()])
    lt_within_model_var = sum(
        [w * df[std_im_keys] ** 2 for (w, df) in ind_results.values()]
    )
    lt_between_model_var = sum(
        [w * (df[mean_im_keys] - lt_mean) ** 2 for (w, df) in ind_results.values()]
    )
    lt_between_model_var.columns = std_im_keys
    lt_std = np.sqrt(lt_within_model_var + lt_between_model_var)

    result_df = pd.merge(lt_mean, lt_std, left_index=True, right_index=True)
    if return_ind_results:
        return result_df, ind_results
    else:
        return result_df


def get_model_from_str(model_name: str) -> constants.GMM | constants.GMMLogicTree:
    """
    Convert a string to a GMM or GMMLogicTree.

    Parameters
    ----------
    model_name : str
        Name of the model

    Returns
    -------
    Union[constants.GMM, constants.GMMLogicTree]
        Corresponding GMM or GMMLogicTree object

    Raises
    ------
    ValueError
        If the model name is not recognized
    """
    if model_name in constants.GMM:
        return constants.GMM[model_name]
    elif model_name in constants.GMMLogicTree:
        return constants.GMMLogicTree[model_name]
    else:
        raise ValueError(f"Model {model_name} not recognized.")


def load_gmm_logic_tree_config(
    gmm_lt: constants.GMMLogicTree,
    tect_type: constants.TectType,
    im: str,
) -> dict[str, float]:
    """
    Load GMM logic tree configuration
    for the specified IM and tectonic type.

    Parameters
    ----------
    gmm_lt : constants.GMMLogicTree
        Logic tree
    tect_type : constants.TectType
        Tectonic type of interest
    im : str
        IM of interest

    Returns
    -------
    dict
        Dictionary mapping model names to their weights for the given IM and tectonic type
    """
    logic_tree_config_ffp = constants.GMM_LT_CONFIG_MAPPING[gmm_lt]
    with logic_tree_config_ffp.open("r") as f:
        logic_tree_config = yaml.safe_load(f)

    if im not in logic_tree_config:
        return None

    return logic_tree_config[im][tect_type.name]


def prepare_model_inputs(
    model: constants.GMM,
    oq_model: gsim.base.GMPE,
    rupture_df: pd.DataFrame,
    im: str,
) -> pd.DataFrame:
    """
    Prepares the rupture dataframe and IM for OpenQuake.
    Handles missing model specific properties.

    Checks that all required inputs for the specified model
    are present in the rupture dataframe.

    Parameters
    ----------
    model : constants.GMM
        Ground motion model identifier
    oq_model : gsim.base.GMPE
        Instantiated OpenQuake ground motion model
    rupture_df : pd.DataFrame
        DataFrame containing rupture parameters
    im : str
        Intensity measure

    Returns
    -------
    pd.DataFrame
        Updated rupture_df

    Notes
    -----
    This function handles special cases for different models,
    such as adding missing parameters or renaming columns.
    """
    # Use a copy
    rupture_df = rupture_df.copy()

    # ASK 14
    _handle_missing_property(
        rupture_df, model, "ASK_14", "vs30measured", value_factory=lambda: False
    )
    _handle_missing_property(
        rupture_df,
        model,
        "ASK_14",
        "width",
        value_factory=lambda: estimations.estimate_width_ASK14(
            rupture_df["dip"], rupture_df["mag"]
        ),
    )
    _handle_missing_property(rupture_df, model, "ASK_14", "ry0", col_to_rename="ry")

    _handle_missing_property(
        rupture_df, model, "BCH_16", "xvf", value_factory=lambda: 0
    )

    # Model specified estimation that cannot be done within OQ as paper does not specify
    # CB_14's width will always be estimated. Hence, by passing np.nan first then,
    # we know the updated width values are from the estimation
    _handle_missing_property(
        rupture_df, model, "CB_14", "width", value_factory=lambda: np.nan
    )

    _handle_missing_property(
        rupture_df, model, "CY_14", "vs30measured", value_factory=lambda: False
    )

    # GA_11
    _handle_missing_property(
        rupture_df,
        model,
        "GA_11",
        "width",
        value_factory=lambda: estimations.estimate_width_ASK14(
            rupture_df["dip"], rupture_df["mag"]
        ),
    )
    _handle_missing_property(rupture_df, model, "GA_11", "ry0", col_to_rename="ry")

    # Check all model requirements are met
    rupture_ctx_properties = set(rupture_df.columns.values)
    _confirm_all_properties_exist(
        model, im, "site", oq_model.REQUIRES_SITES_PARAMETERS, rupture_ctx_properties
    )
    _confirm_all_properties_exist(
        model,
        im,
        "rupture",
        oq_model.REQUIRES_RUPTURE_PARAMETERS,
        rupture_ctx_properties,
    )
    _confirm_all_properties_exist(
        model, im, "distance", oq_model.REQUIRES_DISTANCES, rupture_ctx_properties
    )

    return rupture_df


def get_oq_model(
    model: constants.GMM,
    tect_type: constants.TectType,
    epistemic_branch: constants.EpistemicBranch = constants.EpistemicBranch.CENTRAL,
    **kwargs: dict[str, Any],
) -> tuple[gsim.base.GMPE, list[oq_const.StdDev]]:
    """
    Get the OpenQuake GMM model for specified GMM and tectonic type.

    Parameters
    ----------
    model : constants.GMM
        Ground motion model
    tect_type : constants.TectType
        Tectonic type
    epistemic_branch : constants.EpistemicBranch
        Epistemic uncertainty branch to use for the model.
        Defaults to constants.EpistemicBranch.CENTRAL.
    **kwargs : dict
        Extra model-specific parameters passed to the model constructor

    Returns
    -------
    gsim.base.GMPE
        Instantiated OpenQuake ground motion model
    list
        Standard deviation types supported by the model

    Warnings
    --------
    UserWarning
        When using a non-volcanic model for tectonic type VOLCANIC

    Raises
    ------
    ValueError
        If the model's defined tectonic type doesn't match the specified tectonic type
        (unless tect_type is VOLCANIC)
    """
    # Get correct epistemic uncertainty branch, only applicable for backbone models
    # and the mappings needs to be defined in constants.GMM_EPISTEMIC_BRANCH_KWARGS_MAPPING
    if (
        epistemic_branch is not constants.EpistemicBranch.CENTRAL
        and (epis_mapping := constants.GMM_EPISTEMIC_BRANCH_KWARGS_MAPPING.get(model))
        is not None
    ):
        kwargs = {**kwargs, **epis_mapping[epistemic_branch]}

    oq_model = OQ_MODEL_MAPPING[model][tect_type](**kwargs)

    if (
        constants.OQ_TECT_TYPE_MAPPING[oq_model.DEFINED_FOR_TECTONIC_REGION_TYPE]
        != tect_type
    ):
        if tect_type == constants.TectType.VOLCANIC:
            warnings.warn(
                f"Using {constants.OQ_TECT_TYPE_MAPPING[oq_model.DEFINED_FOR_TECTONIC_REGION_TYPE]} type "
                f"model for {tect_type.name} tectonic type. Ensure this is on purpose!",
                UserWarning,
            )
        else:
            raise ValueError(
                f"Invalid model type for {tect_type.name}."
                f"Model expects {constants.OQ_TECT_TYPE_MAPPING[oq_model.DEFINED_FOR_TECTONIC_REGION_TYPE]} type."
            )

    # Model standard deviation types
    stddev_types = [
        std
        for std in constants.SPT_STD_DEVS
        if std in oq_model.DEFINED_FOR_STANDARD_DEVIATION_TYPES
    ]

    return oq_model, stddev_types


def _get_model_pSA_periods(model: gsim.base.GMPE) -> np.ndarray:  # noqa: N802
    """
    Get the pSA periods supported by the specified model.

    Parameters
    ----------
    model : gsim.base.GMPE
        OpenQuake GMMM

    Returns
    -------
    np.ndarray
        Array of periods supported by the model
    """
    return np.asarray(
        [
            im.period
            for im in (
                model.COEFFS.sa_coeffs.keys()
                if not isinstance(
                    model,
                    (
                        gsim.zhao_2006.ZhaoEtAl2006Asc,
                        gsim.zhao_2006.ZhaoEtAl2006SSlab,
                        gsim.zhao_2006.ZhaoEtAl2006SInter,
                    ),
                )
                else model.COEFFS_ASC.sa_coeffs.keys()
            )
        ]
    )


def _oq_run_EAS(  # noqa: N802
    model_type: constants.GMM,
    model: gsim.base.GMPE,
    rupture_ctx: contexts.RuptureContext,
    frequencies: Sequence[float],
    stddev_types: Sequence[oq_const.StdDev],
) -> pd.DataFrame:
    """
    Run OpenQuake GMM for EAS.

    Parameters
    ----------
    model_type : constants.GMM
        Ground motion model identifier
    model : gsim.base.GMPE
        Instantiated OpenQuake ground motion model
    rupture_ctx : contexts.RuptureContext
        OpenQuake rupture context containing site, distance, and rupture information
    frequencies : Sequence[float]
        Frequencies to calculate effective amplitude spectra for
    stddev_types : Sequence[oq_const.StdDev]
        Standard deviation types to calculate

    Returns
    -------
    pd.DataFrame
        DataFrame containing effective amplitude spectra results for all requested frequencies

    Raises
    ------
    ValueError
        If the model does not support effective amplitude spectrum
    """
    if imt.EAS not in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
        raise ValueError(
            f"Model {model_type.name} does not support EAS. Supported types are {model.DEFINED_FOR_INTENSITY_MEASURE_TYPES}"
        )

    # Get model frequencies
    results = []
    for frequency in frequencies:
        im = imt.EAS(frequency=frequency)
        try:
            result = _run_oq_model(model, rupture_ctx, im, stddev_types)
        except Exception as e:
            # Any other exceptions that we cannot handle.
            logging.exception(e)
            raise

        results.append(result)

    return pd.concat(results, axis=1)


def _oq_run_pSA(  # noqa: N802
    model_type: constants.GMM,
    model: gsim.base.GMPE,
    rupture_ctx: contexts.RuptureContext,
    periods: Sequence[float],
    stddev_types: Sequence[oq_const.StdDev],
) -> pd.DataFrame:
    """
    Run OpenQuake GMM for pSA.
    Handles extrapolation for periods larger than the model's maximum period.
    Handles interpolation for periods smaller than the model's minimum period using PGA.

    Parameters
    ----------
    model_type : constants.GMM
        Ground motion model identifier
    model : gsim.base.GMPE
        Instantiated OpenQuake ground motion model
    rupture_ctx : contexts.RuptureContext
        OpenQuake rupture context containing site, distance, and rupture information
    periods : Sequence[float]
        Periods to calculate spectral acceleration for
    stddev_types : Sequence[oq_const.StdDev]
        Standard deviation types to calculate

    Returns
    -------
    pd.DataFrame
        DataFrame containing spectral acceleration results for all requested periods

    Warnings
    --------
    UserWarning
        When extrapolating pSA values

    Raises
    ------
    ValueError
        If the model does not support spectral acceleration
        If pSA period interpolation fails

    """
    if imt.SA not in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
        raise ValueError(
            f"Model {model_type.name} does not support pSA. Supported types are {model.DEFINED_FOR_INTENSITY_MEASURE_TYPES}"
        )

    # Get model periods
    model_periods = _get_model_pSA_periods(model)
    max_model_period = max(model_periods)

    results = []
    for period in periods:
        im = imt.SA(period=min(period, max_model_period))
        try:
            result = _run_oq_model(model, rupture_ctx, im, stddev_types)
        except KeyError as ke:
            cause = ke.args[0]
            # Ensure the KeyError is about missing pSA's period
            if (
                isinstance(cause, imt.IMT)
                and str(cause).startswith("SA")
                and cause.period > 0.0
                and period < min(model_periods)
            ):
                # Period is smaller than model's supported min_period E.g., ZA_06
                # Interpolate between PGA(0.0) and model's min_period
                low_result = _run_oq_model(
                    model,
                    rupture_ctx,
                    imt.PGA(),
                    stddev_types,
                )
                high_period = model_periods[period <= model_periods][0]
                high_result = _run_oq_model(
                    model,
                    rupture_ctx,
                    imt.SA(period=high_period),
                    stddev_types,
                )
                result = estimations.interpolate_with_pga(
                    period, high_period, low_result, high_result
                )
            else:
                # KeyError that we cannot handle
                logging.exception(ke)
                raise
        except Exception as e:
            # Any other exceptions that we cannot handle
            logging.exception(e)
            raise

        # extrapolate pSA value up based on maximum available period
        if period > max_model_period:
            warnings.warn(
                f"Extrapolating pSA({period}) based on maximum "
                f"available period {max_model_period} for model {model_type.name}.",
                UserWarning,
            )
            result.loc[:, result.columns.str.endswith("mean")] += 2 * np.log(
                max_model_period / period
            )
            # Updating the period from max_period to the given period
            # e.g with ZA_06, replace 5.0 to period > 5.0
            result.columns = result.columns.str.replace(
                str(max_model_period), str(period), regex=False
            )
        results.append(result)

    return pd.concat(results, axis=1)


def _run_oq_model(
    oq_model: gsim.base.GMPE,
    ctx: contexts.RuptureContext,
    im: imt.IMT,
    stddev_types: Sequence[oq_const.StdDev],
) -> pd.DataFrame:
    """
    Call OpenQuake to compute mean and standard deviations
    for the given GMM model and rupture context.

    Converts OpenQuake results to standard output format.

    Parameters
    ----------
    oq_model : gsim.base.GMPE
        OpenQuake ground motion model
    ctx : contexts.RuptureContext
        OpenQuake rupture context containing site, distance, and rupture information
    im : imt.IMT
        Intensity measure type
    stddev_types : Sequence[oq_const.StdDev]
        Standard deviation types to calculate

    Returns
    -------
    pd.DataFrame
        DataFrame containing mean and standard deviation results
    """
    # Run model. Returns array with
    # mean, std_total, std_inter and std_intra.
    # Order of the standard deviation can vary.
    results = contexts.get_mean_stds(oq_model, ctx, [im])
    mean_stddev_dict = {f"{_convert_from_oq_im(im)}_mean": results[0][0]}
    for idx, std_dev in enumerate(stddev_types):
        # std_devs index are between 1 and 3
        mean_stddev_dict[f"{_convert_from_oq_im(im)}_std_{std_dev.split()[0]}"] = (
            results[idx + 1][0]
        )

    return pd.DataFrame(mean_stddev_dict)


def _confirm_all_properties_exist(
    model: constants.GMM,
    im: str,
    type: str,
    params_required: set,
    params_present: set,
) -> None:
    """
    Check if all required columns are present in the rupture dataframe.

    Parameters
    ----------
    model : constants.GMM
        Ground motion model being checked
    im : str
        Intensity measure associated with the model
    type : str
        Property type ('site', 'rupture', or 'distance')
    params_required : set
        Set of parameter names required by the model
    params_present : set
        Set of parameter names present in the rupture dataframe

    Raises
    ------
    ValueError
        If required parameters are missing
    """
    extra_params = set(params_required - params_present)
    if len(extra_params) > 0:
        raise ValueError(
            f"Unknown {type} property: {extra_params} required by {model.name} {im}"
            f"\nPlease review {__file__} to handle missing properties."
        )


def _handle_missing_property(
    rupture_df: pd.DataFrame,
    model: constants.GMM,
    model_name: str,
    col_missing: str,
    value_factory: Callable[[], Any] = lambda: None,
    col_to_rename: str | None = None,
) -> None:
    """
    Handle missing properties in the rupture dataframe for
    the specified model type.

    Parameters
    ----------
    rupture_df : pd.DataFrame
        Rupture dataframe to modify
    model : constants.GMM
        Current model being used
    model_name : str
        Name of the model to check against
    col_missing : str
        Column name that is missing in the rupture dataframe
    value_factory : Callable, optional
        Function that generates a default value when the column is missing,
        by default lambda: None
    col_to_rename : str, optional
        Column name to rename to col_missing, by default None
    """
    if model.name == model_name:
        if col_missing not in rupture_df and col_to_rename is not None:
            rupture_df.rename(
                {col_to_rename: col_missing}, axis="columns", inplace=True
            )
        elif col_missing not in rupture_df:
            rupture_df[col_missing] = value_factory()


def _convert_to_oq_im(im: str) -> str:
    """
    Convert IM name to OpenQuake's IM name.

    Only supports conversion for
    Ds575 and Ds595 to RSD575 and RSD595.

    Parameters
    ----------
    im : str
        IM name

    Returns
    -------
    str
        Converted IM name
    """
    if im in ("Ds575", "Ds595"):
        im = im.replace("Ds", "RSD")

    return im


def _convert_from_oq_im(im: imt.IMT) -> str:
    """
    Convert OpenQuake's IM name.

    Parameters
    ----------
    im : imt.IMT
        OpenQuake IM object

    Returns
    -------
    str
        Converted IM label

    Notes
    -----
    Examples of conversions:
    - EAS(frequency) → EAS_frequency
    - SA(period) → pSA_period
    - RSD575 → Ds575
    - RSD595 → Ds595
    """
    if im[0].startswith("EAS"):
        period = im[1]
        frequency = 1 / period
        return f"EAS_{frequency}"

    imt_tuple = imt.imt2tup(im.string)
    if len(imt_tuple) == 1:
        return (
            imt_tuple[0].replace("RSD", "Ds")
            if imt_tuple[0].startswith("RSD")
            else imt_tuple[0]
        )

    return f"pSA_{imt_tuple[-1]}"
