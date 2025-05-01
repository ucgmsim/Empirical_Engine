import logging
import warnings
from collections.abc import Callable, Sequence
from functools import partial
from typing import Union

import numpy as np
import pandas as pd
from openquake.hazardlib import const as oq_const
from openquake.hazardlib import contexts, gsim, imt

from . import constants, estimations, utils


def _oq_model(model: gsim.base.GMPE, **kwargs):
    """Partial function to simplify model instanstiation
    model: gsim.base.GMPE
    kwargs: pass extra (model specific) parameters to models
    E.g:
        region=NZL for models end with _NZ
        estimate_width=True for CB_14 to estimate width
    """
    return model(**kwargs)

OQ_MODEL_MAPPING = {
    constants.GMM.Br_10: {
        constants.TectType.ACTIVE_SHALLOW: gsim.bradley_2013.Bradley2013
    },
    constants.GMM.AS_16: {
        constants.TectType.ACTIVE_SHALLOW: gsim.afshari_stewart_2016.AfshariStewart2016
    },
    constants.GMM.A_18: {
        constants.TectType.SUBDUCTION_SLAB: gsim.abrahamson_2018.AbrahamsonEtAl2018SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_2018.AbrahamsonEtAl2018SInter,
    },
    # OQ's CB_08 includes the CB_10's CAV
    constants.GMM.CB_10: {
        constants.TectType.ACTIVE_SHALLOW: gsim.campbell_bozorgnia_2008.CampbellBozorgnia2008
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
        constants.TectType.SUBDUCTION_SLAB: partial(
            _oq_model, model=gsim.kuehn_2020.KuehnEtAl2020SSlab, region="NZL"
        ),
        constants.TectType.SUBDUCTION_INTERFACE: partial(
            _oq_model, model=gsim.kuehn_2020.KuehnEtAl2020SInter, region="NZL"
        ),
    },
    constants.GMM.AG_20_NZ: {
        constants.TectType.SUBDUCTION_SLAB: partial(
            _oq_model,
            model=gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
            region="NZL",
        ),
        constants.TectType.SUBDUCTION_INTERFACE: partial(
            _oq_model,
            model=gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
            region="NZL",
        ),
    },
    constants.GMM.S_22: {
        constants.TectType.ACTIVE_SHALLOW: gsim.nz22.stafford_2022.Stafford2022
    },
    constants.GMM.A_22: {
        constants.TectType.ACTIVE_SHALLOW: gsim.nz22.atkinson_2022.Atkinson2022Crust,
        constants.TectType.SUBDUCTION_SLAB: gsim.nz22.atkinson_2022.Atkinson2022SSlab,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.nz22.atkinson_2022.Atkinson2022SInter,
    },
    constants.GMM.ASK_14: {
        constants.TectType.ACTIVE_SHALLOW: gsim.abrahamson_2014.AbrahamsonEtAl2014
    },
    constants.GMM.CY_14: {
        constants.TectType.ACTIVE_SHALLOW: gsim.chiou_youngs_2014.ChiouYoungs2014
    },
    constants.GMM.CB_14: {
        constants.TectType.ACTIVE_SHALLOW: partial(
            _oq_model,
            model=gsim.campbell_bozorgnia_2014.CampbellBozorgnia2014,
            estimate_width=True,
        )
    },
    constants.GMM.BSSA_14: {
        constants.TectType.ACTIVE_SHALLOW: gsim.boore_2014.BooreEtAl2014
    },
    constants.GMM.Br_13: {
        constants.TectType.ACTIVE_SHALLOW: gsim.bradley_2013.Bradley2013
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
        constants.TectType.SUBDUCTION_SLAB: gsim.nz22.nz_nshm2022_parker.NZNSHM2022_ParkerEtAl2020SInterB,
        constants.TectType.SUBDUCTION_INTERFACE: gsim.nz22.nz_nshm2022_parker.NZNSHM2022_ParkerEtAl2020SInterB,
    },
    constants.GMM.GA_11: {
        constants.TectType.ACTIVE_SHALLOW: gsim.gulerce_abrahamson_2011.GulerceAbrahamson2011
    },
}

def run_gmm(
    model: constants.GMM,
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[Union[int, float]] = None,
    **kwargs,
):
    """Run an openquake model with dataframe
    model_type: GMM
        OQ model
    tect_type: TectType
        One of the tectonic types from
        ACTIVE_SHALLOW, SUBDUCTION_SLAB and SUBDUCTION_INTERFACE
    rupture_df: Rupture DF
        Columns for properties. E.g., vs30, z1pt0, rrup, rjb, mag, rake, dip....
        Rows be the separate site-fault pairs
    im: string
        intensity measure
    periods: Sequence[Union[int, float]]
        for spectral acceleration, openquake tables automatically
        interpolate values between specified values, fails if outside range
    kwargs: pass extra (model specific) parameters to models
    """
    # Get the OQ model
    oq_model, stddev_types = get_oq_model(model, tect_type, **kwargs)

    # Prepare inputs
    rupture_df, im = prepare_model_inputs(
        model,
        oq_model,
        rupture_df,
        im,
    )

    # Convert Z1.0 from km to m
    assert np.all(rupture_df["z1pt0"] < 10), "Z1.0 must be in kilometres"
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
            raise ValueError(
                "Periods must be specified for pSA."
            )

        return _oq_run_pSA(
            model,
            oq_model,
            rupture_ctx,
            periods,
            stddev_types,
        )
    else:
        # Check that the IM is supported
        # Supported IMs are specified as functions per model, hence the getattr
        imc = getattr(imt, im)
        if imc not in oq_model.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
            raise ValueError(
                f"Model {model.name} does not support {im}. Supported types are {oq_model.DEFINED_FOR_INTENSITY_MEASURE_TYPES}"
            )
        return _run_oq_model(oq_model, rupture_ctx, imc(), stddev_types)



def prepare_model_inputs(
    model: constants.GMM,
    oq_model: gsim.base.GMPE,
    rupture_df: pd.DataFrame,
    im: str,
):
    """
    Handle exceptions for the given model and rupture_df, and returns an updated set of
    (model, rupture_df, im)

    Parameters
    ----------
    model_type: GMM
        OQ model
    tect_type: TectType
        One of the tectonic types from ACTIVE_SHALLOW, SUBDUCTION_SLAB and SUBDUCTION_INTERFACE
    rupture_df: pd.DataFrame
        Columns for properties. E.g., vs30, z1pt0, rrup, rjb, mag, rake, dip....
    im: str
        intensity measure
    kwargs: pass extra (model specific) parameters to models


    Returns
    -------
    model: gsim.base.GMPE
        OQ_MODEL for a given model_type and tect_type
    rupture_df: pd.DataFrame
        updated rupture_df
    im: str
        intensity measure (updated if necessary)
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

    _handle_missing_property(rupture_df, model, "BCH_16", "xvf", value_factory=lambda: 0)

    _handle_missing_property(
        rupture_df, model, "Br_10", "vs30measured", value_factory=lambda: False
    )

    # Model specified estimation that cannot be done within OQ as paper does not specify
    # CB_14's width will always be estimated. Hence, by passing np.nan first then,
    # we know the updated width values are from the estimation
    _handle_missing_property(rupture_df, model, "CB_14", "width", value_factory=lambda: np.nan)

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

    # Rename IMs to OQ's term
    if im in ("Ds575", "Ds595"):
        im = im.replace("Ds", "RSD")

    # Check all model requirements are met
    rupture_ctx_properties = set(rupture_df.columns.values)
    _confirm_all_properties_exist(
        model, im, "site", oq_model.REQUIRES_SITES_PARAMETERS, rupture_ctx_properties
    )
    _confirm_all_properties_exist(
        model, im, "rupture", oq_model.REQUIRES_RUPTURE_PARAMETERS, rupture_ctx_properties
    )
    _confirm_all_properties_exist(
        model, im, "distance", oq_model.REQUIRES_DISTANCES, rupture_ctx_properties
    )

    return rupture_df, im

def get_oq_model(
    model_type: constants.GMM,
    tect_type: constants.TectType,
    **kwargs,
):
    model = OQ_MODEL_MAPPING[model_type][tect_type](**kwargs)

    # Sanity check
    assert (
        constants.OQ_TECT_TYPE_MAPPING[model.DEFINED_FOR_TECTONIC_REGION_TYPE]
        == tect_type
    )

    # Model standard deviation types
    stddev_types = [
        std for std in constants.SPT_STD_DEVS if std in model.DEFINED_FOR_STANDARD_DEVIATION_TYPES
    ]

    return model, stddev_types


def _get_model_pSA_periods(model: gsim.base.GMPE):
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


def _oq_run_pSA(
    model_type: constants.GMM,
    model: gsim.base.GMPE,
    rupture_ctx: contexts.RuptureContext,
    periods: Sequence[float],
    stddev_types: Sequence[oq_const.StdDev],
):
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
            ):
                # Period is smaller than model's supported min_period E.g., ZA_06
                # Interpolate between PGA(0.0) and model's min_period
                assert period < min(model_periods)
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
                f"Extrapolating pSA({period}) based on maximum available period {max_model_period} for model {model_type.name}.",
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
):
    """
    Calculate mean and standard deviations using 
    the given model and rupture context.

    model: gsim.base.GMPE
        OQ models we use are subclass of gsim.base.GMPE
    ctx: contexts.RuptureContext
        OQ RuptureContext that contains the following information
        - Site
        - Distance
        - Rupture
    im: imt.IMT
    stddev_types: Sequence[const.StdDev]
    """
    # Run model. Returns array with
    # mean, std_total, std_inter and std_intra.
    # Order of the standard deviation can vary.
    results = contexts.get_mean_stds(oq_model, ctx, [im])

    mean_stddev_dict = {f"{utils.convert_im_label(im)}_mean": results[0][0]}
    for idx, std_dev in enumerate(stddev_types):
        # std_devs index are between 1 and 3 
        mean_stddev_dict[f"{utils.convert_im_label(im)}_std_{std_dev.split()[0]}"] = (
            results[idx + 1][0]
        )

    return pd.DataFrame(mean_stddev_dict)


def _confirm_all_properties_exist(
    model: constants.GMM,
    im: str,
    type: str,
    params_required: set,
    params_present: set,
):
    """
    Check if all required columns are present. If not, raise an error.

    Parameters
    ----------
    model_type: constants.GMM
        The model type being checked.
    im: str
        The intensity measure associated with the model.
    type: str
        site, rupture or distance
    params_required: set
        set of required columns
    params_present: set
        set of columns present in the dataframe
    """
    extra_params = set(params_required - params_present)
    if len(extra_params) > 0:
        raise ValueError(
            f"Unknown {type} property: {extra_params} required by {model.name} {im}"
            f"\nPlease review {__file__} to handle missing properties."
        )


def _handle_missing_property(
    rupture_df: pd.DataFrame,
    model_type: constants.GMM,
    model_type_name: str,
    col_missing: str,
    value_factory: Callable = lambda: None,
    col_to_rename: str = None,
):
    """
    If the specific model requires any additional columns that are not in the rupture_df
    we can manually assign a value, work out from the existing columns or raise an error

    Parameters
    ----------
    model_type_name: str
        model type name
    col_missing : str
        column name that is missing in the rupture_df
    value_factory: Callable
        default value generator if the column is missing but no
        column is specified to rename from.
    col_to_rename: str
        column name to rename to col_missing
    """
    if model_type.name == model_type_name:
        if col_missing not in rupture_df and col_to_rename is not None:
            rupture_df.rename(
                {col_to_rename: col_missing}, axis="columns", inplace=True
            )
        elif col_missing not in rupture_df:
            rupture_df[col_missing] = value_factory()

