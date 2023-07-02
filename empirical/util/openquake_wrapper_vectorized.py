"""Wrapper for openquake vectorized models."""
import logging
from typing import Sequence, Union, Dict
from functools import partial

import numpy as np
import pandas as pd
from scipy import interpolate
from openquake.hazardlib import const, imt, gsim, contexts

from empirical.util import estimations
from empirical.util.classdef import TectType, GMM


def OQ_model(model, **kwargs):
    """Partial function to simplify model instanstiation
    model: gsim.base.GMPE
    kwargs: pass extra (model specific) parameters to models
    E.g:
        region=NZL for models end with _NZ
        estimate_width=True for CB_14 to estimate width
    """
    return model(**kwargs)


OQ_MODELS = {
    GMM.Br_10: {TectType.ACTIVE_SHALLOW: gsim.bradley_2013.Bradley2013},
    GMM.AS_16: {TectType.ACTIVE_SHALLOW: gsim.afshari_stewart_2016.AfshariStewart2016},
    GMM.A_18: {
        TectType.SUBDUCTION_SLAB: gsim.abrahamson_2018.AbrahamsonEtAl2018SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_2018.AbrahamsonEtAl2018SInter,
    },
    # OQ's CB_08 includes the CB_10's CAV
    GMM.CB_10: {
        TectType.ACTIVE_SHALLOW: gsim.campbell_bozorgnia_2008.CampbellBozorgnia2008
    },
    GMM.BCH_16: {
        TectType.SUBDUCTION_SLAB: gsim.bchydro_2016_epistemic.BCHydroESHM20SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.bchydro_2016_epistemic.BCHydroESHM20SInter,
    },
    GMM.ZA_06: {
        TectType.ACTIVE_SHALLOW: gsim.zhao_2006.ZhaoEtAl2006Asc,
        TectType.SUBDUCTION_SLAB: gsim.zhao_2006.ZhaoEtAl2006SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.zhao_2006.ZhaoEtAl2006SInter,
    },
    GMM.K_20_NZ: {
        TectType.SUBDUCTION_SLAB: partial(
            OQ_model, model=gsim.kuehn_2020.KuehnEtAl2020SSlab, region="NZL"
        ),
        TectType.SUBDUCTION_INTERFACE: partial(
            OQ_model, model=gsim.kuehn_2020.KuehnEtAl2020SInter, region="NZL"
        ),
    },
    GMM.AG_20_NZ: {
        TectType.SUBDUCTION_SLAB: partial(
            OQ_model,
            model=gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
            region="NZL",
        ),
        TectType.SUBDUCTION_INTERFACE: partial(
            OQ_model,
            model=gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
            region="NZL",
        ),
    },
    GMM.S_22: {TectType.ACTIVE_SHALLOW: gsim.stafford_2022.Stafford2022},
    GMM.A_22: {
        TectType.ACTIVE_SHALLOW: gsim.atkinson_2022.Atkinson2022Crust,
        TectType.SUBDUCTION_SLAB: gsim.atkinson_2022.Atkinson2022SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.atkinson_2022.Atkinson2022SInter,
    },
    GMM.ASK_14: {TectType.ACTIVE_SHALLOW: gsim.abrahamson_2014.AbrahamsonEtAl2014},
    GMM.CY_14: {TectType.ACTIVE_SHALLOW: gsim.chiou_youngs_2014.ChiouYoungs2014},
    GMM.CB_14: {
        TectType.ACTIVE_SHALLOW: partial(
            OQ_model,
            model=gsim.campbell_bozorgnia_2014.CampbellBozorgnia2014,
            estimate_width=True,
        )
    },
    GMM.BSSA_14: {TectType.ACTIVE_SHALLOW: gsim.boore_2014.BooreEtAl2014},
    GMM.Br_13: {TectType.ACTIVE_SHALLOW: gsim.bradley_2013.Bradley2013},
    GMM.AG_20: {
        TectType.SUBDUCTION_SLAB: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
    },
    GMM.P_20: {
        TectType.SUBDUCTION_SLAB: gsim.parker_2020.ParkerEtAl2020SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.parker_2020.ParkerEtAl2020SInter,
    },
    GMM.P_21: {
        TectType.SUBDUCTION_SLAB: gsim.parker_2021.ParkerEtAl2021SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.parker_2021.ParkerEtAl2021SInter,
    },
    GMM.K_20: {
        TectType.SUBDUCTION_SLAB: gsim.kuehn_2020.KuehnEtAl2020SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.kuehn_2020.KuehnEtAl2020SInter,
    },
    GMM.P_21: {
        TectType.SUBDUCTION_SLAB: gsim.parker_2021.ParkerEtAl2021SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.parker_2021.ParkerEtAl2021SInter,
    },
}

SPT_STD_DEVS = [const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT]


def convert_im_label(im: imt.IMT):
    """Convert OQ's IM term into the internal term.
    E.g:
        pSA_period (OQ uses SA(period))
        Ds575 (OQ uses RSD575)
        Ds595 (OQ uses RSD595)
    im: imt.IMT
    """
    imt_tuple = imt.imt2tup(im.string)
    if len(imt_tuple) == 1:
        return (
            imt_tuple[0].replace("RSD", "Ds")
            if imt_tuple[0].startswith("RSD")
            else imt_tuple[0]
        )

    return f"pSA_{imt_tuple[-1]}"


def oq_mean_stddevs(
    model: gsim.base.GMPE,
    ctx: contexts.RuptureContext,
    im: imt.IMT,
    stddev_types: Sequence[const.StdDev],
):
    """Calculate mean and standard deviations given openquake input structures.
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
    # contexts.get_mean_stds returns ndarray, size of 4
    # mean, std_total, std_inter and std_intra
    # std_devs order may vary
    results = contexts.get_mean_stds(model, ctx, [im])

    mean_stddev_dict = {f"{convert_im_label(im)}_mean": results[0][0]}
    for idx, std_dev in enumerate(stddev_types):
        # std_devs are index between 1 and 3 from results
        mean_stddev_dict[f"{convert_im_label(im)}_std_{std_dev.split()[0]}"] = results[
            idx + 1
        ][0]

    return pd.DataFrame(mean_stddev_dict)


def interpolate_with_pga(
    period: Union[float, int],
    high_period: float,
    low_y: pd.DataFrame,
    high_y: pd.DataFrame,
):
    """Use interpolation to find the value of new points at the given period
    which is between 0.0(PGA) and the model's minimum period.

    period: Union[float, int]
        target period for interpolation
    high_period: float
        also known as a minimum period from the given model
        which to be used in x coordinates range which looks like
        [0.0 (PGA), high_period]
    low_y: pd.DataFrame
        DataFrame that contains GMM computational results at period = 0.0
    high_y: pd.DataFrame
        DataFrame that contains GMM computational results
        at period = model's minimum period
    """
    x = [0.0, high_period]
    # each subarray represents values at period=0.0 and period=high_period
    # E.g., two site/rupture data would look something like
    # mean_y = np.array([[a,b], [c,d]])
    # where a,c are at period=0.0 and b,d are at period=high_period
    mean_y = np.concatenate(
        (
            np.exp(low_y.loc[:, low_y.columns.str.endswith("mean")].to_numpy()),
            np.exp(high_y.loc[:, high_y.columns.str.endswith("mean")].to_numpy()),
        ),
        axis=1,
    )
    sigma_total_y = np.concatenate(
        (
            low_y.loc[:, low_y.columns.str.endswith("std_Total")].to_numpy(),
            high_y.loc[:, high_y.columns.str.endswith("std_Total")].to_numpy(),
        ),
        axis=1,
    )
    sigma_inter_y = np.concatenate(
        (
            low_y.loc[:, low_y.columns.str.endswith("std_Inter")].to_numpy(),
            high_y.loc[:, high_y.columns.str.endswith("std_Inter")].to_numpy(),
        ),
        axis=1,
    )
    sigma_intra_y = np.concatenate(
        (
            low_y.loc[:, low_y.columns.str.endswith("std_Intra")].to_numpy(),
            high_y.loc[:, high_y.columns.str.endswith("std_Intra")].to_numpy(),
        ),
        axis=1,
    )

    return pd.DataFrame(
        {
            f"pSA_{period}_mean": np.log(interpolate.interp1d(x, mean_y)(period)),
            f"pSA_{period}_std_Total": interpolate.interp1d(x, sigma_total_y)(period),
            f"pSA_{period}_std_Inter": interpolate.interp1d(x, sigma_inter_y)(period),
            f"pSA_{period}_std_Intra": interpolate.interp1d(x, sigma_intra_y)(period),
        }
    )


def oq_run(
    model_type: GMM,
    tect_type: TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[Union[int, float]] = None,
    meta_config: Dict = None,
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
        But Site information must be identical across the rows,
        only the faults can be different.
    im: string
        intensity measure
    periods: Sequence[Union[int, float]]
        for spectral acceleration, openquake tables automatically
        interpolate values between specified values, fails if outside range
    meta_config: Dict
        A dictionary contains models and its weight
    kwargs: pass extra (model specific) parameters to models
    """

    if model_type.name == "META":
        meta_results = pd.Series(
            [
                oq_run(GMM[model], tect_type, rupture_df, im, periods)
                for model in meta_config.keys()
            ]
        )

        # Compute the weighted average
        return np.sum(meta_results * pd.Series(meta_config.values()))

    model = OQ_MODELS[model_type][tect_type](**kwargs)

    # Check the given tect_type with its model's tect type
    trt = model.DEFINED_FOR_TECTONIC_REGION_TYPE
    if trt == const.TRT.SUBDUCTION_INTERFACE:
        assert (
            tect_type == TectType.SUBDUCTION_INTERFACE
        ), "Tect Type must be SUBDUCTION_INTERFACE"
    elif trt == const.TRT.SUBDUCTION_INTRASLAB:
        assert (
            tect_type == TectType.SUBDUCTION_SLAB
        ), "Tect Type must be SUBDUCTION_SLAB"
    elif trt == const.TRT.ACTIVE_SHALLOW_CRUST:
        assert tect_type == TectType.ACTIVE_SHALLOW, "Tect Type must be ACTIVE_SHALLOW"
    else:
        raise ValueError("unknown tectonic region: " + trt)

    stddev_types = [
        std for std in SPT_STD_DEVS if std in model.DEFINED_FOR_STANDARD_DEVIATION_TYPES
    ]

    # Make a copy in case the original rupture_df used with other functions
    rupture_df = rupture_df.copy()

    # Model specified estimation that cannot be done within OQ as paper does not specify
    # CB_14's width will always be estimated. Hence, by passing np.nan first then,
    # we know the updated width values are from the estimation
    if model_type.name == "CB_14" and "width" not in rupture_df:
        rupture_df["width"] = np.nan

    elif model_type.name == "ASK_14":
        if "width" not in rupture_df:
            rupture_df["width"] = estimations.estimate_width_ASK14(
                rupture_df["dip"], rupture_df["mag"]
            )
        if "ry0" not in rupture_df:
            rupture_df.rename({"ry": "ry0"}, axis="columns", inplace=True)

    elif model_type.name == "BCH_16":
        # Equivalent to classdef.Site's backarc and the default value we set is False
        # Within OQ, they use either 0 or 1
        if "xvf" not in rupture_df:
            rupture_df["xvf"] = 0
        # abrahamson_2015 uses dists = rrup for SUBDUCTION_INTERFACE
        # or dists = rhypo for SUBDUCTION_SLAB. Hence, I believe we can use rrup
        # Also, internal bc_hydro_2016 script uses rrup
        if "rhypo" not in rupture_df:
            rupture_df["rhypo"] = rupture_df["rrup"]

    # Rename to OQ's term
    if im in ("Ds575", "Ds595"):
        im = im.replace("Ds", "RSD")

    # Check if df contains what model requires
    rupture_ctx_properties = set(rupture_df.columns.values)
    extra_site_parameters = set(model.REQUIRES_SITES_PARAMETERS).difference(
        rupture_ctx_properties
    )
    if len(extra_site_parameters) > 0:
        raise ValueError("unknown site property: " + extra_site_parameters)

    extra_rup_properties = set(model.REQUIRES_RUPTURE_PARAMETERS).difference(
        rupture_ctx_properties
    )
    if len(extra_rup_properties) > 0:
        raise ValueError("unknown rupture property: " + " ".join(extra_rup_properties))

    extra_dist_properties = set(model.REQUIRES_DISTANCES).difference(
        rupture_ctx_properties
    )
    if len(extra_dist_properties) > 0:
        raise ValueError(
            "unknown distance property: " + " ".join(extra_dist_properties)
        )

    # Convert z1pt0 from km to m
    rupture_df["z1pt0"] *= 1000
    #print("z1 = %s" %rupture_df["z1pt0"])
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

    if periods is not None:
        assert imt.SA in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        # use sorted instead of max for full list
        avail_periods = np.asarray(
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
        max_period = max(avail_periods)
        if not hasattr(periods, "__len__"):
            periods = [periods]
        results = []
        for period in periods:
            im = imt.SA(period=min(period, max_period))
            try:
                result = oq_mean_stddevs(model, rupture_ctx, im, stddev_types)
            except KeyError as ke:
                cause = ke.args[0]
                # To make sure the KeyError is about missing pSA's period
                if (
                    isinstance(cause, imt.IMT)
                    and str(cause).startswith("SA")
                    and cause.period > 0.0
                ):
                    # Period is smaller than model's supported min_period E.g., ZA_06
                    # Interpolate between PGA(0.0) and model's min_period
                    low_result = oq_mean_stddevs(
                        model, rupture_ctx, imt.PGA(), stddev_types
                    )
                    high_period = avail_periods[period <= avail_periods][0]
                    high_result = oq_mean_stddevs(
                        model,
                        rupture_ctx,
                        imt.SA(period=high_period),
                        stddev_types,
                    )

                    result = interpolate_with_pga(
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
            if period > max_period:
                result.loc[:, result.columns.str.endswith("mean")] += 2 * np.log(
                    max_period / period
                )
                # Updating the period from max_period to the given period
                # e.g with ZA_06, replace 5.0 to period > 5.0
                result.columns = result.columns.str.replace(
                    str(max_period), str(period), regex=False
                )
            results.append(result)

        return pd.concat(results, axis=1)
    else:
        imc = getattr(imt, im)
        assert imc in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        return oq_mean_stddevs(model, rupture_ctx, imc(), stddev_types)
