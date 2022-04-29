"""Wrapper for openquake vectorized models."""
from typing import Sequence, Union
from enum import Enum

import numpy as np
import pandas as pd
from scipy import interpolate
from openquake.hazardlib import const, imt, gsim, contexts

from empirical.util.classdef import TectType, GMM


OQ_MODELS = {
    GMM.Br_10: {TectType.ACTIVE_SHALLOW: gsim.bradley_2013.Bradley2013},
    GMM.ASK_14: {TectType.ACTIVE_SHALLOW: gsim.abrahamson_2014.AbrahamsonEtAl2014},
    GMM.CB_14: {
        TectType.ACTIVE_SHALLOW: gsim.campbell_bozorgnia_2014.CampbellBozorgnia2014
    },
    GMM.BSSA_14: {TectType.ACTIVE_SHALLOW: gsim.boore_2014.BooreEtAl2014},
    GMM.CY_14: {TectType.ACTIVE_SHALLOW: gsim.chiou_youngs_2014.ChiouYoungs2014},
    GMM.ZA_06: {
        TectType.ACTIVE_SHALLOW: gsim.zhao_2006.ZhaoEtAl2006Asc,
        TectType.SUBDUCTION_SLAB: gsim.zhao_2006.ZhaoEtAl2006SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.zhao_2006.ZhaoEtAl2006SInter,
    },
    GMM.P_20: {
        TectType.SUBDUCTION_SLAB: gsim.parker_2020.ParkerEtAl2020SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.parker_2020.ParkerEtAl2020SInter,
    },
    GMM.K_20: {
        TectType.SUBDUCTION_SLAB: gsim.kuehn_2020.KuehnEtAl2020SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.kuehn_2020.KuehnEtAl2020SInter,
    },
    GMM.K_20_NZ: {
        TectType.SUBDUCTION_SLAB: gsim.kuehn_2020.KuehnEtAl2020SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.kuehn_2020.KuehnEtAl2020SInter,
    },
    GMM.AG_20: {
        TectType.SUBDUCTION_SLAB: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
    },
    GMM.AG_20_NZ: {
        TectType.SUBDUCTION_SLAB: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
        TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
    },
}

SPT_STD_DEVS = [const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT]


def convert_im_label(im: imt.IMT):
    """Convert OQ's SA(period) to the internal naming, pSA_period
    im: imt.IMT
    """
    imt_tuple = imt.imt2tup(im.string)
    return im if len(imt_tuple) == 1 else f"pSA_{imt_tuple[-1]}"


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


def interpolate_to_closest(
    period: Union[float, int], x: np.ndarray, low_y: pd.DataFrame, high_y: pd.DataFrame
):
    """Use interpolation to find the value of new points at the given period.

    period: Union[float, int]
        target period for interpolation
    x: np.ndarray
        x coordinates range which looks like
        [0.0 (PGA), model's minimum period]
    low_y: pd.DataFrame
        DataFrame that contains GMM computational results at period = 0.0
    high_y: pd.DataFrame
        DataFrame that contains GMM computational results
        at period = model's minimum period
    """

    # Create new DFs by columns from low_y and high_y
    mean_df = pd.DataFrame().assign(
        low=np.exp(low_y.loc[:, low_y.columns.str.endswith("mean")].iloc[:, 0]),
        high=np.exp(high_y.loc[:, high_y.columns.str.endswith("mean")].iloc[:, 0]),
    )
    sigma_total_df = pd.DataFrame().assign(
        low=np.exp(low_y.loc[:, low_y.columns.str.endswith("std_Total")].iloc[:, 0]),
        high=np.exp(high_y.loc[:, high_y.columns.str.endswith("std_Total")].iloc[:, 0]),
    )
    sigma_inter_df = pd.DataFrame().assign(
        low=np.exp(low_y.loc[:, low_y.columns.str.endswith("std_Inter")].iloc[:, 0]),
        high=np.exp(high_y.loc[:, high_y.columns.str.endswith("std_Inter")].iloc[:, 0]),
    )
    sigma_intra_df = pd.DataFrame().assign(
        low=np.exp(low_y.loc[:, low_y.columns.str.endswith("std_Intra")].iloc[:, 0]),
        high=np.exp(high_y.loc[:, high_y.columns.str.endswith("std_Intra")].iloc[:, 0]),
    )

    # Create interpolation functions
    mean = interpolate.interp1d(x, mean_df.to_numpy())
    sigma_total = interpolate.interp1d(x, sigma_total_df.to_numpy())
    sigma_inter = interpolate.interp1d(x, sigma_inter_df.to_numpy())
    sigma_intra = interpolate.interp1d(x, sigma_intra_df.to_numpy())

    return pd.DataFrame(
        {
            f"pSA_{period}_mean": np.log(mean(period)),
            f"pSA_{period}_std_Total": np.log(sigma_total(period)),
            f"pSA_{period}_std_Inter": np.log(sigma_inter(period)),
            f"pSA_{period}_std_Intra": np.log(sigma_intra(period)),
        }
    )


def oq_run(
    model: Enum,
    tect_type: Enum,
    rupture_df: pd.DataFrame,
    im: str,
    periods: Sequence[int] = None,
    **kwargs,
):
    """Run an openquake model with dataframe
    model: Enum
        OQ model name
    tect_type: Enum
        One of the tectonic types from
        ACTIVE_SHALLOW, SUBDUCTION_SLAB and SUBDUCTION_INTERFACE
    rupture_df: Rupture DF
        Columns for properties. E.g., vs30, z1pt0, rrup, rjb, mag, rake, dip....
        Rows be the separate site-fault pairs
        But Site information must be identical across the rows,
        only the faults can be different.
    im: string
        intensity measure
    periods: Sequence[int]
        for spectral acceleration, openquake tables automatically
        interpolate values between specified values, fails if outside range
    kwargs: pass extra (model specific) parameters to models
    """
    model = (
        OQ_MODELS[model][tect_type](**kwargs)
        if not model.name.endswith("_NZ")
        else OQ_MODELS[model][tect_type](region="NZL", **kwargs)
    )

    # Check the given tect_type with its model's tect type
    trt = model.DEFINED_FOR_TECTONIC_REGION_TYPE
    if trt == const.TRT.SUBDUCTION_INTERFACE:
        assert tect_type == TectType.SUBDUCTION_INTERFACE
    elif trt == const.TRT.SUBDUCTION_INTRASLAB:
        assert tect_type == TectType.SUBDUCTION_SLAB
    elif trt == const.TRT.ACTIVE_SHALLOW_CRUST:
        assert tect_type == TectType.ACTIVE_SHALLOW
    else:
        raise ValueError("unknown tectonic region: " + trt)

    stddev_types = [
        std for std in SPT_STD_DEVS if std in model.DEFINED_FOR_STANDARD_DEVIATION_TYPES
    ]

    # Make a copy in case the original rupture_df used with other functions
    rupture_df = rupture_df.copy()

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
                    (column, rupture_df.loc[:, column].values,)
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
        single = False
        if not hasattr(periods, "__len__"):
            single = True
            periods = [periods]
        results = []
        for period in periods:
            im = imt.SA(period=min(period, max_period))
            try:
                result = oq_mean_stddevs(model, rupture_ctx, im, stddev_types)
            except:
                # Period is smaller than model's supported min_period E.g., ZA_06
                # Interpolate between PGA(0.0) and model's min_period
                low_result = oq_mean_stddevs(
                    model, rupture_ctx, imt.PGA(), stddev_types
                )
                high_period = avail_periods[period <= avail_periods][0]
                high_result = oq_mean_stddevs(
                    model, rupture_ctx, imt.SA(period=high_period), stddev_types
                )

                result = interpolate_to_closest(
                    period, np.array([0.0, high_period]), low_result, high_result
                )

            # extrapolate pSA value up based on maximum available period
            if period > max_period:
                result.loc[:, result.columns.str.endswith("mean")] = np.log(
                    np.exp(result.loc[:, result.columns.str.endswith("mean")])
                    * (max_period / period) ** 2
                )
            results.append(result)
        if single:
            return results[0]
        return pd.concat(results, axis=1)
    else:
        imc = getattr(imt, im)
        assert imc in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        return oq_mean_stddevs(model, rupture_ctx, imc(), stddev_types)
