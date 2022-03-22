"""Wrapper for openquake vectorized models."""
from typing import Sequence
from enum import Enum

import pandas as pd
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
    mean, stddevs = model.get_mean_and_stddevs(ctx, ctx, ctx, im, stddev_types)
    mean_stddev_dict = {f"{convert_im_label(im)}_mean": mean}
    for idx, std_dev in enumerate(stddev_types):
        mean_stddev_dict[f"{convert_im_label(im)}_std_{std_dev.split()[0]}"] = stddevs[
            idx
        ]

    return pd.DataFrame(mean_stddev_dict)


def oq_run(
    model: Enum,
    tect_type: Enum,
    rupture_df: pd.DataFrame,
    im: str,
    period: Sequence[int] = None,
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
    period: Sequence[int]
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
                    (
                        column,
                        rupture_df.loc[:, column].values,
                    )
                    for column in rupture_df.columns.values
                ),
            ]
        )
    )

    if period is not None:
        assert imt.SA in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        # use sorted instead of max for full list
        max_period = (
            max([i.period for i in model.COEFFS.sa_coeffs.keys()])
            if not isinstance(
                model,
                (
                    gsim.zhao_2006.ZhaoEtAl2006Asc,
                    gsim.zhao_2006.ZhaoEtAl2006SSlab,
                    gsim.zhao_2006.ZhaoEtAl2006SInter,
                ),
            )
            else max([i.period for i in model.COEFFS_ASC.sa_coeffs.keys()])
        )
        single = False
        if not hasattr(period, "__len__"):
            single = True
            period = [period]
        results = []
        for p in period:
            im = imt.SA(period=min(p, max_period))
            result = oq_mean_stddevs(model, rupture_ctx, im, stddev_types)
            # interpolate pSA value up based on maximum available period
            if p > max_period:
                result.update(
                    pd.Series(result.get("mean") * (max_period / p) ** 2, name="mean")
                )
            results.append(result)
        if single:
            return results[0]
        return pd.concat(results, axis=1)
    else:
        imc = getattr(imt, im)
        assert imc in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        return oq_mean_stddevs(model, rupture_ctx, imc(), stddev_types)
