"""Wrapper for openquake vectorized models."""
from typing import List

import pandas as pd

from openquake.hazardlib import const, imt, gsim, contexts


OQ_MODELS = {
    "Br_13": gsim.bradley_2013.Bradley2013,
    "ASK_14": gsim.abrahamson_2014.AbrahamsonEtAl2014,
    "BSSA_14": gsim.boore_2014.BooreEtAl2014,
    "CY_14": gsim.chiou_youngs_2014.ChiouYoungs2014,
    "Z_06": gsim.zhao_2006.ZhaoEtAl2006Asc,
    "P_20": gsim.parker_2020.ParkerEtAl2020SInter,
    "K_20": gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
    "AG_20": gsim.kuehn_2020.KuehnEtAl2020SInter,
}


def convert_im_label(im: imt):
    """Convert OQ's SA(period) to the internal naming, pSA_period
    im: imt
    """
    imt_tuple = imt.imt2tup(im.string)
    return im if len(imt_tuple) == 1 else f"pSA_{imt_tuple[-1]}"


def oq_mean_stddevs(
    model: gsim, ctx: contexts.RuptureContext, im: imt, stddev_types: List
):
    """Calculate mean and standard deviations given openquake input structures.
    model: OQ Model
    ctx: contexts.RuptureContext
        OQ RuptureContext that contains the following information
        - Site
        - Distance
        - Rupture
    im: imt
    stddev_types: List
    """
    mean, stddevs = model.get_mean_and_stddevs(ctx, ctx, ctx, im, stddev_types)
    mean_stddev_dict = {f"{convert_im_label(im)}_mean": mean}
    for idx, std_dev in enumerate(stddev_types):
        mean_stddev_dict[f"{convert_im_label(im)}_std_{std_dev.split()[0]}"] = stddevs[
            idx
        ]

    return pd.DataFrame(mean_stddev_dict)


def oq_run(
    model: str, rupture_df: pd.DataFrame, im: str, period: List = None, **kwargs
):
    """Run an openquake model with dataframe
    model: OQ model name, string
        Only support Bradley_2013 for now
    rupture_df: Rupture DF
        Columns for properties. E.g., vs30, z1pt0, rrup, rjb, mag, rake, dip....
        Rows be the separate site-fault pairs
        But Site information must be identical across the rows,
        only the faults can be different.
    im: intensity measure name
    period: for spectral acceleration, openquake tables automatically
            interpolate values between specified values, fails if outside range
    kwargs: pass extra (model specific) parameters to models
    """
    model = OQ_MODELS[model](**kwargs)

    stddev_types = []
    for st in [const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT]:
        if st in model.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
            stddev_types.append(st)

    # Check if df contains what model requires
    rupture_ctx_properties = list(rupture_df.columns.values)
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

    # Make a copy in case the original rupture_df used with other functions
    copied_rupture_df = rupture_df.copy()
    # Convert z1pt0 from km to m
    copied_rupture_df["z1pt0"] *= 1000
    # OQ's single new-style context which contains all site, distance and rupture's information
    rupture_ctx = contexts.RuptureContext(
        tuple(
            [
                # Openquake requiring occurrence_rate attribute to exist
                ("occurrence_rate", None),
                # sids is basically the number of sites provided
                # each row of DF is site & rupture pair
                ("sids", [None] * len(rupture_df.index)),
                *(
                    (
                        column,
                        copied_rupture_df.loc[:, column].values,
                    )
                    for column in copied_rupture_df.columns.values
                ),
            ]
        )
    )

    if period is not None:
        assert imt.SA in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        # use sorted instead of max for full list
        max_period = max([i.period for i in model.COEFFS.sa_coeffs.keys()])
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
