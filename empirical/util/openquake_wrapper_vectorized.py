"""
Wrapper for openquake models.
Can import without openquake but using openquake models will raise ImportError.

Currently, focussing on implementing Bradley model only
"""
from math import exp

import numpy as np
import pandas as pd

from empirical.util.classdef import TectType, GMM

try:
    # openquake constants and models
    from openquake.hazardlib import const, imt, gsim
    from openquake.hazardlib.site import Site, SiteCollection
    from openquake.hazardlib.geo import Point
    from openquake.hazardlib.contexts import RuptureContext

    OQ = True
except ImportError:
    # fail silently, only an issue if openquake models wanted
    OQ = False

# (oq_property name, ee_property_name)
CTX_PROPERTIES = [
    ("vs30", "vs30"),
    ("vs30measured", "vs30measured"),
    ("z1pt0", "z1p0"),
    ("z2pt5", "z2p5"),
    ("mag", "Mw"),
    ("rake", "rake"),
    ("dip", "dip"),
    ("ztor", "ztor"),
    ("hypo_depth", "hdepth"),
    ("rrup", "Rrup"),
    ("rjb", "Rjb"),
    ("rx", "Rx"),
    ("ry0", "Ry"),
    ("rvolc", "Rtvz"),
]


def oq_mean_stddevs(model, ctx, imr, stddev_types):
    """
    Calculate mean and standard deviations given openquake input structures.
    """
    mean, stddevs = model.get_mean_and_stddevs(ctx, ctx, ctx, imr, stddev_types)
    mean_stddev_dict = {"mean": mean}
    for idx, std_dev in enumerate(stddev_types):
        mean_stddev_dict[f"std_{std_dev.split()[0]}"] = stddevs[idx]

    return pd.DataFrame(mean_stddev_dict)


def oq_run(model, rupture_df, im, period=None, **kwargs):
    """
    Run an openquake model with dataframe
    model: OQ models
        Only support Bradley_2013 for now
    rupture_df: Rupture DF and it's a single new-style context OQ uses
        Columns for properties. E.g., vs30, z1pt0, rrup, rjb, mag, rake, dip....
        Rows be the separate site-fault pairs
        But Site information must be identical across the rows,
        only the faults can be different
    im: intensity measure name
    period: for spectral acceleration, openquake tables automatically
            interpolate values between specified values, fails if outside range
    kwargs: pass extra (model specific) parameters to models
    """
    if not OQ:
        raise ImportError("openquake is not installed, models not available")

    model = model(**kwargs)

    stddev_types = []
    for st in [const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT]:
        if st in model.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
            stddev_types.append(st)

    # Check if df contains what we need
    extra_site_parameters = set(model.REQUIRES_SITES_PARAMETERS).difference(
        list(rupture_df.columns.values)
    )
    if len(extra_site_parameters) > 0:
        raise ValueError("unknown site property: " + extra_site_parameters)

    extra_rup_properties = set(model.REQUIRES_RUPTURE_PARAMETERS).difference(
        list(rupture_df.columns.values)
    )
    if len(extra_rup_properties) > 0:
        raise ValueError("unknown rupture property: " + " ".join(extra_rup_properties))

    extra_dist_properties = set(model.REQUIRES_DISTANCES).difference(
        list(rupture_df.columns.values)
    )
    if len(extra_dist_properties) > 0:
        raise ValueError(
            "unknown distance property: " + " ".join(extra_dist_properties)
        )

    rupture = RuptureContext(
        (
            # Sites
            # Create a dummy location as OQ calculation doesn't use a location
            ("location", Point(0.0, 0.0, 0.0)),
            ("vs30", rupture_df.vs30.values),
            ("z1pt0", rupture_df.z1pt0.values),
            ("z2pt5", rupture_df.z2pt5.values),
            ("vs30measured", rupture_df.vs30measured.values),
            # Distances
            ("rrup", rupture_df.rrup.values),
            ("rjb", rupture_df.rjb.values),
            ("rx", rupture_df.rx.values),
            ("rvolc", rupture_df.rvolc.values),
            # Rupture
            ("mag", rupture_df.mag.values),
            ("rake", rupture_df.rake.values),
            ("dip", rupture_df.dip.values),
            ("ztor", rupture_df.ztor.values),
            ("hypo_depth", rupture_df.hypo_depth.values),
            # Openquake requiring occurrence_rate attribute to exist
            ("occurrence_rate", None),
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
            imr = imt.SA(period=min(p, max_period))
            result = oq_mean_stddevs(model, rupture, imr, stddev_types)
            # interpolate pSA value up based on maximum available period
            if p > max_period:
                result.update(
                    pd.Series(result.get("mean") * (max_period / p) ** 2, name="mean")
                )
            results.append(result)
        if single:
            return results[0]
        return results
    else:
        imc = getattr(imt, im)
        assert imc in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        return oq_mean_stddevs(model, rupture, imc(), stddev_types)
