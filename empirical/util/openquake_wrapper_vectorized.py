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
SITE_PROPERTIES = [
    ("vs30", "vs30"),
    ("vs30measured", "vs30measured"),
    ("z1pt0", "z1p0"),
    ("z2pt5", "z2p5"),
    ("fpeak", "fpeak"),
]
RUPTURE_PROPERTIES = [
    ("mag", "Mw"),
    ("rake", "rake"),
    ("width", "width"),
    ("ztor", "ztor"),
    ("hypo_depth", "hdepth"),
]
DISTANCE_PROPERTIES = [
    ("rrup", "Rrup"),
    ("rjb", "Rjb"),
    ("rx", "Rx"),
    ("ry0", "Ry"),
    ("rvolc", "Rtvz"),
]

OQ_GMM_LIST = [
    GMM.P_20,
    GMM.HA_20,
    GMM.G_17,
    GMM.BC_16,
    GMM.S_16,
    GMM.Ph_20,
    GMM.Ch_20,
    GMM.AG_20,
    GMM.AG_20_NZ,
    GMM.K_20,
    GMM.K_20_NZ,
    GMM.Si_20,
    GMM.Z_16,
    GMM.Br_13,
]
if OQ:
    oq_models = {
        GMM.P_20: {
            TectType.SUBDUCTION_SLAB: gsim.parker_2020.ParkerEtAl2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.parker_2020.ParkerEtAl2020SInter,
        },
        GMM.HA_20: {
            TectType.ACTIVE_SHALLOW: gsim.hassani_atkinson_2020.HassaniAtkinson2020Asc,
            TectType.SUBDUCTION_SLAB: gsim.hassani_atkinson_2020.HassaniAtkinson2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.hassani_atkinson_2020.HassaniAtkinson2020SInter,
        },
        GMM.G_17: {TectType.ACTIVE_SHALLOW: gsim.gulerce_2017.GulerceEtAl2017},
        GMM.BC_16: {
            TectType.ACTIVE_SHALLOW: gsim.bozorgnia_campbell_2016.BozorgniaCampbell2016
        },
        GMM.S_16: {TectType.ACTIVE_SHALLOW: gsim.stewart_2016_vh.StewartEtAl2016VH},
        GMM.Ph_20: {
            TectType.ACTIVE_SHALLOW: gsim.phung_2020.PhungEtAl2020Asc,
            TectType.SUBDUCTION_SLAB: gsim.phung_2020.PhungEtAl2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.phung_2020.PhungEtAl2020SInter,
        },
        GMM.Ch_20: {
            TectType.ACTIVE_SHALLOW: gsim.chao_2020.ChaoEtAl2020Asc,
            TectType.SUBDUCTION_SLAB: gsim.chao_2020.ChaoEtAl2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.chao_2020.ChaoEtAl2020SInter,
        },
        GMM.AG_20: {
            TectType.SUBDUCTION_SLAB: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
        },
        GMM.AG_20_NZ: {
            TectType.SUBDUCTION_SLAB: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
        },
        GMM.K_20: {
            TectType.SUBDUCTION_SLAB: gsim.kuehn_2020.KuehnEtAl2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.kuehn_2020.KuehnEtAl2020SInter,
        },
        GMM.K_20_NZ: {
            TectType.SUBDUCTION_SLAB: gsim.kuehn_2020.KuehnEtAl2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.kuehn_2020.KuehnEtAl2020SInter,
        },
        GMM.Si_20: {
            TectType.SUBDUCTION_SLAB: gsim.si_2020.SiEtAl2020SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.si_2020.SiEtAl2020SInter,
        },
        GMM.Z_16: {
            TectType.ACTIVE_SHALLOW: gsim.zhao_2016.ZhaoEtAl2016Asc,
            TectType.SUBDUCTION_SLAB: gsim.zhao_2016.ZhaoEtAl2016SSlab,
            TectType.SUBDUCTION_INTERFACE: gsim.zhao_2016.ZhaoEtAl2016SInter,
        },
        GMM.Br_13: {TectType.ACTIVE_SHALLOW: gsim.bradley_2013.Bradley2013},
    }


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
