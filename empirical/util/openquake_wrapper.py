"""
Wrapper for openquake models.
Can import without openquake but using openquake models will raise ImportError.
"""
from math import exp

import numpy as np

from empirical.util.classdef import TectType, GMM

try:
    # openquake constants and models
    from openquake.hazardlib import const, imt, gsim
    from openquake.hazardlib.site import Site, SiteCollection
    from openquake.hazardlib.geo import Point

    OQ = True
except ImportError:
    # fail silently, only an issue if openquake models wanted
    OQ = False

SITE_PROPERTIES = [("vs30", "vs30"), ("vs30measured", "vs30measured"), ("z1pt0", "z1p0"), ("z2pt5", "z2p5"), ("fpeak", "fpeak")]
RUPTURE_PROPERTIES = [("mag", "Mw"), ("rake", "rake"), ("width", "width"), ("ztor", "ztor")]
DISTANCE_PROPERTIES = [("rrup", "Rrup"), ("rjb", "Rjb"), ("rx", "Rx"), ("ry0", "Ry"), ("rvolc", "Rtvz")]

# GMM numbers to match empirical.util.classdef.GMM
OQ_GMM = [
    1012,
    1013,
    1021,
    1022,
    1023,
    1031,
    1041,
    1051,
    1061,
    1062,
    1063,
    1071,
    1072,
    1073,
    1082,
    1083,
    2082,
    2083,
    1092,
    1093,
    2092,
    2093,
    1102,
    1103,
    1111,
    1112,
    1113,
]
if OQ:
    # model classes in order of empirical.util.classdef.GMM
    oq_models = [
        gsim.parker_2020.ParkerEtAl2020SInter,
        gsim.parker_2020.ParkerEtAl2020SSlab,
        gsim.hassani_atkinson_2020.HassaniAtkinson2020Asc,
        gsim.hassani_atkinson_2020.HassaniAtkinson2020SInter,
        gsim.hassani_atkinson_2020.HassaniAtkinson2020SSlab,
        gsim.gulerce_2017.GulerceEtAl2017,
        gsim.bozorgnia_campbell_2016.BozorgniaCampbell2016,
        gsim.stewart_2016_vh.StewartEtAl2016VH,
        gsim.phung_2020.PhungEtAl2020Asc,
        gsim.phung_2020.PhungEtAl2020SInter,
        gsim.phung_2020.PhungEtAl2020SSlab,
        gsim.chao_2020.ChaoEtAl2020Asc,
        gsim.chao_2020.ChaoEtAl2020SInter,
        gsim.chao_2020.ChaoEtAl2020SSlab,
        gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
        gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
        gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SInter,
        gsim.abrahamson_gulerce_2020.AbrahamsonGulerce2020SSlab,
        gsim.kuehn_2020.KuehnEtAl2020SInter,
        gsim.kuehn_2020.KuehnEtAl2020SSlab,
        gsim.kuehn_2020.KuehnEtAl2020SInter,
        gsim.kuehn_2020.KuehnEtAl2020SSlab,
        gsim.si_2020.SiEtAl2020SInter,
        gsim.si_2020.SiEtAl2020SSlab,
        gsim.zhao_2016.ZhaoEtAl2016Asc,
        gsim.zhao_2016.ZhaoEtAl2016SInter,
        gsim.zhao_2016.ZhaoEtAl2016SSlab,
    ]
    oq_models = dict(zip(OQ_GMM, oq_models))


class Properties(object):
    """
    Stores values for sites, rup and dists.
    """

    def __init__(self):
        # this allows attaching arbitrary attributes to self later
        pass


def oq_mean_stddevs(model, sites, rup, dists, imr, stddev_types):
    """
    Calculate mean and standard deviations given openquake input structures.
    """
    mean, stddevs = model.get_mean_and_stddevs(sites, rup, dists, imr, stddev_types)
    mean = exp(mean[0]) if hasattr(mean, "__len__") else exp(mean)
    stddevs = [s[0] if hasattr(s, "__len__") else s for s in stddevs]

    return mean, stddevs


def oq_run(model, site, fault, im, period=None, **kwargs):
    """
    Run an openquake model using Empirical_Engine input structures.
    model: model or value from empirical.util.classdef.GMM or openquake class:
           GMM.P_20_SI GMM.P_20_SI.value gsim.parker_2020.ParkerEtAl2020SInter
    site / fault: instances from empirical.classdef
    im: intensity measure name
    period: for spectral acceleration, openquake tables automatically
            interpolate values between specified values, fails if outside range
    kwargs: pass extra (model specific) parameters to models
    """
    if not OQ:
        raise ImportError("openquake is not installed, models not available")

    if model in [GMM.K_20_SI_NZ, GMM.K_20_SS_NZ, GMM.AG_20_SI_NZ, GMM.AG_20_SS_NZ]:
        kwargs["region"] = "NZL"

    # model can be given multiple ways
    if type(model).__name__ == "GMM":
        model = oq_models[model.value](**kwargs)
    elif type(model).__name__ == "MetaGSIM":
        model = model(**kwargs)
    elif type(model).__name__ == "int":
        model = oq_models[model](**kwargs)

    trt = model.DEFINED_FOR_TECTONIC_REGION_TYPE
    if trt == const.TRT.SUBDUCTION_INTERFACE:
        assert fault.tect_type == TectType.SUBDUCTION_INTERFACE
    elif trt == const.TRT.SUBDUCTION_INTRASLAB:
        assert fault.tect_type == TectType.SUBDUCTION_SLAB
    elif trt == const.TRT.ACTIVE_SHALLOW_CRUST:
        assert fault.tect_type == TectType.ACTIVE_SHALLOW
    else:
        raise ValueError("unknown tectonic region: " + trt)

    stddev_types = []
    for st in [const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT]:
        if st in model.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
            stddev_types.append(st)

    location = Point(
        0.0, 0.0, 0.0
    )  # Create a dummy location as OQ calculation doesn't use a location
    oq_site = Site(location)
    extra_site_parameters = set(model.REQUIRES_SITES_PARAMETERS).difference(["vs30", "vs30measured", "z1pt0", "z2pt5", "fpeak"])
    if len(extra_site_parameters) > 0:
        raise ValueError("unknown site property: " + extra_site_parameters)
    oq_site = check_properties(site, model, SITE_PROPERTIES, oq_site, np_array=True)

    sites = SiteCollection([oq_site])

    extra_rup_properties = set(model.REQUIRES_RUPTURE_PARAMETERS).difference(["dip", "rake", "hypo_depth", "mag", "width", "ztor"])
    if len(extra_rup_properties) > 0:
            raise ValueError("unknown rupture property: " + " ".join(extra_rup_properties))
    rupture = check_properties(fault, model, RUPTURE_PROPERTIES, Properties())

    extra_dist_properties = set(model.REQUIRES_DISTANCES).difference(["rrup", "rjb", "rx", "ry0", "rvolc"])
    if len(extra_dist_properties) > 0:
            raise ValueError("unknown distance property: " + " ".join(extra_dist_properties))
    dists = check_properties(site, model, DISTANCE_PROPERTIES, Properties(), np_array=True)

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
            m, s = oq_mean_stddevs(model, sites, rupture, dists, imr, stddev_types)
            # interpolate pSA value up based on maximum available period
            if p > max_period:
                m = m * (max_period / p) ** 2
            results.append((m, s))
        if single:
            return results[0]
        return results
    else:
        imc = getattr(imt, im)
        assert imc in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        return oq_mean_stddevs(model, sites, rupture, dists, imc(), stddev_types)


def check_properties(ee_object, model, properties, properties_obj, np_array=False):
    for oq_property_name, ee_property_name in properties:
        ee_property = getattr(ee_object, ee_property_name)
        if ee_property:
            setattr(properties_obj, oq_property_name, np.array([ee_property]) if np_array else ee_property)
        else:
            check_param(model, oq_property_name)
    return properties_obj


def check_param(model, rp):
    if rp in model.REQUIRES_RUPTURE_PARAMETERS:
        raise ValueError(f"{rp} is a required parameter for {model}")
