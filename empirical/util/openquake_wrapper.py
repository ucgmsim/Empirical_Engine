"""
Wrapper for openquake models.
Can import without openquake but using openquake models will raise ImportError.
"""
from math import exp

import numpy as np

from empirical.util.classdef import TectType

try:
    # openquake constants and models
    from openquake.hazardlib import const, imt, gsim
    from openquake.hazardlib.site import Site, SiteCollection
    from openquake.hazardlib.geo import Point

    OQ = True
except ImportError:
    # fail silently, only an issue if openquake models wanted
    OQ = False

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
    1081,
    1082,
    1091,
    1092,

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
        gsim.kuehn_2020.KuehnEtAl2020SInter,
        gsim.kuehn_2020.KuehnEtAl2020SSlab,
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

    location = Point(0.0, 0.0, 0.0)  # Create a dummy location as OQ calculation doesn't use a location
    oq_site = Site(location)
    for sp in model.REQUIRES_SITES_PARAMETERS:
        if sp == "vs30":
            oq_site.vs30 = np.array([site.vs30])
        elif sp == "vs30measured":
            oq_site.vs30measured = np.array([site.vs30measured])
        elif sp == "z1pt0":
            oq_site.z1pt0 = np.array([site.z1p0])
        elif sp == "z2pt5":
            oq_site.z2pt5 = np.array([site.z2p5])
        elif sp == "fpeak":
            oq_site.fpeak = np.array([site.fpeak])
        else:
            raise ValueError("unknown site property: " + sp)

    sites = SiteCollection([oq_site])

    rup = Properties()
    for rp in model.REQUIRES_RUPTURE_PARAMETERS:
        if rp == "dip":
            rup.dip = fault.dip
        elif rp == "hypo_depth":
            rup.hypo_depth = fault.hdepth
        elif rp == "mag":
            rup.mag = fault.Mw
        elif rp == "rake":
            # rake is used instead of classdef.Fault.faultstyle
            # because different models have different rake cutoffs
            rup.rake = fault.rake
        elif rp == "width":
            rup.width = fault.width
        elif rp == "ztor":
            rup.ztor = fault.ztor
        else:
            raise ValueError("unknown rupture property: " + rp)

    dists = Properties()
    for dp in model.REQUIRES_DISTANCES:
        if dp == "rrup":
            dists.rrup = np.array([site.Rrup])
        elif dp == "rjb":
            dists.rjb = np.array([site.Rjb])
        elif dp == "rx":
            dists.rx = np.array([site.Rx])
        elif dp == "ry0":
            dists.ry0 = np.array([site.Ry])
        else:
            raise ValueError("unknown dist property: " + dp)

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
            m, s = oq_mean_stddevs(model, sites, rup, dists, imr, stddev_types)
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
        return oq_mean_stddevs(model, sites, rup, dists, imc(), stddev_types)
