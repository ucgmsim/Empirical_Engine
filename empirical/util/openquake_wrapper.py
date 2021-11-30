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
    }


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
           GMM.P_20 gsim.parker_2020.ParkerEtAl2020SInter
    site / fault: instances from empirical.classdef -- A tect_type must be able to be set to retrieve the correct model
    im: intensity measure name
    period: for spectral acceleration, openquake tables automatically
            interpolate values between specified values, fails if outside range
    kwargs: pass extra (model specific) parameters to models
    """
    if not OQ:
        raise ImportError("openquake is not installed, models not available")

    # model can be given multiple ways
    if type(model).__name__ == "GMM":
        model = oq_models[model][fault.tect_type](**kwargs)
    elif type(model).__name__ == "MetaGSIM":
        model = model(**kwargs)

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
    extra_site_parameters = set(model.REQUIRES_SITES_PARAMETERS).difference(
        list(zip(*SITE_PROPERTIES))[0]
    )
    if len(extra_site_parameters) > 0:
        raise ValueError("unknown site property: " + extra_site_parameters)
    oq_site = check_properties(site, model, SITE_PROPERTIES, oq_site, np_array=True)

    sites = SiteCollection([oq_site])

    extra_rup_properties = set(model.REQUIRES_RUPTURE_PARAMETERS).difference(
        list(zip(*RUPTURE_PROPERTIES))[0]
    )
    if len(extra_rup_properties) > 0:
        raise ValueError("unknown rupture property: " + " ".join(extra_rup_properties))
    rupture = check_properties(fault, model, RUPTURE_PROPERTIES, Properties())
    # Openquake requiring occurrence_rate attribute to exist
    rupture.occurrence_rate = None

    extra_dist_properties = set(model.REQUIRES_DISTANCES).difference(
        list(zip(*DISTANCE_PROPERTIES))[0]
    )
    if len(extra_dist_properties) > 0:
        raise ValueError(
            "unknown distance property: " + " ".join(extra_dist_properties)
        )
    dists = check_properties(
        site, model, DISTANCE_PROPERTIES, Properties(), np_array=True
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
        if ee_property is not None:
            setattr(
                properties_obj,
                oq_property_name,
                np.array([ee_property]) if np_array else ee_property,
            )
        else:
            check_param(model, oq_property_name)
    return properties_obj


def check_param(model, rp):
    if rp in model.REQUIRES_RUPTURE_PARAMETERS:
        raise ValueError(f"{rp} is a required parameter for {model}")
