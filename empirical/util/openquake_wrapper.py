"""
Wrapper for openquake models.
"""
import numpy as np

from empirical.util.classdef import TectType

# openquake constants and models
from openquake.hazardlib import const, imt, gsim


# numbers to match empirical.util.classdef.GMM
OQ_GMM = {
    1012: gsim.parker_2020.ParkerEtAl2020SInter,
    1013: gsim.parker_2020.ParkerEtAl2020SSlab,
    1021: gsim.hassani_atkinson_2020.HassaniAtkinson2020Asc,
    1022: gsim.hassani_atkinson_2020.HassaniAtkinson2020SInter,
    1023: gsim.hassani_atkinson_2020.HassaniAtkinson2020SSlab,
}


class Properties(object):
    """
    Stores values for sites, rup and dists.
    """

    def __init__(self):
        pass


def oq_run(model_id, site, fault, im, period, **kwargs):
    model = OQ_GMM[model_id](**kwargs)

    trt = model.DEFINED_FOR_TECTONIC_REGION_TYPE
    if trt == const.TRT.SUBDUCTION_INTERFACE:
        assert fault.tect_type == TectType.SUBDUCTION_INTERFACE
    elif trt == const.TRT.SUBDUCTION_INTRASLAB:
        assert fault.tect_type == TectType.SUBDUCTION_SLAB
    elif trt == const.TRT.ACTIVE_SHALLOW_CRUST:
        assert fault.tect_type == TectType.ACTIVE_SHALLOW
    else:
        raise ValueError("unknown tectonic region: " + trt)

    if period is not None:
        assert imt.SA in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        # periods = sorted([i.period for i in model.COEFFS.sa_coeffs.keys()])
        imr = imt.SA(period=period)
    elif im == "PGV":
        assert imt.PGV in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        imr = imt.PGV()
    elif im == "PGA":
        assert imt.PGA in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        imr = imt.PGA()
    elif im == "PGD":
        assert imt.PGD in model.DEFINED_FOR_INTENSITY_MEASURE_TYPES
        imr = imt.PGD()
    else:
        raise ValueError("unknown im: " + im)

    stddev_types = []
    for st in [const.StdDev.TOTAL, const.StdDev.INTER_EVENT, const.StdDev.INTRA_EVENT]:
        if st in model.DEFINED_FOR_STANDARD_DEVIATION_TYPES:
            stddev_types.append(st)

    sites = Properties()
    for sp in model.REQUIRES_SITES_PARAMETERS:
        if sp == "vs30":
            sites.vs30 = np.array([site.vs30])
        elif sp == "vs30measured":
            sites.vs30measured = np.array([site.vs30measured])
        elif sp == "z1pt0":
            sites.z1pt0 = np.array([site.z1p0])
        elif sp == "z2pt5":
            sites.set_z2pt5 = np.array([site.z2p5])
        else:
            raise ValueError("unknown site property: " + sp)

    rup = Properties()
    for rp in model.REQUIRES_RUPTURE_PARAMETERS:
        if rp == "dip":
            rup.dip = fault.dip
        elif rp == "hypo_depth":
            rup.hypo_depth = fault.hdepth
        elif rp == "mag":
            rup.mag = fault.Mw
        elif rp == "rake":
            rup.rake = fault.rake
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
        else:
            raise ValueError("unknown dist property: " + dp)

    mean, stddevs = model.get_mean_and_stddevs(sites, rup, dists, imr, stddev_types)
    mean = mean[0] if hasattr(mean, "__len__") else mean
    stddevs = [s[0] if hasattr(s, "__len__") else s for s in stddevs]

    return mean, stddevs
