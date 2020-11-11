"""
Wrapper for openquake models.
"""
import numpy as np

from empirical.util.classdef import TectType
from empirical.openquake import const, imt

# all openquake models
from empirical.GMM_models import parker_2020


# numbers to match empirical.util.classdef.GMM
OQ_GMM = {
    1012: parker_2020.ParkerEtAl2020SInter,
    1013: parker_2020.ParkerEtAl2020SSlab,
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

    stddev_types = []
    # DEFINED_FOR_STANDARD_DEVIATION_TYPES
    # const.StdDev.{TOTAL,INTER_EVENT,INTRA_EVENT}

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
    return mean
