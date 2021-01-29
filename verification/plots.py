#!/usr/bin/env python
"""
Plots of models over changes in parameters.
Creates plots to be joined by plots_join.sh into PDFs.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from empirical.util.classdef import Site, Fault, TectType, GMM
from empirical.util.empirical_factory import compute_gmm

from openquake.hazardlib import gsim

site = Site()
fault = Fault()


# IMs to loop through for magnitude and rrup scaling plots
ims = ["PGA", "PGV", 0.1, 0.5, 0.75, 1, 3, 5]
# rrup scaling plots fixed magnitudes
mws = [8, 8.25, 8.5, 8.75, 9, 9.2]
# magnitude scaling plots fixed rrups
rrs = [25, 50, 75, 100, 300, 600]

# set of subduction interface models
gmms_if = {
    gsim.parker_2020.ParkerEtAl2020SInter: (
        "Parker 2020",
        TectType.SUBDUCTION_INTERFACE,
    ),
    gsim.phung_2020.PhungEtAl2020SInter: ("Phung 2020", TectType.SUBDUCTION_INTERFACE),
    gsim.chao_2020.ChaoEtAl2020SInter: ("Chao 2020", TectType.SUBDUCTION_INTERFACE),
    gsim.hassani_atkinson_2020.HassaniAtkinson2020SInter: (
        "Hassani Atkinson 2020",
        TectType.SUBDUCTION_INTERFACE,
    ),
    GMM.ZA_06: ("Zhao 2006", TectType.SUBDUCTION_INTERFACE),
    GMM.BCH_16: ("BC Hydro 2016", TectType.SUBDUCTION_INTERFACE),
}
# set of subduction slab models
gmms_sl = {
    gsim.parker_2020.ParkerEtAl2020SSlab: ("Parker 2020", TectType.SUBDUCTION_SLAB),
    gsim.phung_2020.PhungEtAl2020SSlab: ("Phung 2020", TectType.SUBDUCTION_SLAB),
    gsim.chao_2020.ChaoEtAl2020SSlab: ("Chao 2020", TectType.SUBDUCTION_SLAB),
    gsim.hassani_atkinson_2020.HassaniAtkinson2020SSlab: (
        "Hassani Atkinson 2020",
        TectType.SUBDUCTION_SLAB,
    ),
    GMM.ZA_06: ("Zhao 2006", TectType.SUBDUCTION_SLAB),
    GMM.BCH_16: ("BC Hydro 2016", TectType.SUBDUCTION_SLAB),
}


# control parameters (unchanging)
site.z1p0 = 80
site.fpeak = np.array([12])
site.z2p5 = 245
fault.hdepth = 20
fault.ztor = 4
fault.rake = 30
fault.dip = 45
fault.width = 20

# fixed magnitude, rrup range
for i, im in enumerate(ims):
    for m, mag in enumerate(mws):
        fault.Mw = mag
        if type(im).__name__ in ["int", "float"]:
            imt = "SA"
        else:
            imt = im
        for gi, gmms in enumerate([gmms_if, gmms_sl]):
            for g in gmms:
                if type(g).__name__ == "MetaGSIM":
                    if imt not in [
                        x.__name__ for x in g.DEFINED_FOR_INTENSITY_MEASURE_TYPES
                    ]:
                        # model does not support IM
                        continue
                else:
                    if imt not in ["SA", "PGA"]:
                        # empirical engine ones don't have PGV etc...
                        # need to update as required
                        continue
                fault.tect_type = gmms[g][1]
                x = np.logspace(1, 3)
                y = []
                for rrup in x:
                    site.Rrup = rrup
                    site.Rjb = rrup * 0.9
                    period = None if imt != "SA" else im
                    # but zhao expects a list?
                    if g == GMM.ZA_06:
                        v = compute_gmm(fault, site, g, imt, period=[period])
                        if imt == "PGA":
                            # result
                            v, stdvs = v
                        else:
                            # list of result for each period
                            v, stdvs = v[0]
                    else:
                        v, stdvs = compute_gmm(fault, site, g, imt, period=period)
                    y.append(v[0] if hasattr(v, "__len__") else v)
    
                y = np.array(y)
                plt.loglog(x, y, label=gmms[g][0])
                plt.fill_between(x, y * np.exp(-stdvs[0]), y * np.exp(stdvs[0]), alpha=0.1)
            plt.legend()
            plt.xlabel("rrup")
            y = imt
            if imt == "SA":
                y += " " + str(im)
            plt.ylabel(y)
            plt.title("Mw = " + str(mag))
            plt.savefig(f"r{gi}{i}{m}.png")
            plt.close()

# fixed rrup, magnitude range
for i, im in enumerate(ims):
    for r, rrup in enumerate(rrs):
        site.Rrup = rrup
        site.Rjb = rrup * 0.9
        if type(im).__name__ in ["int", "float"]:
            imt = "SA"
        else:
            imt = im
        for gi, gmms in enumerate([gmms_if, gmms_sl]):
            for g in gmms:
                if type(g).__name__ == "MetaGSIM":
                    if imt not in [
                        x.__name__ for x in g.DEFINED_FOR_INTENSITY_MEASURE_TYPES
                    ]:
                        # model does not support IM
                        continue
                else:
                    if imt not in ["SA", "PGA"]:
                        # empirical engine ones don't have PGV etc...
                        # need to update as required
                        continue
                fault.tect_type = gmms[g][1]
                x = np.linspace(6, 9)
                y = []
                for mag in x:
                    fault.Mw = mag
                    period = None if imt != "SA" else im
                    # but zhao expects a list?
                    if g == GMM.ZA_06:
                        v = compute_gmm(fault, site, g, imt, period=[period])
                        if imt == "PGA":
                            # result
                            v, stdvs = v
                        else:
                            # list of result for each period
                            v, stdvs = v[0]
                    else:
                        v, stdvs = compute_gmm(fault, site, g, imt, period=period)
                    y.append(v[0] if hasattr(v, "__len__") else v)
    
                y = np.array(y)
                plt.loglog(x, y, label=gmms[g][0])
                plt.fill_between(x, y * np.exp(-stdvs[0]), y * np.exp(stdvs[0]), alpha=0.1)
            plt.legend()
            plt.xlabel("Moment magnitude, Mw")
            y = imt
            if imt == "SA":
                y += " " + str(im)
            plt.ylabel(y)
            plt.title("Rrup = " + str(rrup) + " km")
            plt.savefig(f"m{gi}{i}{r}.png")
            plt.close()

# spectra with fixed rrup / magnitude
imt = "SA"
for m, mag in enumerate([8, 9]):
    fault.Mw = mag
    for r, rrup in enumerate([25, 50, 100]):
        site.Rrup = rrup
        site.Rjb = rrup * 0.9
        for gi, gmms in enumerate([gmms_if, gmms_sl]):
            for g in gmms:
                if type(g).__name__ == "MetaGSIM":
                    if imt not in [
                        x.__name__ for x in g.DEFINED_FOR_INTENSITY_MEASURE_TYPES
                    ]:
                        # model does not support IM
                        continue
                fault.tect_type = gmms[g][1]
                x = np.logspace(-2, 1)
                y = []
                for period in x:
                    # but zhao must have a list input
                    if g == GMM.ZA_06:
                        v = compute_gmm(fault, site, g, imt, period=[period])
                        if imt == "PGA":
                            # result
                            v, stdvs = v
                        else:
                            # list of result for each period
                            v, stdvs = v[0]
                    else:
                        v, stdvs = compute_gmm(fault, site, g, imt, period=period)
                    y.append(v[0] if hasattr(v, "__len__") else v)
    
                y = np.array(y)
                plt.loglog(x, y, label=gmms[g][0])
                plt.fill_between(x, y * np.exp(-stdvs[0]), y * np.exp(stdvs[0]), alpha=0.1)
            plt.legend()
            plt.xlabel("Oscillator period (s)")
            y = imt
            if imt == "SA":
                y += " " + str(im)
            plt.ylabel(y)
            plt.title("Mw = " + str(mag)  + " Rrup = " + str(rrup) + " km")
            plt.savefig(f"s{gi}{m}{r}.png")
            plt.close()
