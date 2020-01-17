import os

import numpy as np
import pandas as pd

from empirical.GMM_models import Abrahamson_2018 as a18
from empirical.util.classdef import Site
from empirical.util.classdef import Fault
from empirical.util.classdef import TectType

os.mkdir("plots")

###
# Standard Deviation
###

results = {}
for i, p in enumerate(a18.imt[:-1]):
    stdev = a18.compute_stdev(i, -1, 0, 9999)
    results[p] = stdev

df = pd.DataFrame.from_dict(results).T
df.columns = ["total", "tau", "phi"]
ax = df[["tau", "phi"]].plot()
ax.set_xscale("log")
ax.set_xlabel("Period (sec)")
ax.set_ylabel("Std Dev (LN units)")

fig = ax.get_figure()
fig.savefig("plots/std.png")

###
# Magnitude scaling for T=3 sec for intraslab earthquakes. (Rrup=75 km, VS30=760 m/s, ZTOR=50 km)
###

site = Site()
site.vs30 = 760
site.Rrup = 75
fault = Fault()
fault.ztor = 50
fault.tect_type = TectType.SUBDUCTION_SLAB

MAG_RANGE = np.append(np.arange(5, 8.5, 0.1), 8.5)
results = {}
for Mw in MAG_RANGE:
    fault.Mw = Mw
    results[Mw] = a18.Abrahamson_2018(site, fault, "pSA", [3.0])[0][0]
df = pd.DataFrame.from_dict(results, orient="index", columns=["pSA_3s"])
ax = df.plot()
ax.set_yscale("log")
ax.set_xlabel("Magnitude")
ax.set_ylabel("pSA (g)")
ax.set_ylim(0.0001, 1)
ax.grid(True, "both", linestyle="-", color="lightgray")

fig = ax.get_figure()
fig.savefig("plots/mw_sslab_pSA_3s.png")

###
# Magnitude scaling for T=0.2 sec for interface earthquakes. (Rrup=75 km, VS30=760 m/s)
###

site = Site()
site.vs30 = 760
site.Rrup = 75
fault = Fault()
fault.tect_type = TectType.SUBDUCTION_INTERFACE

MAG_RANGE = np.append(np.arange(6, 9.5, 0.1), 9.5)
results = {}
for Mw in MAG_RANGE:
    fault.Mw = Mw
    results[Mw] = a18.Abrahamson_2018(site, fault, "pSA", [0.2])[0][0]
df = pd.DataFrame.from_dict(results, orient="index", columns=["pSA_0s2"])
ax = df.plot()
ax.set_yscale("log")
ax.set_xlabel("Magnitude")
ax.set_ylabel("pSA (g)")
ax.set_ylim(0.01, 1)
ax.grid(True, "both", linestyle="-", color="lightgray")

fig = ax.get_figure()
fig.savefig("plots/mw_sint_pSA_0s2.png")

###
# ZTOR scaling of spectra for M=7 intraslab events
###

site = Site()
site.vs30 = 760
fault = Fault()
fault.tect_type = TectType.SUBDUCTION_SLAB
fault.Mw = 7.0
fault.ztor = 50

rrup_range = np.append(np.arange(40, 1000, 10), 1000)
results = {}
for rrup in rrup_range:
    site.Rrup = rrup
    results[rrup] = a18.Abrahamson_2018(site, fault, "PGA")[0]
df = pd.DataFrame.from_dict(results, orient="index", columns=["PGA"])
ax = df.plot()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("PGA (g)")
ax.set_ylim(0.0001, 1)
ax.grid(True, "both", linestyle="-", color="lightgray")

fig = ax.get_figure()
fig.savefig("plots/rrup_sslab_PGA.png")

###
#
###

site = Site()
site.vs30 = 760
fault = Fault()
fault.tect_type = TectType.SUBDUCTION_INTERFACE
fault.Mw = 9.0

rrup_array = np.array([50, 100, 200, 400, 1000])
# periods = np.linspace(0.01, 10)
results = {}
for rrup in rrup_array:
    results[rrup] = {}
    site.Rrup = rrup
    for period in a18.imt[:-1]:
        results[rrup][period] = a18.Abrahamson_2018(site, fault, "pSA", [period])[0][0]
df = pd.DataFrame.from_dict(results)
ax = df.plot()
ax.set_yscale("log")
ax.set_xscale("log")
ax.set_xlabel("Period (sec)")
ax.set_ylabel("pSA (g)")
ax.set_ylim(0.000001, 1)
ax.set_title("Subduction Interface")
ax.grid(True, "both", linestyle="-", color="lightgray")

fig = ax.get_figure()
fig.savefig("plots/rrup_s_pSA_scaling.png")
