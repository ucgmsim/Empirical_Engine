
import sys

sys.path.append('../..')

from empirical.GMM_models.zhou_2006 import Zhaoetal_2006_Sa
from empirical.util.classdef import Site, Fault, TectType, SiteClass

rrups = [10, 70, 200]
siteclasses = [SiteClass.SOFTSOIL, SiteClass.MEDIUMSOIL, SiteClass.HARDSOIL, SiteClass.ROCK, SiteClass.HARDROCK]
period = [0, 0.01, 0.5, 3.0, 10.0]
tect_types = [TectType.SUBDUCTION_SLAB, TectType.SUBDUCTION_INTERFACE, TectType.ACTIVE_SHALLOW]

site = Site()

fault = Fault()
fault.Mw = 7.2
fault.faultstyle = 'interface'
fault.h = 10

print("period, rrup, siteclass, tect_type, mean, std_total")
for tect_type in tect_types:
    for siteclass in siteclasses:
        for rrup in rrups:
            fault.tect_type = tect_type
            site.Rrup = rrup
            site.siteclass = siteclass
            results = Zhaoetal_2006_Sa(site, fault, 'pSA', period)
            for result, p in zip(results, period):
                mean, std = result
                print(p, rrup, siteclass, tect_type, mean, std[0])
        print("")
