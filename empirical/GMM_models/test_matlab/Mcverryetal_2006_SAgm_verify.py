#!/usr/bin/env python

import numpy as np

from empirical.GMM_models.McVerry_2006_Sa import McVerry_2006_Sa
from empirical.util.classdef import Fault, FaultStyle, Site, SiteClass

# compare with Matlab version

periods = [-1.0, 0.078, 0.14, 0.18, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3]
mags = [4.8, 6, 10]
rrups = [0, 30, 800]
faultstyles = [FaultStyle.NORMAL, FaultStyle.REVERSE, FaultStyle.OBLIQUE, FaultStyle.STRIKESLIP, FaultStyle.INTERFACE, FaultStyle.SLAB]
rtvzs = [82.24, 2.244]
hcs = [4.422, 39.24]

answers = np.fromfile('mv2006.f32', dtype=np.float32)
a = 0

site = Site()
fault = Fault()

for p in periods:
    for m in mags:
        fault.Mw = m
        for s in SiteClass:
            site.siteclass = s
            for r in rrups:
                site.Rrup = r
                for f in faultstyles:
                    fault.faultstyle = f
                    for v in rtvzs:
                        site.Rtvz = v
                        for h in hcs:
                            fault.hdepth = h

                            sa, sigma = McVerry_2006_Sa(site, fault, period=p)
                            assert np.isclose(sa, answers[a])
                            assert np.isclose(sigma, answers[a + 1:a + 4]).all()
                            a += 4
print("All tests passed.")
