#!/usr/bin/env python

import numpy as np

from empirical.GMM_models.CB_2014_nga import CB_2014_nga
from empirical.util.classdef import Fault, Site

# compare with Matlab version

periods = [0.013, 0.024, 0.032, 0.045, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10, -1]
mags = [4.8, 5.5, 6.5, 9.2]
rrups = np.array([[30, 80], [29, 30], [12, 103], [None, 20]]).T
regions = [0, 2, 3, 4]
ld = np.array([[29, 30, -120], [0, 45, 25]]).T
vs30s = [678, 485, 2504, 145]
z25 = [None, 32]
zs = np.array([[None, 17, 17, 18], [18, 20, 21, 22], [None, 5, 25, 35]]).T

answers = np.fromfile('cb2014.f32', dtype=np.float32)
a = 0

site = Site()
fault = Fault()

for p in periods:
    for m in mags:
        fault.Mw = m
        for g in ld:
            fault.rake = g[0]
            fault.dip = g[1]
            for r in rrups:
                site.Rrup = r[0]
                site.Rjb = r[1]
                site.Rx = r[2]
                fault.width = r[3]
                for f in range(2):
                    for h in zs:
                        fault.ztor = h[0]
                        fault.zbot = h[1]
                        fault.hdepth = h[2]
                        for v in vs30s:
                            site.vs30 = v
                            for z in z25:
                                site.z2p5 = z
                                for l in regions:
                                    sa, sigma = CB_2014_nga(site, fault, period=p, region=l, f_hw=f)
                                    assert np.isclose(sa, answers[a])
                                    assert np.isclose(sigma, answers[a + 1])
                                    a += 2
print("All tests passed.")