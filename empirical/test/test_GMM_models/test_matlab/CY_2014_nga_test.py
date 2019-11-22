#!/usr/bin/env python

import os

import numpy as np

from empirical.GMM_models.CY_2014_nga import CY_2014_nga
from empirical.util.classdef import Fault, Site

# compare with Matlab version

periods = [-1, 0.011, 0.022, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.6, 2.4, 2.8, 4.3, 6, 8, 9]
mags = [4.2, 6.2, 7.2]
rrups = np.array([[12.6, 1552.22], [10, 1000], [3, 24], [2, 20]]).T
ld = np.array([[30, 150, 20, -120, -60], [45, 60, 80, 110, 10]]).T
vs30s = np.array([[67, 485, 2504, 185], [0, 0, 1, 1]]).T
z10 = [None, 3, 243]

answers = np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cy2014.f32'), dtype=np.float32)

site = Site()
fault = Fault()


def test_run():
    a = 0
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
                    fault.ztor = r[3]
                    for f in range(2):
                        for v in vs30s:
                            site.vs30 = v[0]
                            site.vs30measured = v[1]
                            for z in z10:
                                site.z1p0 = z
                                for l in range(6):
                                    sa, sigma = CY_2014_nga(site, fault, im='pSA', period=p, region=l, f_hw=f)
                                    assert np.isclose(sa, answers[a])
                                    assert np.isclose(sigma[0], answers[a + 1])
                                    a += 2


if __name__ == "__main__":
    test_run()
