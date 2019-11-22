#!/usr/bin/env python

import os

import numpy as np

from empirical.GMM_models.BSSA_2014_nga import BSSA_2014_nga
from empirical.util.classdef import Fault, Site, FaultStyle

# compare with Matlab version

periods = [-1, 0.011, 0.02, 0.035, 0.06, 0.066, 0.9, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10]
mags = [4.4, 5, 5.5, 9.2]
rrups = [0, 98, 2345]
regions = [0, 1, 2, 3, 4]
vs30s = [134, 760, 1357]
z1 = [None, 3, 243]
fault_types = FaultStyle.UNKNOWN, FaultStyle.STRIKESLIP, FaultStyle.NORMAL, FaultStyle.REVERSE

answers = np.fromfile(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bssa2014.f32'), dtype=np.float32)

site = Site()
fault = Fault()

def test_run():
    a = 0
    for p in periods:
        for m in mags:
            fault.Mw = m
            for r in rrups:
                site.Rjb = r
                for t in fault_types:
                    fault.faultstyle = t
                    for v in vs30s:
                        site.vs30 = v
                        for z in z1:
                            site.z1p0 = z
                            for l in regions:
                                sa, sigma = BSSA_2014_nga(site, fault, im="pSA", period=p, region=l)
                                assert np.isclose(sa, answers[a])
                                assert np.isclose(sigma[0], answers[a + 1])  # only tests total sigma
                                a += 2

if __name__ == "__main__":
    test_run()
