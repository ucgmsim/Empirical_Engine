#!/usr/bin/env python

import numpy as np

from empirical.GMM_models.bc_hydro_2016_subduction import bc_hydro_2016_subduction
from empirical.util.classdef import Fault, Site, TectType

# compare with Matlab version


periods = [0.021, 0.06, 0.075, 0.099, 0.14, 0.201, 0.262, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 10]
mags = [7.5, 7.8, 8.0, 8.2]
rrups = [0, 12.6, 90, 1052.22]
vs30s = [67, 485, 2504]
hyps = [131, 0, np.nan]
tect_types = [TectType.SUBDUCTION_INTERFACE, TectType.SUBDUCTION_SLAB]

answers = np.fromfile('bchydro.f32', dtype=np.float32)
a = 0

site = Site()
fault = Fault()

for p in periods:
    for m in mags:
        fault.Mw = m
        for h in hyps:
            fault.hdepth = h
            fault.tect_type = tect_types[np.isnan(h)]
            for r in rrups:
                site.Rrup = r
                for f in range(2):
                    site.backarc = f
                    for v in vs30s:
                        site.vs30 = v
                        sa, sigma = bc_hydro_2016_subduction(site, fault, p)
                        assert np.isclose(sa, answers[a])
                        assert np.isclose(sigma, answers[a + 1])
                        a += 2
print("All tests passed.")
