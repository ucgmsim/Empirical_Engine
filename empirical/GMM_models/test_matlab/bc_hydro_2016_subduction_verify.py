#!/usr/bin/env python

import numpy as np

import sys
sys.path.append('..')
from bc_hydro_2016_subduction import bc_hydro_2016_subduction

# compare with Matlab version


periods = [0.021, 0.06, 0.075, 0.099, 0.14, 0.201, 0.262, 0.3, 0.4, 0.5, 0.6, 0.75, 1, 1.5, 2, 2.5, 3, 4, 5, 6, 7.5, 10]
mags = [7.5, 7.8, 8.0, 8.2]
rrups = [0, 12.6, 90, 1052.22]
vs30s = [67, 485, 2504]
hyps = [131, 0, np.nan]

answers = np.fromfile('bchydro.f32', dtype=np.float32)
a = 0


for p in periods:
    for m in mags:
        for h in hyps:
            for r in rrups:
                for f in range(2):
                    for v in vs30s:
                        sa, sigma = bc_hydro_2016_subduction(p, m, r, f, v, np.isnan(h), h)
                        assert np.isclose(sa, answers[a])
                        assert np.isclose(sigma, answers[a + 1])
                        a += 2
print("All tests passed.")
