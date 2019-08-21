#!/usr/bin/env python

import numpy as np

import sys
sys.path.append('..')
from CY_2014_nga import CY_2014_nga

# compare with Matlab version

periods = [-1, 0.011, 0.022, 0.03, 0.04, 0.05, 0.075, 0.1, 0.12, 0.15, 0.17, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.6, 2.4, 2.8, 4.3, 6, 8, 9]
mags = [4.2, 6.2, 7.2]
rrups = np.array([[12.6, 1552.22], [10, 1000], [3, 24], [2, 20]]).T
ld = np.array([[30, 150, 20, -120, -60], [45, 60, 80, 110, 10]]).T
vs30s = np.array([[67, 485, 2504, 185], [0, 0, 1, 1]]).T
z10 = [None, 3, 243]

answers = np.fromfile('cy2014.f32', dtype=np.float32)
a = 0

for p in periods:
    for m in mags:
        for g in ld:
            for r in rrups:
                for f in range(2):
                    for v in vs30s:
                        for z in z10:
                            for l in range(6):
                                sa, sigma = CY_2014_nga(
                                    m, p, r[0], r[1], r[2], r[3], g[1], g[0], z, v[0], f, v[1], region=l
                                )
                                assert np.isclose(sa, answers[a])
                                assert np.isclose(sigma, answers[a + 1])
                                a += 2
print("All tests passed.")
