#!/usr/bin/env python

import numpy as np

import sys
sys.path.append('..')
from ASK_2014_nga import ASK_2014_nga

# compare with Matlab version

periods = [0.012, 0.018, 0.03, 0.05, 0.07, 0.1, 0.16, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 6.5, 8, 10, -1]
mags = [4.8, 5, 6.5, 9.2]
rrups = np.array([[30, 80], [29, 30], [12, 103], [999, 20], [8, 18], [23, 68]]).T
regions = [0, 2, 3, 6]
ld = np.array([[29, 30, -120], [0, 45, 25]]).T
vs30s = np.array([[67, 485, 2504, 185], [0, 0, 1, 1]]).T
z10 = [None, 3, 243]

answers = np.fromfile('ask2014.f32', dtype=np.float32)
a = 0

for p in periods:
    for m in mags:
        for g in ld:
            for r in rrups:
                for f in range(2):
                    for h in range(2):
                        for v in vs30s:
                            for z in z10:
                                for l in regions:
                                    sa, sigma = ASK_2014_nga(m, p, r[0], r[1], r[2], r[3], g[1], g[0], ztor=r[4], f_as=f, f_hw=h, w=r[5], z10=z, vs30=v[0], f_vs30=v[1], region=l)
                                    assert np.isclose(sa, answers[a])
                                    assert np.isclose(sigma, answers[a + 1])
                                    a += 2
print("All tests passed.")
