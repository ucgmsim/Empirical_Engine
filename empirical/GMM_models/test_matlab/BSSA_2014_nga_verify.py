#!/usr/bin/env python

import numpy as np

import sys
sys.path.append('..')
from BSSA_2014_nga import BSSA_2014_nga

# compare with Matlab version

periods = [-1, 0.011, 0.02, 0.035, 0.06, 0.066, 0.9, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3, 4, 5, 7.5, 10];
mags = [4.4, 5, 5.5, 9.2];
rrups = [0, 98, 2345];
regions = [0, 1, 2, 3, 4];
vs30s = [134, 760, 1357];
z1 = [None, 3, 243];

answers = np.fromfile('bssa2014.f32', dtype=np.float32)
a = 0

for p in periods:
    for m in mags:
        for r in rrups:
            for t in range(4):
                for v in vs30s:
                    for z in z1:
                        for l in regions:
                            sa, sigma = BSSA_2014_nga(m, p, r, t, region=l, z1=z, vs30=v)
                            assert np.isclose(sa, answers[a])
                            assert np.isclose(sigma, answers[a + 1])
                            a += 2
print("All tests passed.")
