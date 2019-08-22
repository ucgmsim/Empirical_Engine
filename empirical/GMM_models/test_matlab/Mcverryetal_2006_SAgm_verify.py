#!/usr/bin/env python

import numpy as np

import sys
sys.path.append('..')
from McVerry_2006_Sa import McVerry_2006_Sa

# compare with Matlab version

periods = [-1.0, 0.078, 0.14, 0.18, 0.3, 0.4, 0.5, 0.75, 1, 1.5, 2, 3]
mags = [4.8, 6, 10]
rrups = [0, 30, 800]
siteclasses = ['A', 'B', 'C', 'D', 'E']
faultstyles = ["normal", "reverse", "oblique", "strikeslip", "interface", "slab"]
rvols = [82.24, 2.244]
hcs = [4.422, 39.24]

answers = np.fromfile('mv2006.f32', dtype=np.float32)
a = 0

class siteprop:
    period = None
    siteclass = None
    Rrup = None
    rvol = None
class faultprop:
    Mw = None
    faultstyle = None
    Hc = None

for p in periods:
    siteprop.period = p
    for m in mags:
        faultprop.Mw = m
        for s in siteclasses:
            siteprop.siteclass = s
            for r in rrups:
                siteprop.Rrup = r
                for f in faultstyles:
                    faultprop.faultstyle = f
                    for v in rvols:
                        siteprop.rvol = v
                        for h in hcs:
                            faultprop.Hc = h

                            sa, sigma = McVerry_2006_Sa(siteprop, faultprop)
                            assert np.isclose(sa, answers[a])
                            assert np.isclose(sigma, answers[a + 1:a + 4]).all()
                            a += 4
print("All tests passed.")
