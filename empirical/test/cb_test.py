import sys

sys.path.append('../..')

from empirical.util.empirical_factory import compute_gmm

from empirical.util.classdef import Site, Fault, GMM
M = [4.0, 5.4, 7.8]
RRUP = [10, 70, 200]
IMS = ['CAV', 'AI']
site = Site()

fault = Fault()
fault.faultstyle = 'SHALLOWCRUSTAL'
fault.ztor = 0
fault.Ztor = 0
fault.rake = 180
fault.dip = 45

site.Rjb = 10
site.vs30 = 500
site.V30 = 500
site.V30measured = None
site.Rx = -1

site.Rtvz = 50
site.z2p5 = 0.9186718412435146

for im in IMS:
    print(im)
    for rrup in RRUP:
        site.Rrup = rrup
        for mag in M:
            fault.Mw = mag
            results = compute_gmm(fault, site, GMM.CB_12, im)
            print(results)
            print('\n')