import sys

sys.path.append('../..')

from empirical.util.empirical_factory import compute_gmm

from empirical.util.classdef import Site, Fault, GMM

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


M = [4.0, 5.4, 7.8]
RRUP = [10, 70, 200]
CB_IMS = ['CAV', 'AI']
AS_IMS = ['Ds575','Ds595', 'Ds2080']
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

# for im in CB_IMS:
#     print(im)
#     for rrup in RRUP:
#         site.Rrup = rrup
#         for mag in M:
#             fault.Mw = mag
#             results = compute_gmm(fault, site, GMM.CB_12, im)
#             print(results)
#             print('\n')
M2 = [5, 6.25, 7.5]
d = {}
for mag in M2:
    fault.Mw = mag
    d[mag] = {}
    for im in AS_IMS:
        print(im)
        d[mag][im] = []
        for rrup in RRUP:
            site.Rrup = rrup
            results = compute_gmm(fault, site, GMM.AS_16, im)
            d[mag][im].append(results[0])
            print(results)
            print('\n')
print(d)

for k in AS_IMS:
    plt.plot(RRUP, d[6.25][k])
plt.legend(AS_IMS)
plt.savefig('/home/melody/astest')
