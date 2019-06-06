import sys
sys.path.append('../..')

from empirical.util.empirical_factory import compute_gmm
from empirical.util.classdef import Site, Fault, GMM

fault = Fault()
fault.faultstyle = 'SHALLOWCRUSTAL'
fault.ztor = 0
fault.Ztor = 0
fault.rake = 180
fault.dip = 45

site = Site()
site.Rjb = 10
site.vs30 = 500
site.V30 = 500
site.V30measured = None
site.Rx = -1

site.Rtvz = 50
site.z2p5 = 0.9186718412435146

RRUP = [10, 70, 200]

# TEST FOR CB
CB_M = [4.0, 5.4, 7.8]
CB_IMS = ['CAV', 'AI']

import pickle


# for im in CB_IMS:
#     print(im)
#     for mag in CB_M:
#         fault.Mw = mag
#         for rrup in RRUP:
#             site.Rrup = rrup
#             results = compute_gmm(fault, site, GMM.CB_12, im)
#             print("rrup, mag", rrup, mag, results)
#
#             with open('/home/melody/Empirical_Engine/pickled/cb_2012/output/cb_2012_ret_val_rrup_{}_mag_{}_{}.P'.format(rrup, mag, im), 'wb') as f:
#                 pickle.dump(results, f)
# TEST FOR AS
# import matplotlib
# matplotlib.use('agg')
# import matplotlib.pyplot as plt

AS_M = [5, 6.25, 7.5]
AS_IMS = ['Ds575', 'Ds595', 'Ds2080']
PHI1 = [0.54, 0.43, 0.56]
PHI2 = [0.41, 0.35, 0.45]
d = {}
for mag in AS_M:
    fault.Mw = mag
    d[mag] = {}
    for im in AS_IMS:
        d[mag][im] = []
        for rrup in RRUP:
            site.Rrup = rrup
            results = compute_gmm(fault, site, GMM.AS_16, im)
            with open('/home/melody/Empirical_Engine/pickled/as_2016/output/as_2016_ret_val_rrup_{}_mag_{}_{}.P'.format(
                    rrup, mag, im), 'wb') as f:
                pickle.dump(results, f)
            d[mag][im].append(results)
            print(results)
            print('\n')

# print(d)

# # path duration vs rrup
# for mag in AS_M:
#     for im in AS_IMS:
#         path_durations = [result[0] for result in d[mag][im]]
#         print("path durations", path_durations)
#         plt.plot(RRUP, path_durations)
#     plt.legend(AS_IMS)
#     print(mag)
#     plt.savefig('/home/melody/astest_pathduration_rrup_mag{}.png'.format(mag))
#     plt.clf()
#
#
# # phi vs magnitude
# for im in AS_IMS:
#     phis = [d[mag][im][RRUP[0]][-1][-1] for mag in AS_M]
#     print("rrup, phis",RRUP[0], phis)
#     plt.plot(AS_M, phis)
# plt.legend(AS_IMS)
# plt.savefig('/home/melody/astest_phi_mag.png')
#

