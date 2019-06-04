import os
import pickle
TEST_DATA_SAVE_DIR = '/home/melody/Empirical_Engine/pickled/bradley_2013_sa/rrup200'
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

import sys
sys.path.append('../..')

from empirical.GMM_models.Bradley_2013_Sa import Bradley_2013_Sa
from empirical.util.empirical_factory import compute_gmm
from im_processing.computations.GMPE_models.Bradley_2010_Sa import Bradley_2010_Sa

from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM

rrups = [200]
siteclasses = [SiteClass.SOFTSOIL, SiteClass.MEDIUMSOIL, SiteClass.HARDSOIL, SiteClass.ROCK, SiteClass.HARDROCK]
period = [0, 0.01, 0.40370172586, 0.5, 3.0, 8.6974900]
tect_types = [TectType.SUBDUCTION_SLAB, TectType.SUBDUCTION_INTERFACE, TectType.ACTIVE_SHALLOW]
IM = 'pSA'

site = Site()

fault = Fault()
fault.Mw = 10.5
fault.faultstyle = 'interface'
fault.ztor = 0
fault.Ztor = 0
fault.rake = 180
fault.dip = 45

#site.Rrup = 10
site.Rjb = 10
site.vs30 = 500
site.V30 = 500
site.V30measured = None
site.Rx = -1
site.Rtvz = 50

gmm = GMM.Br_13

# with open(os.path.join(TEST_DATA_SAVE_DIR, INPUT_DIR, 'bradley_2013_sa_rrups.P'), 'wb') as save_file:
#     pickle.dump(rrups, save_file)
#
# with open(os.path.join(TEST_DATA_SAVE_DIR, INPUT_DIR, 'bradley_2013_sa_siteclasses.P'), 'wb') as save_file:
#     pickle.dump(siteclasses, save_file)
#
# with open(os.path.join(TEST_DATA_SAVE_DIR, INPUT_DIR, 'bradley_2013_sa_periods.P'), 'wb') as save_file:
#     pickle.dump(period, save_file)
#
# with open(os.path.join(TEST_DATA_SAVE_DIR, INPUT_DIR, 'bradley_2013_sa_tect_types.P'), 'wb') as save_file:
#     pickle.dump(tect_types, save_file)
#
# with open(os.path.join(TEST_DATA_SAVE_DIR, INPUT_DIR, 'bradley_2013_sa_im.P'), 'wb') as save_file:
#     pickle.dump(IM, save_file)
#
# with open(os.path.join(TEST_DATA_SAVE_DIR, INPUT_DIR, 'bradley_2013_sa_fault.P'), 'wb') as save_file:
#     pickle.dump(fault, save_file)
#
# with open(os.path.join(TEST_DATA_SAVE_DIR, INPUT_DIR, 'bradley_2013_sa_gmm.P'), 'wb') as save_file:
#     pickle.dump(gmm, save_file)


for rrup in rrups:
    site.Rrup = rrup
    with open(os.path.join(TEST_DATA_SAVE_DIR, INPUT_DIR, 'bradley_2013_sa_site_rrup{}.P'.format(rrup)), 'wb') as save_file:
        pickle.dump(site, save_file)

    results = compute_gmm(fault, site, gmm, IM, period)
    with open(os.path.join(TEST_DATA_SAVE_DIR, OUTPUT_DIR, 'bradley_2013_sa_results_rrup{}.P'.format(rrup)), 'wb') as save_file:
        pickle.dump(results, save_file)
    # for r in results:
    #     print("new", r)
    # print('-'*90)
#
# print("*"*90)
# for rrup in rrups:
#     site.Rrup = rrup
#     site.Z1pt0 = site.z1p0
#     for p in period:
#         site.period = p
#         print("old",Bradley_2010_Sa(site, fault))
#     print("-"*90)
