
import sys

sys.path.append('../..')

from empirical.GMM_models.zhou_2006 import Zhaoetal_2006_Sa
from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM
from empirical.util.empirical_factory import compute_gmm

# rrups = [10, 70, 200]
# siteclasses = [SiteClass.SOFTSOIL, SiteClass.MEDIUMSOIL, SiteClass.HARDSOIL, SiteClass.ROCK, SiteClass.HARDROCK]
# period = [0, 0.01, 0.5, 3.0, 10.0]
# tect_types = [TectType.SUBDUCTION_SLAB, TectType.SUBDUCTION_INTERFACE, TectType.ACTIVE_SHALLOW]

site = Site()

fault = Fault()
fault.Mw = 7.2
fault.faultstyle = 'interface'
fault.hdepth = 0
fault.ztor = 0
fault.rake = 0
fault.dip = 0

# print("period, rrup, siteclass, tect_type, mean, std_total")
# for tect_type in tect_types:
#     for siteclass in siteclasses:
#         for rrup in rrups:
#             fault.tect_type = tect_type
#             site.Rrup = rrup
#             site.siteclass = siteclass
#             #results = compute_gmm(fault, site, GMM.ZA_06, 'pSA', period)
#             results = Zhaoetal_2006_Sa(site, fault, 'pSA', period)
#             for result, p in zip(results, period):
#                 mean, std = result
#                 print(p, rrup, siteclass, tect_type, mean, std[0])
#         print("")
tect_types={'TectType.SUBDUCTION_SLAB':TectType.SUBDUCTION_SLAB, 'TectType.SUBDUCTION_INTERFACE':TectType.SUBDUCTION_INTERFACE,'TectType.ACTIVE_SHALLOW':TectType.ACTIVE_SHALLOW}
siteclasses={'SiteClass.SOFTSOIL':SiteClass.SOFTSOIL, 'SiteClass.MEDIUMSOIL':SiteClass.MEDIUMSOIL, 'SiteClass.HARDSOIL':SiteClass.HARDSOIL,'SiteClass.ROCK':SiteClass.ROCK, 'SiteClass.HARDROCK':SiteClass.HARDROCK}
import pandas as pd

benchmark = '/home/melody/Downloads/Zhao_Test_Cases_2.xlsx'

df = pd.read_excel(benchmark)
# resultss = []
# import pickle
for index, row in df.iterrows():
    if row.name == 194:
        fault.hdepth = 10
    if not row.isnull().any():
        fault.tect_type = tect_types[row['tect_type']]
        site.Rrup = float(row['rrup'])
        site.siteclass = siteclasses[row['siteclass']]
        results = compute_gmm(fault, site, GMM.ZA_06, 'pSA', [row['period']])
        print("results", results)
#         resultss.append(results)
# with open('/home/melody/Empirical_Engine/pickled/zhao_2006/output/zhao_2006_ret_val.P', 'wb') as f:
#     pickle.dump(resultss, f)
        # for mean, std in results:
        #
        #     if "{:.4f}".format(mean)!="{:.4f}".format(row['mean']):
        #         print(row['period'], site.Rrup, site.siteclass, fault.tect_type, "{:.4f}".format(mean), "{:.4f}".format(row['mean']))
        #     if "{:.4f}".format(std[0])!="{:.4f}".format(row['std_total']):
        #         print(row['period'], site.Rrup, site.siteclass, fault.tect_type,"{:.4f}".format(std[0]), "{:.4f}".format(row['std_total']))

# # H=10!
# fault.tect_type=TectType.SUBDUCTION_SLAB
# site.Rrup = float(10)
# site.siteclass = SiteClass.SOFTSOIL
# print(compute_gmm(fault, site, GMM.ZA_06, 'pSA', [0]))