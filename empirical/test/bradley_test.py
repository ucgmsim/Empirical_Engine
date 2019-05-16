import sys

sys.path.append('../..')

from empirical.GMM_models.Bradley_2013_Sa import Bradley_2013_Sa
from empirical.util.empirical_factory import compute_gmm
from im_processing.computations.GMPE_models.Bradley_2010_Sa import Bradley_2010_Sa

from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM

rrups = [10, 70, 200]
siteclasses = [SiteClass.SOFTSOIL, SiteClass.MEDIUMSOIL, SiteClass.HARDSOIL, SiteClass.ROCK, SiteClass.HARDROCK]
period = [0, 0.01, 0.40370172586, 0.5, 3.0, 8.6974900]
tect_types = [TectType.SUBDUCTION_SLAB, TectType.SUBDUCTION_INTERFACE, TectType.ACTIVE_SHALLOW]

site = Site()

fault = Fault()
fault.Mw = 10.5
fault.faultstyle = 'interface'
fault.ztor = 0
fault.Ztor = 0
fault.rake = 180
fault.dip = 45


site.Rrup = 10
site.Rjb = 10
site.vs30 = 500
site.V30 = 500
site.V30measured = None
site.Rx = -1
site.Rtvz = 50

for rrup in rrups:
    site.Rrup = rrup
    results = compute_gmm(fault, site, GMM.Br_13, 'pSA', period)
    for r in results:
        print("new", r)
    print('-'*90)

print("*"*90)
for rrup in rrups:
    site.Rrup = rrup
    site.Z1pt0 = site.z1p0
    for p in period:
        site.period = p
        print("old",Bradley_2010_Sa(site, fault))
    print("-"*90)
