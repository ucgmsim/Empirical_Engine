import argparse
from classdef import Site, Fault
from classdef import TectType
from classdef import GMM
from GMM_models.Bradley_2010_Sa import Bradley_2010_Sa
from GMM_models.zhou_2006 import Zhaoetal_2006_Sa
from GMM_models.AfshariStewart_2016_Ds import Afshari_Stewart_2016_Ds
from GMM_models.CampbellBozorgina_2012_AI import CampbellBozorgina_2012


def determine_gmm(fault, im):
    if fault.tect_type is None:
        print "tect-type not found assuming 'ACTIVE_SHALLOW'"
        tect_type = TectType.ACTIVE_SHALLOW
    else:
        tect_type = fault.tect_type

    if tect_type in [TectType.ACTIVE_SHALLOW, TectType.VOLCANIC, TectType.OTHER_CRUSTAL_FAULTING]:
        if im in ['PGA', 'PGV', 'pSA']:
            return GMM.Br_13
    elif tect_type in [TectType.INTERFACE_FAULTING, TectType.SUBDUCTION_INTERFACE]:
        if im in ['PGA', 'pSA']:
            return GMM.ZA_06
    elif tect_type is TectType.ACTIVE_SHALLOW:
        if im in ['Ds575', 'Ds595']:
            return GMM.AS_16
        if im in ['CAV', 'AI']:
            return GMM.CB_12
    else:
        print "No valid empirical model found"


def compute_gmm(fault, site, gmm, im, period=None):
    if im is 'PGA':
        period = 0
    if im is 'PGV':
        period = -1

    if gmm is GMM.Br_13:
        Bradley_2010_Sa(site, fault, period)
    elif gmm is GMM.AS_16:
        Afshari_Stewart_2016_Ds(site, fault, im)
    elif gmm is GMM.CB_12:
        CampbellBozorgina_2012(site, fault, im)
    elif gmm is GMM.ZA_06:
        Zhaoetal_2006_Sa(site, fault, im, period)


def test():
    fault = Fault()
    fault.Mw = 7.18
    fault.Ztor = 0.0
    fault.dip = 35.0
    fault.rake = 90.0
    fault.rupture_type = 'R'
    fault.faultstyle = 'reverse'
    fault.h = 15

    print determine_gmm(fault, 'PGA')


if __name__ == '__main__':
    test()