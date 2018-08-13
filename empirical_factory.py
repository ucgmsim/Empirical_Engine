import argparse
from GMM_models.classdef import Site, Fault
from GMM_models.classdef import TectType
from GMM_models.classdef import GMM
from GMM_models.classdef import SiteClass
from GMM_models.classdef import FaultStyle
from GMM_models.Bradley_2013_Sa import Bradley_2013_Sa
from GMM_models.zhou_2006 import Zhaoetal_2006_Sa
from GMM_models.AfshariStewart_2016_Ds import Afshari_Stewart_2016_Ds
from GMM_models.CampbellBozorgina_2012_AI import CampbellBozorgina_2012
from GMM_models import classdef
import numpy as np
import yaml
import os


def read_model_dict(config=None):
    if config is None:
        dir = os.path.dirname(__file__)
        config_file = os.path.join(dir, 'model_config.yaml')
    else:
        config_file = config
    model_dict = yaml.load(open(config_file))
    return model_dict


def determine_gmm(fault, im, model_dict):
    if fault.tect_type is None:
        print "tect-type not found assuming 'ACTIVE_SHALLOW'"
        tect_type = TectType.ACTIVE_SHALLOW.name
    else:
        tect_type = fault.tect_type.name

    if tect_type in model_dict and im in model_dict[tect_type]:
        model = model_dict[tect_type][im]
        return GMM[model]
    else:
        print "No valid empirical model found"
        return None


def compute_gmm(fault, site, gmm, im, period=None):
    if site.vs30 is None:
        site.vs30 = classdef.VS30_DEFAULT

    if site.Rrup is None:
        print "Rrup is a required parameter"
        exit()

    if site.z1p0 is None:
        site.z1p0 = estimate_z1p0(site.vs30)

    if site.z2p5 is None:
        site.z2p5 = estimate_z2p5(z1p0=site.z1p0, z1p5=site.z1p5)

    if site.vs30measured is None:
        site.vs30measured = False  # assume not measured unless set

    if site.siteclass is None:
        site.siteclass = determine_siteclass(site.vs30)

    if site.Rtvz is None:
        site.Rtvz = 0

    if site.Rjb is None:
        site.Rjb = np.sqrt(site.rrup ^ 2 - fault.ztor ^ 2)

    if site.Rx is None:
        site.Rx = -site.Rjb  # incorrect assumption but keeping for legacy reasons

    if fault.Mw is None:
        print "Moment magnitude is a required parameter"
        exit()

    if fault.rake is None and GMM in [GMM.Br_13, GMM.CB_12]:
        print "rake is a required parameter for Br_13 and CB_12"
        exit()

    if fault.dip is None and GMM in [GMM.Br_13, GMM.CB_12]:
        print "dip is a required parameter for Br_13 and CB_12"
        exit()

    if fault.ztor is None and GMM in [GMM.Br_13, GMM.CB_12]:
        print "ztor is a required parameter for Br_13 and CB_12"
        exit()

    if fault.rupture_type is None:
        if 45 < fault.rake < 135:
            fault.rupture_type = FaultStyle.REVERSE
        elif -135 < fault.rake < -45:
            fault.rupture_type = FaultStyle.NORMAL
        elif 0 < abs(fault.rake) < 45 or 135 < abs(fault.rake) < 180:
            fault.rupture_type = FaultStyle.STRIKESLIP
        else:
            fault.rupture_type = FaultStyle.UNKNOWN

    if fault.tect_type is None:
        fault.tect_type = TectType.ACTIVE_SHALLOW

    if fault.hdepth is None and GMM == GMM.ZA_06:
        print "hypocentre depth is a required parameter for ZA06"
        exit()

    value = None

    if gmm is GMM.Br_13:
        value = Bradley_2013_Sa(site, fault, im, period)
    elif gmm is GMM.AS_16:
        value = Afshari_Stewart_2016_Ds(site, fault, im)
    elif gmm is GMM.CB_12:
        value = CampbellBozorgina_2012(site, fault, im)
    elif gmm is GMM.ZA_06:
        value = Zhaoetal_2006_Sa(site, fault, im, period)

    return value


def determine_siteclass(vs30):
    if vs30 < 200:
        siteclass = SiteClass.SOFTSOIL
    elif vs30 < 300:
        siteclass = SiteClass.MEDIUMSOIL
    elif vs30 < 600:
        siteclass = SiteClass.HARDSOIL
    elif vs30 < 1100:
        siteclass = SiteClass.ROCK
    else:
        siteclass = SiteClass.HARDROCK

    return siteclass


# CB08 estimate of Z2p5
def estimate_z2p5(z1p0=None, z1p5=None):
    if z1p5 is not None:
        return 0.636 + 1.549 * z1p5
    elif z1p0 is not None:
        return 0.519 + 3.595 * z1p0
    else:
        print 'no z2p5 able to be estimated'
        exit()


def estimate_z1p0(vs30):
    return np.exp(28.5 - 3.82 / 8.0 * np.log(vs30 ** 8 + 378.7 ** 8)) / 1000.0  # CY08 estimate in KM
