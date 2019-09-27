from empirical.util import classdef
from empirical.util.classdef import TectType, GMM, SiteClass, FaultStyle
from empirical.GMM_models.AfshariStewart_2016_Ds import Afshari_Stewart_2016_Ds
from empirical.GMM_models.ASK_2014_nga import ASK_2014_nga
from empirical.GMM_models.bc_hydro_2016_subduction import bc_hydro_2016_subduction
from empirical.GMM_models.Bradley_2013_Sa import Bradley_2013_Sa
from empirical.GMM_models.BSSA_2014_nga import BSSA_2014_nga
from empirical.GMM_models.CampbellBozorgina_2012_AI import CampbellBozorgina_2012
from empirical.GMM_models.CB_2014_nga import CB_2014_nga
from empirical.GMM_models.CY_2014_nga import CY_2014_nga
from empirical.GMM_models.McVerry_2006_Sa import McVerry_2006_Sa
from empirical.GMM_models.zhou_2006 import Zhaoetal_2006_Sa
import numpy as np
import yaml
import os


def read_model_dict(config=None):
    if config is None:
        dir = os.path.dirname(__file__)
        config_file = os.path.join(dir, "model_config.yaml")
    else:
        config_file = config
    model_dict = yaml.safe_load(open(config_file))
    return model_dict


def get_models_from_dict(config):
    """
    :param config: yaml
    :return: a list of the unique models present in a configuration file
    """
    model_dict = yaml.safe_load(open(config))
    return list(set([model for im_model_dict in model_dict.values() for models in im_model_dict.values() for model in models]))


def determine_gmm(fault, im, model_dict):
    if fault.tect_type is None:
        print("tect-type not found assuming 'ACTIVE_SHALLOW'")
        tect_type = TectType.ACTIVE_SHALLOW.name
    else:
        tect_type = fault.tect_type.name

    if tect_type in model_dict and im in model_dict[tect_type]:
        model = model_dict[tect_type][im]
        return GMM[model[0]]
    else:
        print("No valid empirical model found")
        return None


def compute_gmm(fault, site, gmm, im, period=None):
    if site.vs30 is None:
        site.vs30 = classdef.VS30_DEFAULT

    if site.Rrup is None and gmm not in [GMM.BSSA_14]:
        print("Rrup is a required parameter for", gmm.name)
        exit()

    if site.z1p0 is None:
        site.z1p0 = classdef.estimate_z1p0(site.vs30)

    if site.z2p5 is None:
        site.z2p5 = classdef.estimate_z2p5(z1p0=site.z1p0, z1p5=site.z1p5)

    if site.vs30measured is None:
        site.vs30measured = False  # assume not measured unless set

    if site.siteclass is None:
        site.siteclass = determine_siteclass(site.vs30)

    if site.Rtvz is None:
        site.Rtvz = 0

    if site.Rjb is None:
        site.Rjb = np.sqrt(site.Rrup ** 2 - fault.ztor ** 2)

    if site.Rx is None:
        site.Rx = -site.Rjb  # incorrect assumption but keeping for legacy reasons

    if fault.Mw is None:
        print("Moment magnitude is a required parameter")
        exit()

    if fault.rake is None and gmm in [GMM.Br_13, GMM.CB_12]:
        print("rake is a required parameter for Br_13 and CB_12")
        exit()

    if fault.dip is None and gmm in [GMM.Br_13, GMM.CB_12]:
        print("dip is a required parameter for Br_13 and CB_12")
        exit()

    if fault.ztor is None and gmm in [GMM.Br_13, GMM.CB_12]:
        print("ztor is a required parameter for Br_13 and CB_12")
        exit()

    if fault.faultstyle is None:
        if 45 < fault.rake < 135:
            fault.faultstyle = FaultStyle.REVERSE
        elif -135 < fault.rake < -45:
            fault.faultstyle = FaultStyle.NORMAL
        elif 0 < abs(fault.rake) < 45 or 135 < abs(fault.rake) < 180:
            fault.faultstyle = FaultStyle.STRIKESLIP
        else:
            fault.faultstyle = FaultStyle.UNKNOWN

    if fault.tect_type is None:
        if gmm is GMM.BC_16:
            fault.tect_type = TectType.SUBDUCTION_INTERFACE
        else:
            fault.tect_type = TectType.ACTIVE_SHALLOW

    if fault.hdepth is None and gmm in [GMM.ZA_06, GMM.MV_06]:
        print("hypocentre depth is a required parameter for", gmm.name)
        exit()

    if gmm is GMM.AS_16:
        return Afshari_Stewart_2016_Ds(site, fault, im)
    elif gmm is GMM.ASK_14:
        return ASK_2014_nga(site, fault, im=im, period=period)
    elif gmm is GMM.BC_16:
        return bc_hydro_2016_subduction(site, fault, period=period)
    elif gmm is GMM.Br_13:
        return Bradley_2013_Sa(site, fault, im, period)
    elif gmm is GMM.BSSA_14:
        return BSSA_2014_nga(site, fault, im=im, period=period)
    elif gmm is GMM.CB_12:
        return CampbellBozorgina_2012(site, fault, im)
    elif gmm is GMM.CB_14:
        return CB_2014_nga(site, fault, im=im, period=period)
    elif gmm is GMM.CY_14:
        return CY_2014_nga(site, fault, im=im, period=period)
    elif gmm is GMM.MV_06:
        return McVerry_2006_Sa(site, fault, im=im, period=period)
    elif gmm is GMM.ZA_06:
        return Zhaoetal_2006_Sa(site, fault, im, period)
    else:
        raise ValueError("Invalid GMM")


def determine_siteclass(vs30):
    if vs30 < 200:
        return SiteClass.SOFTSOIL
    elif vs30 < 300:
        return SiteClass.MEDIUMSOIL
    elif vs30 < 600:
        return SiteClass.HARDSOIL
    elif vs30 < 1100:
        return SiteClass.ROCK
    else:
        return SiteClass.HARDROCK
