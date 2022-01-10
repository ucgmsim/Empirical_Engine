import os
from copy import copy
from pathlib import Path
from typing import Iterable

import numpy as np
import six
import yaml

from empirical.util import classdef
from empirical.util.classdef import TectType, GMM, SiteClass, FaultStyle
from empirical.util import openquake_wrapper
from empirical.GMM_models.Abrahamson_2018 import Abrahamson_2018
from empirical.GMM_models.AfshariStewart_2016_Ds import Afshari_Stewart_2016_Ds
from empirical.GMM_models.ASK_2014_nga import ASK_2014_nga
from empirical.GMM_models.bc_hydro_2016_subduction import bc_hydro_2016_subduction
from empirical.GMM_models.Bradley_2010_Sa import Bradley_2010_Sa
from empirical.GMM_models.BSSA_2014_nga import BSSA_2014_nga
from empirical.GMM_models.CampbellBozorgina_2010_CAV_2012_AI import CampbellBozorgina
from empirical.GMM_models.CB_2014_nga import CB_2014_nga
from empirical.GMM_models.CY_2014_nga import CY_2014_nga
from empirical.GMM_models.McVerry_2006_Sa import McVerry_2006_Sa
from empirical.GMM_models.zhou_2006 import Zhaoetal_2006_Sa
from empirical.GMM_models.ShahiBaker_2013_RotD100_50 import ShahiBaker_2013_RotD100_50
from empirical.GMM_models.Burks_Baker_2013_iesdr import Burks_Baker_2013_iesdr
from empirical.GMM_models.meta_model import meta_model
from qcore.constants import Components


DEFAULT_GMM_CONFIG_NAME = "model_config.yaml"
DEFAULT_GMM_WEIGHT_CONFIG_NAME = "gmm_weights.yaml"
DEFAULT_GMM_PARAM_CONFIG_NAME = "gmm_params.yaml"


def iterable_but_not_string(arg):
    """
    :param arg: object
    :return: Returns True if arg is an iterable that isn't a string
    """
    return isinstance(arg, Iterable) and not isinstance(arg, six.string_types)


def read_gmm_weights(emp_weight_conf_ffp=None):
    """
    Reads the weights into a "flat" dictionary
    :param emp_weight_conf_ffp: ffp to yaml configuration file
    :return: dictionary of im, tect-type, model weighting
    """
    if emp_weight_conf_ffp is None:
        emp_weight_conf_ffp = str(
            Path(__file__).parent / DEFAULT_GMM_WEIGHT_CONFIG_NAME
        )
    emp_wc_dict_orig = yaml.load(open(emp_weight_conf_ffp), Loader=yaml.Loader)
    emp_wc_dict = {}

    for ims in emp_wc_dict_orig:
        im_list = ims if iterable_but_not_string(ims) else [ims]
        for im in im_list:
            emp_wc_dict[im] = {}
            for tect_type in emp_wc_dict_orig[ims]:
                tect_type_list = (
                    tect_type if iterable_but_not_string(tect_type) else [tect_type]
                )
                for tt in tect_type_list:
                    if tt not in emp_wc_dict:
                        emp_wc_dict[im][tt] = emp_wc_dict_orig[ims][tect_type]
    return emp_wc_dict


def read_model_dict(config=None):
    if config is None:
        dir = os.path.dirname(__file__)
        config = os.path.join(dir, DEFAULT_GMM_CONFIG_NAME)

    model_dict = yaml.safe_load(open(config))
    return model_dict


def get_gmm_params(config=None):
    if config is None:
        dir = os.path.dirname(__file__)
        config = os.path.join(dir, DEFAULT_GMM_PARAM_CONFIG_NAME)

    oq_model_params_dict = yaml.safe_load(open(config))
    return oq_model_params_dict


def get_models_from_dict(config):
    """
    :param config: yaml
    :return: a list of the unique models present in a configuration file
    """

    if config is None:
        dir = os.path.dirname(__file__)
        config = os.path.join(dir, DEFAULT_GMM_CONFIG_NAME)

    tect_type_model_dict = yaml.safe_load(open(config))
    return list(
        {
            f"{model}_{key}"
            for key in tect_type_model_dict
            for component in tect_type_model_dict[key].values()
            for models in component.values()
            for model in models
        }
    )


def determine_gmm(fault, im, tect_type_model_dict):
    return determine_all_gmm(fault, im, tect_type_model_dict)[0]


def determine_all_gmm(
    fault, im, tect_type_model_dict, components=Components.cgeom.str_value
):
    if fault.tect_type is None:
        print("tect-type not found assuming 'ACTIVE_SHALLOW'")
        tect_type = TectType.ACTIVE_SHALLOW.name
    else:
        tect_type = TectType(fault.tect_type).name
    if tect_type in tect_type_model_dict and im in tect_type_model_dict[tect_type]:
        comps = tect_type_model_dict[tect_type][im]
        models = []
        for comp in comps:
            if (
                comp in components
                and tect_type_model_dict[tect_type][im][comp] is not None
            ):
                for model in tect_type_model_dict[tect_type][im][comp]:
                    models.append((GMM[model], Components.from_str(comp)))
        return models
    else:
        print(
            f"No valid empirical model found for im {im} with tectonic type {tect_type}"
        )
        return []


def compute_gmm(fault, site, gmm, im, period=None, gmm_param_config=None, **kwargs):
    """
    Makes a copy of the fault / site object as some modifications are done.

    NOTE: the site/fault object you pass into this function may be modified before calculation of the GMM

    :return: Mean and standard deviation calculated for a given IM and GMM
    """
    site = copy(site)
    fault = copy(fault)
    gmm_params_dict = get_gmm_params(gmm_param_config)
    if gmm.name in gmm_params_dict.keys():
        tmp_params_dict = gmm_params_dict[gmm.name]
    else:
        tmp_params_dict = {}
    if gmm is GMM.META:
        tmp_params_dict["config"] = gmm_params_dict
    tmp_params_dict.update(kwargs)
    kwargs = tmp_params_dict

    if site.vs30 is None:
        site.vs30 = classdef.VS30_DEFAULT

    if site.Rrup is None and gmm not in [GMM.A_18, GMM.BSSA_14, GMM.SB_13, GMM.BB_13]:
        print("Rrup is a required parameter for", gmm.name)
        exit()

    if site.z1p0 is None:
        site.z1p0 = classdef.estimate_z1p0(site.vs30)

    if site.z2p5 is None:
        site.z2p5 = classdef.estimate_z2p5(z1p0=site.z1p0, z1p5=site.z1p5)

    # openquake models will check dependent parameters dynamically
    # therefore placed before local checking of required parameters
    if type(gmm).__name__ == "MetaGSIM" or gmm in openquake_wrapper.OQ_GMM_LIST:
        return openquake_wrapper.oq_run(gmm, site, fault, im, period, **kwargs)

    if site.vs30measured is None:
        site.vs30measured = False  # assume not measured unless set

    if site.siteclass is None and gmm == GMM.ZA_06:
        site.siteclass = determine_siteclass(site.vs30)

    if site.Rtvz is None or site.Rtvz <= 0 or np.isnan(site.Rtvz):
        if fault.tect_type == classdef.TectType.VOLCANIC:
            site.Rtvz = site.Rrup
        else:
            site.Rtvz = 0

    if site.Rjb is None:
        site.Rjb = np.sqrt(site.Rrup ** 2 - fault.ztor ** 2)

    if fault.Mw is None and gmm not in [GMM.SB_13]:
        print("Moment magnitude is a required parameter")
        exit()

    if fault.rake is None and gmm in [GMM.Br_10, GMM.CB_12]:
        print("rake is a required parameter for", gmm.name)
        exit()

    if fault.dip is None and gmm in [GMM.Br_10, GMM.CB_12]:
        print("dip is a required parameter for", gmm.name)
        exit()

    if fault.ztor is None and gmm in [GMM.A_18, GMM.Br_10, GMM.CB_12]:
        print("ztor is a required parameter for", gmm.name)
        exit()

    # this assumes rake is available but faultstyle and rake are not used in A_18
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
        if gmm in [GMM.A_18, GMM.BCH_16]:
            fault.tect_type = TectType.SUBDUCTION_INTERFACE
        else:
            fault.tect_type = TectType.ACTIVE_SHALLOW

    if fault.hdepth is None and gmm in [GMM.ZA_06, GMM.MV_06]:
        print("hypocentre depth is a required parameter for", gmm.name)
        exit()

    if gmm is GMM.A_18:
        return Abrahamson_2018(site, fault, im=im, periods=period, **kwargs)
    elif gmm is GMM.AS_16:
        return Afshari_Stewart_2016_Ds(site, fault, im)
    elif gmm is GMM.ASK_14:
        return ASK_2014_nga(site, fault, im=im, period=period, **kwargs)
    elif gmm is GMM.BCH_16:
        return bc_hydro_2016_subduction(site, fault, im, period=period)
    elif gmm is GMM.Br_10:
        return Bradley_2010_Sa(site, fault, im, period)
    elif gmm is GMM.BSSA_14:
        return BSSA_2014_nga(site, fault, im=im, period=period, **kwargs)
    elif gmm is GMM.CB_12 or gmm is GMM.CB_10:
        return CampbellBozorgina(site, fault, im)
    elif gmm is GMM.CB_14:
        return CB_2014_nga(site, fault, im=im, period=period, **kwargs)
    elif gmm is GMM.CY_14:
        return CY_2014_nga(site, fault, im=im, period=period, **kwargs)
    elif gmm is GMM.MV_06:
        return McVerry_2006_Sa(site, fault, im=im, periods=period)
    elif gmm is GMM.ZA_06:
        return Zhaoetal_2006_Sa(site, fault, im, period)
    elif gmm is GMM.SB_13:
        return ShahiBaker_2013_RotD100_50(im, period)
    elif gmm is GMM.BB_13:
        return Burks_Baker_2013_iesdr(period, fault, **kwargs)
    elif gmm is GMM.META:
        return meta_model(fault, site, im=im, period=period, **kwargs)
    else:
        raise ValueError("Invalid GMM")


def determine_siteclass(vs30):
    if vs30 <= 200:
        return SiteClass.SOFTSOIL
    elif vs30 <= 300:
        return SiteClass.MEDIUMSOIL
    elif vs30 <= 600:
        return SiteClass.HARDSOIL
    elif vs30 <= 1100:
        return SiteClass.ROCK
    else:
        return SiteClass.HARDROCK
