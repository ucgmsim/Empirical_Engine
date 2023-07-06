from typing import Union

import numpy as np
from scipy.special import erf

from empirical.util.classdef import GMM


def kuehn_20_calc_z(vs30: float, region: str):
    """
    Calculates the z1p0 or z2p5 value for the Kuehn et al. (2020) model
    Depends on the region for z1p0 or z2p5

    Parameters
    ----------
    vs30 : float
        The Vs30 value, in meters per second
    region : str
        The region to use, must be one of ["Cascadia", "Japan", "NewZealand", "Taiwan"]

    Returns
    -------
    float
        z1p0, in km if region is ["NewZealand", "Taiwan"]
        z2p5, in km if region is ["Cascadia", "Japan"]
    """
    # Basin depth model parameters for each of the regions for which a
    # basin response model is defined
    z_model = {
        "Cascadia": {
            "c_z_1": 8.294049640102028,
            "c_z_2": 2.302585092994046,
            "c_z_3": 6.396929655216146,
            "c_z_4": 0.27081458999999997,
        },
        "Japan": {
            "c_z_1": 7.689368537500001,
            "c_z_2": 2.302585092994046,
            "c_z_3": 6.309186400000001,
            "c_z_4": 0.7528670225000001,
        },
        "NewZealand": {
            "c_z_1": 6.859789675000001,
            "c_z_2": 2.302585092994046,
            "c_z_3": 5.745692775,
            "c_z_4": 0.91563524375,
        },
        "Taiwan": {
            "c_z_1": 6.30560665,
            "c_z_2": 2.302585092994046,
            "c_z_3": 6.1104992125,
            "c_z_4": 0.43671101999999995,
        },
    }

    try:
        constants = z_model[region]
    except KeyError:
        raise KeyError(f"Region {region} not supported for Kuehn et al. (2020)")
    diff = (np.log(vs30) - constants["c_z_3"]) / constants["c_z_4"]
    ln_z_ref = constants["c_z_1"] + (constants["c_z_2"] - constants["c_z_1"]) * np.exp(
        diff
    ) / (1 + np.exp(diff))
    return ln_z_ref


def chiou_young_14_calc_z1p0(vs30: float, region: str = None):
    """
    Calculates the z1p0 value for the Chiou and Youngs (2014) model

    Parameters
    ----------
    vs30 : float
        The Vs30 value, in meters per second
    region : str, optional
        The region to use, by default None which uses the global region
        other supported options are ["Japan"]

    Returns
    -------
    float
        The z1p0 value, in km
    """
    if region == "Japan":
        z1p0 = (
            -5.23 / 2 * np.log((vs30**2 + 412.39**2) / (1360**4 + 412.39**4))
        )  # In meters
    else:
        z1p0 = (
            -7.15 / 4 * np.log((vs30**4 + 570.94**4) / (1360**4 + 570.94**4))
        )  # In meters
    return np.exp(z1p0) / 1000  # In km


def mod_chiou_young_14_calc_z1p0(vs30: float, region: str = None):
    """
    Calculates the z1p0 value for the Chiou and Youngs (2014) model
    Modified for a different coefficient for the global model

    Parameters
    ----------
    vs30 : float
        The Vs30 value, in meters per second
    region : str, optional
        The region to use, by default None which uses the global region
        other supported options are ["Japan"]

    Returns
    -------
    float
        The z1p0 value, in km
    """
    if region == "Japan":
        z1p0 = (
            -5.23 / 2 * np.log((vs30**2 + 412.39**2) / (1360**4 + 412.39**4))
        )  # In meters
    else:
        z1p0 = (
            -7.15 / 4 * np.log((vs30**4 + 610**4) / (1360**4 + 610**4))
        )  # In meters
    return np.exp(z1p0) / 1000  # In km


def campbell_bozorgina_14_calc_z2p5(vs30: float, region: str = None):
    """
    Calculates the z2p5 value for the Campbell and Bozorgnia (2014) model

    Parameters
    ----------
    vs30 : float
        The Vs30 value, in meters per second
    region : str, optional
        The region to use, by default None which uses the global region
        other supported options are ["Japan"]

    Returns
    -------
    float
        The z2p5 value, in km
    """
    if region == "Japan":
        z2p5 = np.exp(7.089 - 1.144 * np.log(vs30))
    else:
        z2p5 = np.exp(5.359 - 1.102 * np.log(vs30))
    return z2p5  # In km


def chiou_young_08_calc_z1p0(vs30: float):
    """
    Calculates the z2p5 value for the Chiou and Youngs (2008) model

    Parameters
    ----------
    vs30 : float
        The Vs30 value, in meters per second

    Returns
    -------
    float
        The z1p0 value, in km
    """
    z1p0 = np.exp(28.5 - 3.82 / 8 * np.log(vs30**8 + 378.7**8))
    return z1p0  # In km


def abrahamson_gulerce_20_calc_z2p5(vs30: float, region: str):
    """
    Calculates the z2p5 value for the Abrahamson and Gulerce (2020) model

    Parameters
    ----------
    vs30 : float
        The Vs30 value, in meters per second
    region : str
        The region to use, can only be in ["Japan", "Cascadia"]

    Returns
    -------
    float
        The z2p5 value, in km
    """
    if region == "Cascadia":
        ln_zref = np.clip(8.52 - 0.88 * np.log(vs30 / 200.0), 7.6, 8.52)
    elif region == "Japan":
        ln_zref = np.clip(7.3 - 2.066 * np.log(vs30 / 170.0), 4.1, 7.3)
    else:
        raise ValueError("Does not support region %s" % region)
    return np.exp(ln_zref)  # In km


def parker_20_calc_z2p5(vs30: float, region: str):
    """
    Calculates the z2p5 value for the Parker et al. (2020) model

    Parameters
    ----------
    vs30 : float
        The Vs30 value, in meters per second
    region : str
        The region to use, can only be in ["Japan", "Cascadia"]
    """
    if region == "Japan":
        theta0, theta1, vmu, vsig = 3.05, -0.8, 500, 0.33
    elif region == "Cascadia":
        theta0, theta1, vmu, vsig = 3.94, -0.42, 200, 0.2
    else:
        raise ValueError("Does not support region %s" % region)
    z2pt5 = 10 ** (
        theta0
        + theta1 * (1 + erf((np.log10(vs30) - np.log10(vmu)) / (vsig * np.sqrt(2))))
    )
    return z2pt5  # In km


# Dictionary of models, mapping to their supported regions (None is global)
# This leads to each of the models functions and their return value being either ("zp1t0", "z2pt5")
Z_CALC_MODEL_REGION_MAPPING = {
    "S_22": {
        None: (chiou_young_14_calc_z1p0, "z1pt0"),
        "Japan": (chiou_young_14_calc_z1p0, "z1pt0"),
    },
    "CY_14": {
        None: (chiou_young_14_calc_z1p0, "z1pt0"),
        "Japan": (chiou_young_14_calc_z1p0, "z1pt0"),
    },
    "BSSA_14": {
        None: (chiou_young_14_calc_z1p0, "z1pt0"),
        "Japan": (chiou_young_14_calc_z1p0, "z1pt0"),
    },
    "ASK_14": {
        None: (mod_chiou_young_14_calc_z1p0, "z1pt0"),
        "Japan": (mod_chiou_young_14_calc_z1p0, "z1pt0"),
    },
    "CB_14": {
        None: (campbell_bozorgina_14_calc_z2p5, "z2pt5"),
        "Japan": (campbell_bozorgina_14_calc_z2p5, "z2pt5"),
    },
    "Br_13": {
        None: (chiou_young_08_calc_z1p0, "z1pt0"),
    },
    "K_20": {
        "Cascadia": (kuehn_20_calc_z, "z2pt5"),
        "Japan": (kuehn_20_calc_z, "z2pt5"),
        "NewZealand": (kuehn_20_calc_z, "z1pt0"),
        "Taiwan": (kuehn_20_calc_z, "z1pt0"),
    },
    "AG_20": {
        "Cascadia": (abrahamson_gulerce_20_calc_z2p5, "z2pt5"),
        "Japan": (abrahamson_gulerce_20_calc_z2p5, "z2pt5"),
    },
    "P_20": {
        "Cascadia": (parker_20_calc_z2p5, "z2pt5"),
        "Japan": (parker_20_calc_z2p5, "z2pt5"),
    },
}


def calc_z_for_model(
    model: GMM, vs30: Union[float, np.ndarray], region: Union[str, None]
):
    """
    Calculates the z value for a given model, region and Vs30 value / values

    Parameters
    ----------
    model : GMM
        The model to calculate the z value for
    vs30 : Union[float, np.ndarray]
        The Vs30 value or values, in meters per second
    region : Union[str, None]
        The region to use, use None to define a Global region.
        Use full region names, e.g. "NewZealand", "Cascadia", "Japan", "Taiwan"
        If the specific region is not supported for the given model it will attempt to use the Global model
        Otherwise a KeyError will be raised.

    Returns
    -------
    z_value: Union[float, np.ndarray]
        The z value or values, in km
    z_return: str
        The z value return type, either "z1pt0" or "z2pt5"
    """
    # Just incase global is defined as a string, set to None
    if region is not None:
        if region.lower() == "global":
            region = None
    # Find the mapping for Z value calculation
    if model.name in Z_CALC_MODEL_REGION_MAPPING:
        if region in Z_CALC_MODEL_REGION_MAPPING[model.name]:
            z_calc_function, z_return = Z_CALC_MODEL_REGION_MAPPING[model.name][region]
        # Extra check for a global region
        # since the region the user is wanting is not specifically available for the given model
        elif None in Z_CALC_MODEL_REGION_MAPPING[model.name]:
            z_calc_function, z_return = Z_CALC_MODEL_REGION_MAPPING[model.name][region]
            region = None
        else:
            raise KeyError(
                "Region %s not supported for model %s" % (region, model.name)
            )
    else:
        raise KeyError("Model %s not supported" % model.name)

    # Calculate the z value using the function
    if region is None:
        z_value = z_calc_function(vs30)
    else:
        z_value = z_calc_function(vs30, region)
    return z_value, z_return
