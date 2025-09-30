"""
Module that contains functions used to estimate input parameters for
empirical GMMs, such as fault width and Z-values.
"""
import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate
from scipy.special import erf

from source_modelling.sources import Plane

from . import constants


def estimate_width_ASK14(   # noqa: N802
    dip: npt.ArrayLike, mag: npt.ArrayLike
) -> np.ndarray:  
    """
    Estimate the fault rupture width using the ASK14 model.
    This is based on the NGA-West 2 GMM implementation.

    Parameters
    ----------
    dip : array-like
        Dip angle of the fault in degrees. Should be a 1D array or Series of floats.
    mag : array-like
        Magnitude of the earthquake. Should be a 1D array or Series of floats.

    Returns
    -------
    np.ndarray
        Estimated rupture width for each (dip, mag) pair.
    """
    return np.minimum(18 / np.sin(np.radians(dip)), 10 ** (-1.75 + 0.45 * mag))


def circ_mean(
    samples: npt.NDArray[np.floating], weights: npt.NDArray[np.floating]
) -> float:
    """
    Calculate the circular mean of a set of angles in radians, taking into account
    their weights. This is useful for averaging angles like strike and rake.

    Parameters
    ----------
    samples : array-like
        Array of angles in radians.
    weights : array-like
        Array of weights corresponding to each angle.

    Returns
    -------
    float
        The circular mean angle in radians.
    """
    weighted_sines = np.sum(np.sin(samples) * weights)
    weighted_cosines = np.sum(np.cos(samples) * weights)
    argument = np.arctan2(weighted_sines, weighted_cosines)
    return float(argument)


def calculate_avg_multi_plane_properties(
    planes: list[Plane], plane_avg_rake: list[float], plane_areas: list[float]
) -> tuple[float, float, float, float]:
    """
    Calculates the average strike, dip, rake and width of the fault planes
    based on the weighted average of the Area of each plane.
    Useful when taking into account multiple fault planes and trying to calculate
    a single strike, dip, rake and width for the fault/scenario.
    For strike and rake the circular nature is taken into account by using
    the circ_mean function.

    Parameters
    ----------
    planes : list[Plane]
        List of Plane objects
    plane_avg_rake : list[float]
        List of the average rake of the fault planes
    plane_areas : list[float]
        List of the areas of each fault plane

    Returns
    -------
    avg_strike : float
        Average strike of the fault planes
    avg_dip : float
        Average dip of the fault planes
    avg_rake : float
        Average rake of the fault planes
    avg_width : float
        Average width of the fault planes
    """
    try:
        # Calculate the weights based on the area of each plane
        area_weights = np.asarray(plane_areas) / sum(plane_areas)
    except ZeroDivisionError:
        raise ValueError("Sum of plane areas cannot be zero, check input data.")

    # Compute the weighted average of the strike, dip and rake
    avg_strike = circ_mean(
        np.radians([plane.strike for plane in planes]), weights=area_weights
    )
    avg_strike = np.degrees(avg_strike) % 360  # Convert back to degrees
    avg_rake = circ_mean(
        np.radians(plane_avg_rake), weights=area_weights
    )
    avg_rake = (np.degrees(avg_rake) + 180) % 360 - 180  # Convert back to degrees in [-180, 180] range
    avg_dip = np.average([plane.dip for plane in planes], weights=area_weights)
    avg_width = np.average([plane.width for plane in planes], weights=area_weights)

    return avg_strike, avg_dip, avg_rake, avg_width


def kuehn_20_calc_z(
    vs30: npt.ArrayLike, region: str
) -> float | np.ndarray:
    """
    Calculates the z1p0 or z2p5 value for the Kuehn et al. (2020) model
    Depends on the region for z1p0 or z2p5

    Parameters
    ----------
    vs30 : array-like
        The Vs30 value or values, in meters per second
    region : str
        The region to use, must be one of ["Cascadia", "Japan", "NewZealand", "Taiwan"]

    Returns
    -------
    float | np.ndarray
        Z1.0, in km if region is ["NewZealand", "Taiwan"]
        Z2.5, in km if region is ["Cascadia", "Japan"]
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


def chiou_young_14_calc_z1p0(
    vs30: npt.ArrayLike, region: str | None = None
) -> float | np.ndarray:
    """
    Calculates the z1p0 value for the Chiou and Youngs (2014) model

    Parameters
    ----------
    vs30 : float | np.ndarray
        The Vs30 value or values, in meters per second
    region : str, optional
        The region to use, by default None which uses the global region
        other supported options are ["Japan"]

    Returns
    -------
    float | np.ndarray
        The z1p0 value or values, in km
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


def mod_chiou_young_14_calc_z1p0(
    vs30: npt.ArrayLike, region: str | None = None
) -> float | np.ndarray:
    """
    Calculates the z1p0 value for the Chiou and Youngs (2014) model
    Modified for a different coefficient for the global model

    Parameters
    ----------
    vs30 : array-like
        The Vs30 value or values, in meters per second
    region : str, optional
        The region to use, by default None which uses the global region
        other supported options are ["Japan"]

    Returns
    -------
    float | np.ndarray
        The z1p0 value or values, in km
    """
    if region == "Japan":
        z1p0 = (
            -5.23 / 2 * np.log((vs30**2 + 412.39**2) / (1360**4 + 412.39**4))
        )  # In meters
    else:
        z1p0 = -7.15 / 4 * np.log((vs30**4 + 610**4) / (1360**4 + 610**4))  # In meters
    return np.exp(z1p0) / 1000  # In km


def campbell_bozorgina_14_calc_z2p5(
    vs30: npt.ArrayLike, region: str | None = None
) -> float | np.ndarray:
    """
    Calculates the z2p5 value for the Campbell and Bozorgnia (2014) model

    Parameters
    ----------
    vs30 : array-like
        The Vs30 value or values, in meters per second
    region : str, optional
        The region to use, by default None which uses the global region
        other supported options are ["Japan"]

    Returns
    -------
    float | np.ndarray
        The z2p5 value or values, in km
    """
    if region == "Japan":
        z2p5 = np.exp(7.089 - 1.144 * np.log(vs30))
    else:
        z2p5 = np.exp(5.359 - 1.102 * np.log(vs30))
    return z2p5  # In km


def chiou_young_08_calc_z1p0(
    vs30: npt.ArrayLike,
) -> float | np.ndarray:
    """
    Calculates the z2p5 value for the Chiou and Youngs (2008) model

    Parameters
    ----------
    vs30 : array-like
        The Vs30 value or values, in meters per second

    Returns
    -------
    float | np.ndarray
        The z1p0 value or values, in km
    """
    z1p0 = np.exp(28.5 - 3.82 / 8 * np.log(vs30**8 + 378.7**8)) / 1000  # In km
    return z1p0


def chiou_young_08_calc_z2p5(
    z1p0: npt.ArrayLike | None = None,
    z1p5: float | np.ndarray | pd.DataFrame = None,
) -> float | np.ndarray:
    """
    Calculates the z2p5 value using z1p0 or z1p5 for the Chiou and Youngs (2008) model

    Parameters
    ----------
    z1p0 : array-like, optional
        Z1.0 values in km, by default None
    z1p5 : array-like, optional
        Z1.5 values in km, by default None

    Returns
    -------
    float | np.ndarray
        Z2.5 values in the same format as z1p0 or z1p5 in km
    """
    if z1p5 is not None:
        return 0.636 + 1.549 * z1p5
    elif z1p0 is not None:
        return 0.519 + 3.595 * z1p0
    else:
        raise ValueError("no z2p5 able to be estimated")


def abrahamson_gulerce_20_calc_z2p5(
    vs30: npt.ArrayLike, region: str
) -> float | np.ndarray:
    """
    Calculates the z2p5 value for the Abrahamson and Gulerce (2020) model

    Parameters
    ----------
    vs30 : array-like
        The Vs30 value or values, in meters per second
    region : str
        The region to use, can only be in ["Japan", "Cascadia"]

    Returns
    -------
    float | np.ndarray
        The z2p5 value or values, in km
    """
    if region == "Cascadia":
        ln_zref = np.clip(8.52 - 0.88 * np.log(vs30 / 200.0), 7.6, 8.52)
    elif region == "Japan":
        ln_zref = np.clip(7.3 - 2.066 * np.log(vs30 / 170.0), 4.1, 7.3)
    else:
        raise ValueError("Does not support region %s" % region)
    return np.exp(ln_zref)  # In km


def parker_20_calc_z2p5(
    vs30: npt.ArrayLike, region: str
) -> float | np.ndarray:
    """
    Calculates the z2p5 value for the Parker et al. (2020) model

    Parameters
    ----------
    vs30 : array-like
        The Vs30 value or values, in meters per second
    region : str
        The region to use, can only be in ["Japan", "Cascadia"]

    Returns
    -------
    float | np.ndarray
        The z2p5 value or values, in km
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
    model: constants.GMM, vs30: npt.ArrayLike, region: str | None = None
) -> tuple[np.ndarray, str]:
    """
    Calculates the z value for a given model, region and Vs30 value / values

    Parameters
    ----------
    model : constants.GMM
        The model to calculate the z value for
    vs30 : array-like
        The Vs30 value or values, in meters per second
    region : Union[str, None]
        The region to use, use None to define a Global region, default is None.
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


def interpolate_with_pga(
    period: float | int,
    model_min_period: float,
    pga_y: pd.DataFrame,
    min_period_y: pd.DataFrame,
) -> pd.DataFrame:
    """
    Use interpolation to find the pSA value at the given period,
    which is between PGA and the model's minimum period.

    Parameters
    ----------
    period : float or int
        Target period for interpolation
    model_min_period : float
        Minimum supported pSA period for GMM of interest
    pga_y : pd.DataFrame
        DataFrame that contains GMM results for PGA
    min_period_y : pd.DataFrame
        DataFrame that contains GMM results
        at the model's minimum period

    Returns
    -------
    pd.DataFrame
        DataFrame containing the interpolated ground motion metrics (mean,
        total standard deviation, inter-event standard deviation,
        intra-event standard deviation) at the target `period`. The columns
        are named using the pattern ``pSA_{period}_{metric_name}``.
        The mean values are returned in log space.
    """
    x = [0.0, model_min_period]
    # each subarray represents values at period=0.0 and period=high_period
    # E.g., two site/rupture data would look something like
    # mean_y = np.array([[a,b], [c,d]])
    # where a,c are at period=0.0=PGA and b,d are at period=model_min_period
    mean_y = np.concatenate(
        (
            np.exp(pga_y.loc[:, pga_y.columns.str.endswith("mean")].to_numpy()),
            np.exp(min_period_y.loc[:, min_period_y.columns.str.endswith("mean")].to_numpy()),
        ),
        axis=1,
    )
    sigma_total_y = np.concatenate(
        (
            pga_y.loc[:, pga_y.columns.str.endswith("std_Total")].to_numpy(),
            min_period_y.loc[:, min_period_y.columns.str.endswith("std_Total")].to_numpy(),
        ),
        axis=1,
    )
    sigma_inter_y = np.concatenate(
        (
            pga_y.loc[:, pga_y.columns.str.endswith("std_Inter")].to_numpy(),
            min_period_y.loc[:, min_period_y.columns.str.endswith("std_Inter")].to_numpy(),
        ),
        axis=1,
    )
    sigma_intra_y = np.concatenate(
        (
            pga_y.loc[:, pga_y.columns.str.endswith("std_Intra")].to_numpy(),
            min_period_y.loc[:, min_period_y.columns.str.endswith("std_Intra")].to_numpy(),
        ),
        axis=1,
    )

    return pd.DataFrame(
        {
            f"pSA_{period}_mean": np.log(interpolate.interp1d(x, mean_y)(period)),
            f"pSA_{period}_std_Total": interpolate.interp1d(x, sigma_total_y)(period),
            f"pSA_{period}_std_Inter": interpolate.interp1d(x, sigma_inter_y)(period),
            f"pSA_{period}_std_Intra": interpolate.interp1d(x, sigma_intra_y)(period),
        }
    )
