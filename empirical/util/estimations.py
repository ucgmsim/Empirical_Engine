import numpy as np
import pandas as pd

from source_modelling.sources import Plane


def estimate_width_ASK14(dip: pd.Series, mag: pd.Series):
    """Estimate a width for ASK_14 model
    The equation is from NGA-West 2 spreadsheet
    dip: pd.Series
    mag: pd.Series
    """
    return np.minimum(18 / np.sin(np.radians(dip)), 10 ** (-1.75 + 0.45 * mag))


def calculate_avg_strike_dip_rake(
    planes: list[Plane], plane_avg_rake: list[float], plane_total_slip: list[float]
):
    """
    Calculates the average strike, dip and rake of the fault planes
    based on the weighted average of the Total Slip on each plane.
    Useful when taking into account multiple fault planes and trying to calculate
    a single strike, dip and rake for the fault/scenario.

    Parameters
    ----------
    planes : list[Plane]
        List of Plane objects
    plane_avg_rake : list[float]
        List of the average rake of the fault planes
    plane_total_slip : list[float]
        List of the total slip on each fault plane

    Returns
    -------
    avg_strike : float
        Average strike of the fault planes
    avg_dip : float
        Average dip of the fault planes
    avg_rake : float
        Average rake of the fault planes
    """
    # Calculate the weights based on the total slip on each plane
    slip_weights = np.asarray(plane_total_slip) / sum(plane_total_slip)

    # Compute the weighted average of the strike, dip and rake
    avg_strike = np.average([plane.strike for plane in planes], weights=slip_weights)
    avg_dip = np.average([plane.dip for plane in planes], weights=slip_weights)
    avg_rake = np.average(plane_avg_rake, weights=slip_weights)

    return avg_strike, avg_dip, avg_rake
