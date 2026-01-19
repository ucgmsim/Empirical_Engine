import numpy as np
import pandas as pd

from qcore.constants import EXT_PERIOD

from ... import constants, oq_wrapper

# GA_11 Specific coefficients for V/H Ratio
BETWEEN_EVENT_COEFFICIENTS = [
    -0.193,
    -0.068,
    -0.186,
    -0.314,
    -0.413,
    -0.418,
    -0.416,
    -0.351,
    -0.370,
    -0.245,
    -0.361,
    -0.358,
    -0.181,
    -0.2,
    -0.171,
]
WITHIN_EVENT_COEFFICIENTS = [
    -0.389,
    -0.273,
    -0.358,
    -0.417,
    -0.439,
    -0.45,
    -0.466,
    -0.493,
    -0.483,
    -0.443,
    -0.318,
    -0.254,
    -0.295,
    -0.396,
    -0.577,
]
COEFFICIENT_PERIODS = [
    0.01,
    0.05,
    0.1,
    0.15,
    0.2,
    0.3,
    0.4,
    0.5,
    0.75,
    1,
    2,
    3,
    4,
    5,
    10,
]


def get_model_results(
    model: constants.GMM,
    tect_type: constants.TectType,
    rupture_df: pd.DataFrame,
    im: str,
    periods: np.ndarray,
    kwargs: dict = None,
):
    """
    Get the model results through openquake using the vectorized wrapper

    Parameters
    ----------
    model: GMM
        The model to use
    tect_type: TectType
        The tectonic type to use
    rupture_df: pd.DataFrame
        The rupture dataframe to use
    im: str
        The IM to use
    periods: np.ndarray
        The periods to use
    kwargs: dict (optional) (default=None)
        Any additional arguments to pass to the openquake wrapper

    Returns
    -------
    mu_values: pd.DataFrame
        The mean values for each period for each entry in the rupture dataframe
    sigma_values: pd.DataFrame
        The standard deviation values for each period for each entry in the rupture dataframe
    between_values: pd.DataFrame
        The between event standard deviation values for each period for each entry in the rupture dataframe
    within_values: pd.DataFrame
        The within event standard deviation values for each period for each entry in the rupture dataframe
    """
    if kwargs is None:
        kwargs = {}

    result = oq_wrapper.run_gmm(model, tect_type, rupture_df, im, periods, **kwargs)

    # Sort the output by getting the columns with the mean, std_Total, std_Inter, std_Intra
    mu_values = [x for x in result if "mean" in x]
    sigma_values = [x for x in result if "std_Total" in x]
    between_values = [x for x in result if "std_Inter" in x]
    within_values = [x for x in result if "std_Intra" in x]

    # Get the values for each of the columns
    mu_values = result[mu_values]
    sigma_values = result[sigma_values]
    between_values = result[between_values]
    within_values = result[within_values]

    # Rename columns to just IM period
    mu_values.columns = [x.split("_")[1] for x in mu_values.columns]
    sigma_values.columns = [x.split("_")[1] for x in sigma_values.columns]
    between_values.columns = [x.split("_")[1] for x in between_values.columns]
    within_values.columns = [x.split("_")[1] for x in within_values.columns]

    return mu_values, sigma_values, between_values, within_values


def get_period_correlations(periods: np.ndarray):
    """
    Get the period correlations for the vertical spectra by interpolating the
    coefficients from the GA_11 model for the periods of interest

    Parameters
    ----------
    periods: np.array
        Periods of interest

    Returns
    -------
    between_interpolated: np.ndarray
        Between event period correlations
    within_interpolated: np.ndarray
        Within event period correlations
    """
    between_interpolated = np.interp(
        np.log(periods), np.log(COEFFICIENT_PERIODS), BETWEEN_EVENT_COEFFICIENTS
    )
    within_interpolated = np.interp(
        np.log(periods), np.log(COEFFICIENT_PERIODS), WITHIN_EVENT_COEFFICIENTS
    )

    return between_interpolated, within_interpolated


def calculate_vertical_spectra(
    rupture_df: pd.DataFrame,
    model: constants.GMM = constants.GMM.ASK_14,
    periods: np.ndarray = EXT_PERIOD,
):
    """
    Calculate the vertical spectra for a given rupture_df and model
    Note: Can only be used for Active Shallow Tectionic Types due to the
    vertical model GA_11 requirements

    Parameters
    ----------
    rupture_df: pd.DataFrame
        Rupture dataframe
    model: GMM (optional) (default: GMM.ASK_14)
        Model to use for the Horizontal pSA calculation
    periods: np.ndarray (optional) (default: EXT_PERIOD)
        Periods to get for the Vertical pSA

    Returns
    -------
    mu_ln_v_values: pd.DataFrame
        Vertical mean values, same length as rupture_df
    sigma_ln_v_values: pd.DataFrame
        Vertical sigma values, same length as rupture_df
    """
    # Function constant variables
    ratio_model = constants.GMM.GA_11
    tect_type = constants.TectType.ACTIVE_SHALLOW
    im = "pSA"

    # Get the horizontal and vertical ratio
    (
        v_h_mu_values,
        v_h_sigma_values,
        v_h_between_values,
        v_h_within_values,
    ) = get_model_results(
        ratio_model,
        tect_type,
        rupture_df,
        im,
        periods,
        kwargs={"gmpe_name": "AbrahamsonSilva2008"},
    )

    # Get the Horizontal PSA
    h_mu_values, h_sigma_values, h_between_values, h_within_values = get_model_results(
        model, tect_type, rupture_df, im, periods
    )

    # Get the period correlations
    between_correlation, within_correlation = get_period_correlations(periods)

    # Calculate the Vertical PSA
    mu_ln_v_values = v_h_mu_values + h_mu_values
    variance_h = h_sigma_values**2
    variance_v_h = v_h_sigma_values**2
    p = (
        h_within_values * v_h_within_values * within_correlation
        + h_between_values * v_h_between_values * between_correlation
    ) / (variance_h * variance_v_h)
    covariance_h_v = p * h_sigma_values * v_h_sigma_values
    variance_v = variance_h + variance_v_h + covariance_h_v
    sigma_ln_v_values = np.sqrt(variance_v)

    return mu_ln_v_values, sigma_ln_v_values
