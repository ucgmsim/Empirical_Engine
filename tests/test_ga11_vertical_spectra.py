"""Tests for the ga11_vertical_spectra module."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from empirical.util.classdef import GMM, TectType
from empirical.util import ga11_vertical_spectra as ga11


@pytest.fixture
def mock_rupture_df():
    """Create a mock rupture dataframe for testing."""
    return pd.DataFrame({
        "vs30": [760.0],
        "rrup": [10.0],
        "mag": [6.0],
        "dip": [90.0],
        "rake": [0.0],
        "ztor": [0.0],
        "width": [10.0],
        "rx": [5.0],
        "rjb": [8.0]
    })


def test_get_period_correlations():
    """Test that period correlations are correctly interpolated."""
    # Test with periods matching the coefficient periods
    periods = np.array(ga11.COEFFICIENT_PERIODS)
    between, within = ga11.get_period_correlations(periods)
    
    # Check that the correlations match the coefficients when using exact periods
    assert np.allclose(between, ga11.BETWEEN_EVENT_COEFFICIENTS)
    assert np.allclose(within, ga11.WITHIN_EVENT_COEFFICIENTS)
    
    # Test interpolation with a period value between coefficient periods
    periods = np.array([0.25])  # Between 0.2 and 0.3
    between, within = ga11.get_period_correlations(periods)
    
    # Expected values are interpolated values
    expected_between = np.interp(
        np.log(periods), np.log(ga11.COEFFICIENT_PERIODS), ga11.BETWEEN_EVENT_COEFFICIENTS
    )
    expected_within = np.interp(
        np.log(periods), np.log(ga11.COEFFICIENT_PERIODS), ga11.WITHIN_EVENT_COEFFICIENTS
    )
    
    assert np.allclose(between, expected_between)
    assert np.allclose(within, expected_within)


@patch("empirical.util.ga11_vertical_spectra.oq_run")
def test_get_model_results(mock_oq_run, mock_rupture_df):
    """Test the get_model_results function."""
    # Create mock result from oq_run
    mock_result = pd.DataFrame({
        "pSA_0.1_mean": [1.0],
        "pSA_0.2_mean": [0.9],
        "pSA_0.1_std_Total": [0.5],
        "pSA_0.2_std_Total": [0.6],
        "pSA_0.1_std_Inter": [0.3],
        "pSA_0.2_std_Inter": [0.4],
        "pSA_0.1_std_Intra": [0.4],
        "pSA_0.2_std_Intra": [0.5]
    })
    mock_oq_run.return_value = mock_result
    
    # Test parameters
    model = GMM.ASK_14
    tect_type = TectType.ACTIVE_SHALLOW
    periods = np.array([0.1, 0.2])
    
    # Call function
    mu, sigma, between, within = ga11.get_model_results(model, tect_type, mock_rupture_df, "pSA", periods)
    
    # Verify oq_run was called correctly
    mock_oq_run.assert_called_once_with(model, tect_type, mock_rupture_df, "pSA", periods)
    
    # Check results using numpy allclose instead of direct DataFrame comparison
    assert np.allclose(mu.values, [[1.0, 0.9]])
    assert np.allclose(sigma.values, [[0.5, 0.6]])
    assert np.allclose(between.values, [[0.3, 0.4]])
    assert np.allclose(within.values, [[0.4, 0.5]])
    
    # Check column names - convert column names to float for comparison
    # since columns may be stored as strings in the DataFrame
    mu_cols = [float(col) for col in mu.columns]
    sigma_cols = [float(col) for col in sigma.columns]
    assert mu_cols == [0.1, 0.2]
    assert sigma_cols == [0.1, 0.2]


@patch("empirical.util.ga11_vertical_spectra.get_model_results")
@patch("empirical.util.ga11_vertical_spectra.get_period_correlations")
def test_calculate_vertical_spectra(mock_get_period_correlations, mock_get_model_results, mock_rupture_df):
    """Test the calculate_vertical_spectra function."""
    # Setup mock returns
    periods = np.array([0.1, 0.2])
    
    # Mock horizontal model results
    h_mu = pd.DataFrame({0.1: [0.5], 0.2: [0.6]})
    h_sigma = pd.DataFrame({0.1: [0.4], 0.2: [0.5]})
    h_between = pd.DataFrame({0.1: [0.2], 0.2: [0.3]})
    h_within = pd.DataFrame({0.1: [0.3], 0.2: [0.4]})
    
    # Mock V/H ratio results
    vh_mu = pd.DataFrame({0.1: [-0.2], 0.2: [-0.3]})
    vh_sigma = pd.DataFrame({0.1: [0.3], 0.2: [0.4]})
    vh_between = pd.DataFrame({0.1: [0.1], 0.2: [0.2]})
    vh_within = pd.DataFrame({0.1: [0.2], 0.2: [0.3]})
    
    # Mock period correlations
    between_corr = np.array([-0.186, -0.413])  # Values for 0.1s and 0.2s
    within_corr = np.array([-0.358, -0.439])   # Values for 0.1s and 0.2s
    
    # Setup mock return values
    mock_get_model_results.side_effect = [
        (vh_mu, vh_sigma, vh_between, vh_within),  # For V/H ratio
        (h_mu, h_sigma, h_between, h_within)       # For horizontal model
    ]
    mock_get_period_correlations.return_value = (between_corr, within_corr)
    
    # Call function
    mu_v, sigma_v = ga11.calculate_vertical_spectra(mock_rupture_df, model=GMM.ASK_14, periods=periods)
    
    # Check calls
    mock_get_model_results.assert_any_call(GMM.GA_11, TectType.ACTIVE_SHALLOW, mock_rupture_df, 
                                           "pSA", periods, kwargs={"gmpe_name": "AbrahamsonSilva2008"})
    mock_get_model_results.assert_any_call(GMM.ASK_14, TectType.ACTIVE_SHALLOW, mock_rupture_df, 
                                           "pSA", periods)
    mock_get_period_correlations.assert_called_once_with(periods)
    
    # Calculate expected results
    expected_mu_v = vh_mu + h_mu
    
    variance_h = h_sigma**2
    variance_vh = vh_sigma**2
    
    # Calculate p for each period
    p_01 = (h_within.iloc[0, 0] * vh_within.iloc[0, 0] * within_corr[0] + 
            h_between.iloc[0, 0] * vh_between.iloc[0, 0] * between_corr[0]) / (
            variance_h.iloc[0, 0] * variance_vh.iloc[0, 0])
    
    p_02 = (h_within.iloc[0, 1] * vh_within.iloc[0, 1] * within_corr[1] + 
            h_between.iloc[0, 1] * vh_between.iloc[0, 1] * between_corr[1]) / (
            variance_h.iloc[0, 1] * variance_vh.iloc[0, 1])
    
    # Calculate covariance for each period
    cov_01 = p_01 * h_sigma.iloc[0, 0] * vh_sigma.iloc[0, 0]
    cov_02 = p_02 * h_sigma.iloc[0, 1] * vh_sigma.iloc[0, 1]
    
    # Calculate variance for each period
    var_v_01 = variance_h.iloc[0, 0] + variance_vh.iloc[0, 0] + cov_01
    var_v_02 = variance_h.iloc[0, 1] + variance_vh.iloc[0, 1] + cov_02
    
    # Calculate expected sigma
    expected_sigma_v = pd.DataFrame({
        0.1: [np.sqrt(var_v_01)],
        0.2: [np.sqrt(var_v_02)]
    })
    
    # Check results using numpy allclose
    assert np.allclose(mu_v.values, expected_mu_v.values)
    assert np.allclose(sigma_v.values, expected_sigma_v.values, rtol=1e-5)