"""Shared fixtures and utilities for pytest."""

from typing import Dict, Any, Optional, List, Union, Callable, Tuple
import inspect
import pytest
import numpy as np
import pandas as pd
from functools import wraps

from openquake.hazardlib import const, contexts, gsim, imt

# Import these here to make them available to test functions
from empirical.util.classdef import GMM, TectType


@pytest.fixture()  # Using default function scope to match monkeypatch
def mock_openquake_api(monkeypatch, request) -> Dict[str, Any]:
    """Fixture to mock OpenQuake's API functions.
    
    This fixture mocks the key OpenQuake functions used by our wrapper code.
    By default, it returns simple arrays with predictable values for testing,
    but can be customized by adding marker parameters.
    
    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest's monkeypatch fixture for modifying objects
    request : pytest.FixtureRequest
        Pytest's request object for test introspection
        
    Returns
    -------
    Dict[str, Any]
        A dictionary with mocked functions and their return values
        
    Examples
    --------
    Basic usage:
        def test_with_default_mock(mock_openquake_api):
            # The mock is already applied with default values
            assert len(mock_openquake_api['get_mean_stds']) == 4
            
    Customizing return values using marker:
        @pytest.mark.openquake_returns(mean=0.5, total=0.3)
        def test_with_custom_values(mock_openquake_api):
            # Access custom values
            assert mock_openquake_api['get_mean_stds'][0][0][0] == 0.5
    """
    # Get custom return values from marker if present
    marker = request.node.get_closest_marker("openquake_returns")
    
    mean_val = 0.1
    total_val = 0.2
    inter_val = 0.3
    intra_val = 0.4
    
    if marker:
        # Extract custom values from marker kwargs
        marker_kwargs = marker.kwargs
        mean_val = marker_kwargs.get("mean", mean_val)
        total_val = marker_kwargs.get("total", total_val)
        inter_val = marker_kwargs.get("inter", inter_val)
        intra_val = marker_kwargs.get("intra", intra_val)
    
    # Define mock function outputs
    mock_returns = {
        'get_mean_stds': [
            np.array([[mean_val]]),   # mean
            np.array([[total_val]]),  # std_total
            np.array([[inter_val]]),  # std_inter
            np.array([[intra_val]])   # std_intra
        ]
    }
    
    # Define a mock function to replace get_mean_stds
    def mock_get_mean_stds(*args, **kwargs) -> List[np.ndarray]:
        return mock_returns['get_mean_stds']
    
    # Apply the mock
    monkeypatch.setattr(contexts, 'get_mean_stds', mock_get_mean_stds)
    
    return mock_returns


@pytest.fixture(scope="module")
def rupture_data_factory() -> Callable[..., pd.DataFrame]:
    """Factory fixture to create configurable rupture data.
    
    Returns a function that generates rupture DataFrames with custom parameters.
    
    Example:
        def test_with_custom_data(rupture_data_factory):
            # Create data with 3 sites and custom magnitude
            data = rupture_data_factory(num_sites=3, custom_values={'mag': 7.5})
            assert len(data) == 3
            assert data['mag'].iloc[0] == 7.5
    """
    def _create_rupture_data(num_sites: int = 1, 
                             custom_values: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
        """Create rupture data for GMM testing.
        
        Parameters
        ----------
        num_sites : int, optional
            Number of sites to include, by default 1
        custom_values : dict, optional
            Custom values to override defaults, by default None
        
        Returns
        -------
        pandas.DataFrame
            DataFrame with rupture data
        """
        # Default values
        defaults = {
            'vs30': 760.0,
            'vs30measured': False,
            'z1pt0': 0.001,  # km, will be converted to m
            'rrup': 10.0,
            'rjb': 10.0,
            'rx': 5.0,
            'ry0': 5.0,      # Required by ASK_14
            'mag': 6.0,
            'rake': 0.0,
            'dip': 90.0,
            'ztor': 0.0
        }
        
        # Override defaults with custom values if provided
        if custom_values:
            for key, value in custom_values.items():
                defaults[key] = value
        
        # Create a DataFrame with the specified number of sites
        data = {key: [value] * num_sites for key, value in defaults.items()}
        return pd.DataFrame(data)
    
    return _create_rupture_data


@pytest.fixture(scope="module")
def test_rupture_data(rupture_data_factory) -> pd.DataFrame:
    """Fixture to provide basic test rupture data for GMM testing.
    
    Returns a DataFrame with a single site using default parameters.
    """
    return rupture_data_factory()


# Utility functions for common test operations
def assert_result_columns(
    result: pd.DataFrame, 
    im_prefix: str, 
    column_types: Optional[List[str]] = None,
    strict: bool = False
) -> None:
    """Assert that result DataFrame contains expected columns.
    
    Parameters
    ----------
    result : pd.DataFrame
        Result DataFrame to check
    im_prefix : str
        Intensity measure prefix (e.g., "PGA", "pSA_0.5")
    column_types : Optional[List[str]], optional
        Types of columns to check for, defaults to ["mean", "std_Total", "std_Inter", "std_Intra"]
    strict : bool, optional
        If True, checks that only the expected columns exist and nothing else, by default False
    """
    column_types = column_types or ["mean", "std_Total", "std_Inter", "std_Intra"]
    expected_columns = [f"{im_prefix}_{col_type}" for col_type in column_types]
    
    # Check that all expected columns exist
    for expected_column in expected_columns:
        assert expected_column in result.columns, f"Result should contain {expected_column} column"
    
    # If strict mode, also check that no unexpected columns exist
    if strict:
        for col in result.columns:
            assert any(col == exp_col for exp_col in expected_columns), f"Unexpected column {col} in result"
            
    # Check that there are no NaN values in the result
    assert not result[expected_columns].isna().any().any(), "Result contains NaN values"


def check_model_attributes(model_class: Any, model_name: str) -> Tuple[bool, str]:
    """Check that a model class has the expected attributes for a ground motion model.
    
    Parameters
    ----------
    model_class : Any
        The model class to check
    model_name : str
        Name to use in error messages
        
    Returns
    -------
    Tuple[bool, str]
        Success status and error message if any
    """
    try:
        # Check basic type requirements
        if not inspect.isclass(model_class):
            return False, f"Model {model_name} should be a class"
        
        if not issubclass(model_class, gsim.base.GMPE):
            return False, f"Model {model_name} should inherit from GMPE"
        
        # Check for required attributes for a valid Ground Motion Model
        required_attributes = [
            "DEFINED_FOR_TECTONIC_REGION_TYPE",
            "DEFINED_FOR_INTENSITY_MEASURE_TYPES",
            "DEFINED_FOR_STANDARD_DEVIATION_TYPES"
        ]
        
        for attr in required_attributes:
            if not hasattr(model_class, attr):
                return False, f"Model {model_name} is missing required attribute: {attr}"
        
        # Check for at least one supported intensity measure type
        if not model_class.DEFINED_FOR_INTENSITY_MEASURE_TYPES:
            return False, f"Model {model_name} doesn't define any intensity measure types"
            
        return True, ""
    except Exception as e:
        return False, f"Error checking model {model_name}: {str(e)}"


def create_test_im_config() -> List[Dict[str, Any]]:
    """Create a configuration for testing different intensity measures.
    
    This utility function returns a list of test configurations for different
    intensity measure types, which can be used with pytest's parametrize.
    
    Returns
    -------
    List[Dict[str, Any]]
        List of test configurations for different IMs
    
    Example
    -------
    @pytest.mark.parametrize('im_config', create_test_im_config())
    def test_multiple_ims(mock_openquake_api, test_rupture_data, im_config):
        result = oq_run(
            model_type=im_config['model'],
            tect_type=im_config['tect_type'],
            rupture_df=test_rupture_data,
            im=im_config['im'],
            **im_config.get('kwargs', {})
        )
        assert_result_columns(result, im_config['im_check'])
    """
    return [
        # PGA tests
        {
            'id': 'PGA-active-shallow',
            'model': GMM.Br_13,
            'tect_type': TectType.ACTIVE_SHALLOW,
            'im': 'PGA',
            'im_check': 'PGA'
        },
        # SA tests
        {
            'id': 'SA(0.5)-active-shallow',
            'model': GMM.Br_13,
            'tect_type': TectType.ACTIVE_SHALLOW,
            'im': 'SA',
            'kwargs': {'periods': [0.5]},
            'im_check': 'pSA_0.5'
        },
        # Add other common IMs as needed
    ]


def openquake_test_wrapper(func):
    """Decorator for OpenQuake test functions that validates inputs and handles common errors.
    
    This decorator adds consistent error handling and validation to OpenQuake tests.
    It automatically checks for common issues in test results and provides better
    error messages.
    
    Parameters
    ----------
    func : Callable
        Test function to wrap
        
    Returns
    -------
    Callable
        Wrapped test function
        
    Example
    -------
    @openquake_test_wrapper
    def test_my_gmm(mock_openquake_api, test_rupture_data):
        # The wrapper will automatically validate the result
        return oq_wrapper.oq_run(...)
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            # Call the wrapped function
            result = func(*args, **kwargs)
            
            # Basic validation of the result if it's a DataFrame
            if isinstance(result, pd.DataFrame):
                # Check for any infinite values
                assert not np.isinf(result.values).any(), "Result contains infinite values"
                
                # Basic check that we have at least some results
                assert not result.empty, "Result DataFrame is empty"
                
            return result
        except Exception as e:
            print(f"\nError in {func.__name__}: {str(e)}")
            raise
            
    return wrapper