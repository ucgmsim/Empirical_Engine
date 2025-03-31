#!/usr/bin/env python3
"""Tests to verify OpenQuake API integration.

These tests check that OpenQuake's API hasn't changed in ways that would break our wrapper.
If these tests fail, it means that the OpenQuake library has changed its API and
our wrapper code needs to be updated accordingly.
"""

from typing import List, Dict, Any, Type, cast
import inspect
import pytest
import numpy as np
import pandas as pd
from importlib.metadata import version, PackageNotFoundError

from openquake.hazardlib import const, contexts, gsim, imt

from empirical.util.classdef import GMM, TectType
from empirical.util import openquake_wrapper_vectorized as oq_wrapper
from conftest import assert_result_columns, check_model_attributes, create_test_im_config


def test_openquake_version() -> None:
    """Test that we're using the expected OpenQuake version.
    
    This test retrieves the installed OpenQuake version and prints it.
    If the version can't be determined, the test is skipped.
    """
    # Use a more robust approach to get the version
    packages_to_check = ["openquake.engine", "openquake.hazardlib"]
    
    for package in packages_to_check:
        try:
            oq_version = version(package)
            print(f"Current OpenQuake version ({package}): {oq_version}")
            # Test passes if we got a version
            assert oq_version, f"OpenQuake {package} version should be available"
            return
        except PackageNotFoundError:
            continue
    
    # If we get here, none of the packages were found
    pytest.skip("Could not determine OpenQuake version - no packages found")


def test_openquake_context_structure() -> None:
    """Test that OpenQuake's RuptureContext has the expected structure."""
    # Verify RuptureContext exists
    assert hasattr(contexts, 'RuptureContext'), "OpenQuake's contexts module should have RuptureContext"
    
    # Check if we can instantiate RuptureContext as expected
    try:
        ctx = contexts.RuptureContext()
        ctx.occurrence_rate = None
        ctx.sids = [1]
        ctx.vs30 = np.array([760.0])
        ctx.mag = np.array([6.0])
        ctx.dip = np.array([90.0])
    except Exception as e:
        pytest.fail(f"Failed to instantiate RuptureContext as expected: {e}")


def test_get_mean_stds_signature() -> None:
    """Test that OpenQuake's get_mean_stds function has the expected signature."""
    assert hasattr(contexts, 'get_mean_stds'), "OpenQuake's contexts module should have get_mean_stds"
    
    # Check function signature
    sig = inspect.signature(contexts.get_mean_stds)
    # The function currently takes four parameters in the latest version
    # This could be due to API changes in OpenQuake
    assert len(sig.parameters) in [3, 4], "get_mean_stds should take 3-4 parameters depending on OpenQuake version"
    
    # Verify parameter names - the first three are the essential ones
    param_names = list(sig.parameters.keys())
    assert param_names[0] == "gsim", "First parameter should be 'gsim'"
    assert param_names[1] == "ctx", "Second parameter should be 'ctx'"
    assert param_names[2] == "imts", "Third parameter should be 'imts'"


def test_oq_run_calls_api_correctly(mock_openquake_api: Dict[str, Any], test_rupture_data: pd.DataFrame) -> None:
    """Test that our wrapper calls OpenQuake API correctly."""
    # Call our wrapper function
    result = oq_wrapper.oq_run(
        model_type=GMM.Br_13,  # Use Bradley 2013 as a test model
        tect_type=TectType.ACTIVE_SHALLOW,
        rupture_df=test_rupture_data,
        im="PGA"
    )
    
    # Use utility function to check the result structure
    assert_result_columns(result, "PGA")


@pytest.mark.parametrize('test_case', [
    pytest.param(
        {'model_type': GMM.Br_13, 'tect_type': TectType.ACTIVE_SHALLOW, 'im': "PGA", 'im_check': "PGA"},
        id="Bradley2013-PGA"
    ),
    pytest.param(
        {'model_type': GMM.ASK_14, 'tect_type': TectType.ACTIVE_SHALLOW, 'im': "PGA", 'im_check': "PGA"},
        id="ASK2014-PGA"
    ),
    pytest.param(
        {'model_type': GMM.Br_13, 'tect_type': TectType.ACTIVE_SHALLOW, 'im': "SA", 'period': 0.5, 'im_check': "pSA_0.5"},
        id="Bradley2013-SA(0.5)"
    ),
])
def test_multiple_gmm_models(
    mock_openquake_api: Dict[str, Any], 
    test_rupture_data: pd.DataFrame,
    test_case: Dict[str, Any]
) -> None:
    """Test that different GMM models can be run successfully.
    
    This test verifies multiple model types, tectonic settings and intensity measures
    to ensure our wrapper correctly handles various combinations.
    
    Parameters
    ----------
    mock_openquake_api : Dict[str, Any]
        Mocked OpenQuake API functions
    test_rupture_data : pd.DataFrame
        Test rupture data with default parameters
    test_case : Dict[str, Any]
        Parameterized test case with model settings
    """
    # Extract parameters
    model_type = test_case['model_type']
    tect_type = test_case['tect_type']
    im = test_case['im']
    im_check = test_case['im_check']
    
    # Common parameters
    params = {
        'model_type': model_type,
        'tect_type': tect_type,
        'rupture_df': test_rupture_data,
        'im': im
    }
    
    # Add periods for spectral acceleration if needed
    if im == "SA" and 'period' in test_case:
        params['periods'] = [test_case['period']]
    
    # Call our wrapper function
    result = oq_wrapper.oq_run(**params)
    
    # Use utility function to check the result structure
    assert_result_columns(result, im_check)


@pytest.mark.parametrize('attr_name, module', [
    ('PGA', imt),
    ('SA', imt),
    ('imt2tup', imt),
    ('StdDev', const),
    ('TRT', const),
])
def test_critical_attributes_exist(attr_name: str, module: Any) -> None:
    """Test that critical OpenQuake attributes exist."""
    assert hasattr(module, attr_name), f"OpenQuake should have {attr_name}"


@pytest.mark.parametrize('const_name, parent, expected_values', [
    ('StdDev', const, ['TOTAL', 'INTER_EVENT', 'INTRA_EVENT']),
    ('TRT', const, ['ACTIVE_SHALLOW_CRUST', 'SUBDUCTION_INTERFACE', 'SUBDUCTION_INTRASLAB']),
])
def test_critical_constants(const_name: str, parent: Any, expected_values: List[str]) -> None:
    """Test that critical OpenQuake constants have their expected values."""
    const_class = getattr(parent, const_name)
    for value in expected_values:
        assert hasattr(const_class, value), f"{const_name} should have {value}"


def test_all_models_exist() -> None:
    """Verify that all GMMs listed in OQ_MODELS exist in OpenQuake."""
    model_results = []
    
    for model_type, tect_dict in oq_wrapper.OQ_MODELS.items():
        for tect_type, model_class in tect_dict.items():
            # Check if model class is a partial function
            if hasattr(model_class, 'func'):
                model_ref = model_class.func
                # For partial functions, there should be a 'model' keyword arg
                assert 'model' in model_class.keywords, f"Model {model_type.name} should have 'model' keyword"
                model_class = model_class.keywords['model']
            else:
                model_ref = model_class
            
            # Use utility function to check model attributes
            is_valid, error_msg = check_model_attributes(model_class, model_type.name)
            assert is_valid, error_msg
            
            # Store information about the model tested for output
            model_results.append(f"{model_type.name} ({tect_type.name}): {model_class.__name__}")
    
    # Print summary of models verified
    print(f"\nVerified {len(model_results)} ground motion models:")
    for result in sorted(model_results):
        print(f"  • {result}")


@pytest.mark.parametrize('im_config', create_test_im_config(), ids=lambda config: config['id'])
def test_im_configurations(
    mock_openquake_api: Dict[str, Any], 
    test_rupture_data: pd.DataFrame,
    im_config: Dict[str, Any]
) -> None:
    """Test that different intensity measure configurations work as expected.
    
    This test verifies different intensity measure types (PGA, SA, etc.) 
    for various ground motion models and tectonic settings.
    
    Parameters
    ----------
    mock_openquake_api : Dict[str, Any]
        Mocked OpenQuake API functions
    test_rupture_data : pd.DataFrame
        Test rupture data with default parameters
    im_config : Dict[str, Any]
        Parameterized configuration for different intensity measures
    """
    # Extract parameters
    model_type = im_config['model']
    tect_type = im_config['tect_type']
    im = im_config['im']
    im_check = im_config['im_check']
    kwargs = im_config.get('kwargs', {})
    
    # Run the model
    result = oq_wrapper.oq_run(
        model_type=model_type,
        tect_type=tect_type,
        rupture_df=test_rupture_data,
        im=im,
        **kwargs
    )
    
    # Use utility function to check the result structure
    assert_result_columns(result, im_check)