"""Tests for the z_model_calculations module."""

import numpy as np
import pandas as pd
import pytest
from pytest import approx

from empirical.util import z_model_calculations as zmc
from empirical.util.classdef import GMM


@pytest.fixture
def vs30_values():
    """Return a range of vs30 values for testing."""
    return np.array([200.0, 400.0, 760.0, 1100.0])


@pytest.mark.parametrize(
    "region",
    ["Cascadia", "Japan", "NewZealand", "Taiwan"]
)
def test_kuehn_20_calc_z(vs30_values, region):
    """Test kuehn_20_calc_z function returns valid values for different regions."""
    result = zmc.kuehn_20_calc_z(vs30_values, region)
    
    # The result should be a log value
    assert isinstance(result, np.ndarray)
    assert len(result) == len(vs30_values)
    
    # Values should be positive and decrease with increasing vs30
    assert np.all(result > 0)
    assert np.all(np.diff(result) < 0)
    
    # Test with a single value
    single_result = zmc.kuehn_20_calc_z(760.0, region)
    assert single_result > 0


def test_kuehn_20_calc_z_invalid_region():
    """Test kuehn_20_calc_z raises KeyError for invalid region."""
    with pytest.raises(KeyError, match="Region InvalidRegion not supported"):
        zmc.kuehn_20_calc_z(760.0, "InvalidRegion")


def test_chiou_young_14_calc_z1p0(vs30_values):
    """Test chiou_young_14_calc_z1p0 returns valid values for global and Japan regions."""
    # Test global model
    global_result = zmc.chiou_young_14_calc_z1p0(vs30_values)
    assert isinstance(global_result, np.ndarray)
    assert len(global_result) == len(vs30_values)
    assert np.all(global_result > 0)
    assert np.all(np.diff(global_result) < 0)  # Values should decrease with increasing vs30
    
    # Test Japan model
    japan_result = zmc.chiou_young_14_calc_z1p0(vs30_values, region="Japan")
    assert isinstance(japan_result, np.ndarray)
    assert len(japan_result) == len(vs30_values)
    assert np.all(japan_result > 0)
    assert np.all(np.diff(japan_result) < 0)


def test_mod_chiou_young_14_calc_z1p0(vs30_values):
    """Test mod_chiou_young_14_calc_z1p0 returns valid values with modified coefficient."""
    # Test global model
    global_result = zmc.mod_chiou_young_14_calc_z1p0(vs30_values)
    assert isinstance(global_result, np.ndarray)
    assert len(global_result) == len(vs30_values)
    assert np.all(global_result > 0)
    assert np.all(np.diff(global_result) < 0)  # Values should decrease with increasing vs30
    
    # Test Japan model
    japan_result = zmc.mod_chiou_young_14_calc_z1p0(vs30_values, region="Japan")
    assert isinstance(japan_result, np.ndarray)
    assert len(japan_result) == len(vs30_values)
    assert np.all(japan_result > 0)
    assert np.all(np.diff(japan_result) < 0)
    
    # Compare the Japan results should be the same for both functions
    cy14_japan = zmc.chiou_young_14_calc_z1p0(vs30_values, region="Japan")
    mod_cy14_japan = zmc.mod_chiou_young_14_calc_z1p0(vs30_values, region="Japan")
    assert np.allclose(cy14_japan, mod_cy14_japan)
    
    # Global models should be different
    cy14_global = zmc.chiou_young_14_calc_z1p0(vs30_values)
    mod_cy14_global = zmc.mod_chiou_young_14_calc_z1p0(vs30_values)
    assert not np.allclose(cy14_global, mod_cy14_global)


def test_campbell_bozorgina_14_calc_z2p5(vs30_values):
    """Test campbell_bozorgina_14_calc_z2p5 returns valid values for different regions."""
    # Test global model
    global_result = zmc.campbell_bozorgina_14_calc_z2p5(vs30_values)
    assert isinstance(global_result, np.ndarray)
    assert len(global_result) == len(vs30_values)
    assert np.all(global_result > 0)
    assert np.all(np.diff(global_result) < 0)  # Values should decrease with increasing vs30
    
    # Test Japan model
    japan_result = zmc.campbell_bozorgina_14_calc_z2p5(vs30_values, region="Japan")
    assert isinstance(japan_result, np.ndarray)
    assert len(japan_result) == len(vs30_values)
    assert np.all(japan_result > 0)
    assert np.all(np.diff(japan_result) < 0)
    
    # Japan values should be higher than global values
    assert np.all(japan_result > global_result)


def test_chiou_young_08_calc_z1p0(vs30_values):
    """Test chiou_young_08_calc_z1p0 returns valid values."""
    result = zmc.chiou_young_08_calc_z1p0(vs30_values)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(vs30_values)
    assert np.all(result > 0)
    assert np.all(np.diff(result) < 0)  # Values should decrease with increasing vs30
    
    # Test with pandas DataFrame
    df_vs30 = pd.DataFrame({"vs30": vs30_values})
    df_result = zmc.chiou_young_08_calc_z1p0(df_vs30["vs30"])
    assert np.allclose(df_result, result)


def test_chiou_young_08_calc_z2p5():
    """Test chiou_young_08_calc_z2p5 calculates correct values from z1p0 or z1p5."""
    # Test with z1p0
    z1p0 = np.array([0.1, 0.2, 0.3])
    expected_from_z1p0 = 0.519 + 3.595 * z1p0
    result_from_z1p0 = zmc.chiou_young_08_calc_z2p5(z1p0=z1p0)
    assert np.allclose(result_from_z1p0, expected_from_z1p0)
    
    # Test with z1p5
    z1p5 = np.array([0.2, 0.4, 0.6])
    expected_from_z1p5 = 0.636 + 1.549 * z1p5
    result_from_z1p5 = zmc.chiou_young_08_calc_z2p5(z1p5=z1p5)
    assert np.allclose(result_from_z1p5, expected_from_z1p5)
    
    # Test with neither provided
    with pytest.raises(ValueError, match="no z2p5 able to be estimated"):
        zmc.chiou_young_08_calc_z2p5()


@pytest.mark.parametrize(
    "region",
    ["Cascadia", "Japan"]
)
def test_abrahamson_gulerce_20_calc_z2p5(vs30_values, region):
    """Test abrahamson_gulerce_20_calc_z2p5 returns valid values for different regions."""
    result = zmc.abrahamson_gulerce_20_calc_z2p5(vs30_values, region)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(vs30_values)
    assert np.all(result > 0)
    # Values should generally decrease with increasing vs30, but may plateau at limits
    assert result[0] >= result[-1]


def test_abrahamson_gulerce_20_calc_z2p5_invalid_region():
    """Test abrahamson_gulerce_20_calc_z2p5 raises ValueError for invalid region."""
    with pytest.raises(ValueError, match="Does not support region InvalidRegion"):
        zmc.abrahamson_gulerce_20_calc_z2p5(760.0, "InvalidRegion")


@pytest.mark.parametrize(
    "region",
    ["Cascadia", "Japan"]
)
def test_parker_20_calc_z2p5(vs30_values, region):
    """Test parker_20_calc_z2p5 returns valid values for different regions."""
    result = zmc.parker_20_calc_z2p5(vs30_values, region)
    assert isinstance(result, np.ndarray)
    assert len(result) == len(vs30_values)
    assert np.all(result > 0)
    assert np.all(np.diff(result) < 0)  # Values should decrease with increasing vs30


def test_parker_20_calc_z2p5_invalid_region():
    """Test parker_20_calc_z2p5 raises ValueError for invalid region."""
    with pytest.raises(ValueError, match="Does not support region InvalidRegion"):
        zmc.parker_20_calc_z2p5(760.0, "InvalidRegion")


@pytest.mark.parametrize(
    "model, region, expected_type",
    [
        (GMM.CY_14, None, "z1pt0"),
        (GMM.CY_14, "Japan", "z1pt0"),
        (GMM.ASK_14, None, "z1pt0"),
        (GMM.CB_14, None, "z2pt5"),
        (GMM.K_20, "NewZealand", "z1pt0"),
        (GMM.AG_20, "Cascadia", "z2pt5"),
        (GMM.P_20, "Japan", "z2pt5"),
    ]
)
def test_calc_z_for_model(model, region, expected_type):
    """Test calc_z_for_model returns correct types for different models and regions."""
    result_value, result_type = zmc.calc_z_for_model(model, 760.0, region)
    # Check that we get back a valid value
    assert isinstance(result_value, (float, np.ndarray))
    assert result_value > 0
    
    # Check that the returned type matches what we expect
    assert result_type == expected_type
    
    # Test with array
    vs30_array = np.array([760.0, 1100.0])
    array_values, array_type = zmc.calc_z_for_model(model, vs30_array, region)
    assert len(array_values) == 2
    assert np.all(array_values > 0)
    assert array_type == expected_type
    
    # The first value should match the single value result
    assert np.isclose(array_values[0], result_value)


def test_calc_z_for_model_global_region():
    """Test calc_z_for_model treats 'Global' string as None."""
    value1, type1 = zmc.calc_z_for_model(GMM.CY_14, 760.0, None)
    value2, type2 = zmc.calc_z_for_model(GMM.CY_14, 760.0, "Global")
    assert np.isclose(value1, value2)
    assert type1 == type2


def test_calc_z_for_model_invalid_inputs():
    """Test calc_z_for_model raises appropriate errors for invalid inputs."""
    # Invalid model
    with pytest.raises(KeyError):
        zmc.calc_z_for_model(GMM.GA_11, 760.0)
    
    # Invalid region for model
    with pytest.raises(KeyError):
        zmc.calc_z_for_model(GMM.CY_14, 760.0, "InvalidRegion")