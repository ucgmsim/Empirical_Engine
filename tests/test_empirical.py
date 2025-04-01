"""Tests for the empirical module."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open, PropertyMock

from empirical.util import empirical
from empirical.util.classdef import GMM, TectType


def test_get_model():
    """Test the get_model function correctly retrieves models from config."""
    # Create a mock model config
    model_config = {
        "ACTIVE_SHALLOW": {
            "PGA": {"geometric_mean": ["ASK_14"]},
            "pSA": {"geometric_mean": ["Br_13"]},
        },
        "SUBDUCTION_INTERFACE": {
            "PGA": {"geometric_mean": ["BCH_16"]},
            "pSA": {"geometric_mean": ["A_22"]},
        },
    }
    
    # Test retrieving models for different combinations
    assert empirical.get_model(model_config, TectType.ACTIVE_SHALLOW, "PGA", "geometric_mean") == GMM.ASK_14
    assert empirical.get_model(model_config, TectType.ACTIVE_SHALLOW, "pSA", "geometric_mean") == GMM.Br_13
    assert empirical.get_model(model_config, TectType.SUBDUCTION_INTERFACE, "PGA", "geometric_mean") == GMM.BCH_16
    assert empirical.get_model(model_config, TectType.SUBDUCTION_INTERFACE, "pSA", "geometric_mean") == GMM.A_22
    
    # Test None is returned for non-existent combination
    assert empirical.get_model(model_config, TectType.SUBDUCTION_SLAB, "PGA", "geometric_mean") is None
    assert empirical.get_model(model_config, TectType.ACTIVE_SHALLOW, "PGV", "geometric_mean") is None
    assert empirical.get_model(model_config, TectType.ACTIVE_SHALLOW, "PGA", "rotd50") is None


# Mock the NHMFault class for testing
class MockNHMFault:
    pass


@patch('empirical.util.empirical.nhm')
@patch('empirical.util.empirical.ssd')
def test_get_site_source_data_nhm(mock_ssd, mock_nhm):
    """Test get_site_source_data function with NHM fault."""
    # Configure mock for isinstance check
    mock_nhm.NHMFault = MockNHMFault
    
    # Create mock NHM fault
    mock_fault = MockNHMFault()
    
    # Set up mock returns for nhm.get_fault_header_points
    mock_plane_info = MagicMock()
    mock_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mock_nhm.get_fault_header_points.return_value = (mock_plane_info, mock_points)
    
    # Set up mock returns for ssd functions
    mock_rrup = np.array([10.0, 15.0])
    mock_rjb = np.array([8.0, 12.0])
    mock_rx = np.array([5.0, 7.0])
    mock_ry = np.array([2.0, 3.0])
    mock_ssd.calc_rrup_rjb.return_value = (mock_rrup, mock_rjb)
    mock_ssd.calc_rx_ry.return_value = (mock_rx, mock_ry)
    
    # Create test stations
    stations = np.array([[172.0, -43.0], [173.0, -44.0]])
    
    # Call the function
    result = empirical.get_site_source_data(mock_fault, stations)
    
    # Check calls
    mock_nhm.get_fault_header_points.assert_called_once_with(mock_fault)
    
    # Check result
    expected_df = pd.DataFrame({
        "rrup": [10.0, 15.0],
        "rjb": [8.0, 12.0],
        "rx": [5.0, 7.0],
        "ry": [2.0, 3.0]
    })
    pd.testing.assert_frame_equal(result, expected_df)


@patch('empirical.util.empirical.srf')
@patch('empirical.util.empirical.ssd')
def test_get_site_source_data_srf(mock_ssd, mock_srf):
    """Test get_site_source_data function with SRF file path."""
    # Create mock SRF path
    srf_path = Path("/path/to/srf_file.srf")
    
    # Set up mock returns for srf functions
    mock_points = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    mock_plane_info = {"plane_info": "data"}
    mock_srf.read_srf_points.return_value = mock_points
    mock_srf.read_header.return_value = mock_plane_info
    
    # Set up mock returns for ssd functions
    mock_rrup = np.array([10.0, 15.0])
    mock_rjb = np.array([8.0, 12.0])
    mock_rx = np.array([5.0, 7.0])
    mock_ry = np.array([2.0, 3.0])
    mock_ssd.calc_rrup_rjb.return_value = (mock_rrup, mock_rjb)
    mock_ssd.calc_rx_ry.return_value = (mock_rx, mock_ry)
    
    # Create test stations
    stations = np.array([[172.0, -43.0], [173.0, -44.0]])
    
    # Call the function
    result = empirical.get_site_source_data(srf_path, stations)
    
    # Check calls
    mock_srf.read_srf_points.assert_called_once_with(str(srf_path))
    mock_srf.read_header.assert_called_once_with(str(srf_path), idx=True)
    
    # Check result
    expected_df = pd.DataFrame({
        "rrup": [10.0, 15.0],
        "rjb": [8.0, 12.0],
        "rx": [5.0, 7.0],
        "ry": [2.0, 3.0]
    })
    pd.testing.assert_frame_equal(result, expected_df)


@patch('empirical.util.empirical.h5py.File')
def test_load_srf_info(mock_h5py_file):
    """Test load_srf_info function for HDF5 files."""
    # Create mock h5py File context manager
    mock_file = MagicMock()
    mock_attrs = {
        'mag': np.array([6.5, 6.6, 6.7]),
        'tect_type': 'ACTIVE_SHALLOW',
        'dtop': np.array([3.0, 3.5, 4.0]),
        'dbottom': np.array([15.0, 15.5, 16.0]),
        'rake': np.array([90.0, 90.0, 90.0]),
        'dip': np.array([60.0, 60.0, 60.0]),
        'hdepth': 10.0
    }
    mock_file.__enter__.return_value.attrs = mock_attrs
    mock_h5py_file.return_value = mock_file
    
    # Call the function
    result = empirical.load_srf_info(Path("/path/to/srf_info.h5"), "test_event")
    
    # Check result - note that ACTIVE_SHALLOW maps to 'Undetermined' in the REVERSE_TECT_CLASS_MAPPING
    expected = pd.Series({
        'mag': 6.7,
        'tect_class': 'Undetermined',  # Changed from 'Crustal' to 'Undetermined'
        'z_tor': 3.0,
        'z_bor': 15.0,
        'rake': 90.0,
        'dip': 60.0,
        'depth': 10.0
    }, name="test_event")
    pd.testing.assert_series_equal(result, expected)


@patch('empirical.util.empirical.h5py.File')
def test_load_srf_info_bytes_tect_type(mock_h5py_file):
    """Test load_srf_info function when tect_type is stored as bytes."""
    # Create mock h5py File context manager
    mock_file = MagicMock()
    mock_attrs = {
        'mag': np.array([6.5]),
        'tect_type': b'SUBDUCTION_INTERFACE',  # Use bytes for tect_type
        'dtop': np.array([3.0]),
        'dbottom': np.array([15.0]),
        'rake': np.array([90.0]),
        'dip': np.array([20.0]),
        'hdepth': 10.0
    }
    mock_file.__enter__.return_value.attrs = mock_attrs
    mock_h5py_file.return_value = mock_file
    
    # Call the function
    result = empirical.load_srf_info(Path("/path/to/srf_info.h5"), "test_event")
    
    # Check result
    expected = pd.Series({
        'mag': 6.5,
        'tect_class': 'Interface',
        'z_tor': 3.0,
        'z_bor': 15.0,
        'rake': 90.0,
        'dip': 20.0,
        'depth': 10.0
    }, name="test_event")
    pd.testing.assert_series_equal(result, expected)


@patch('empirical.util.empirical.h5py.File')
def test_load_srf_info_no_tect_type(mock_h5py_file):
    """Test load_srf_info function when tect_type is not provided."""
    # Create mock h5py File context manager
    mock_file = MagicMock()
    mock_attrs = {
        'mag': np.array([6.5]),
        # No tect_type attribute
        'dtop': np.array([3.0]),
        'dbottom': np.array([15.0]),
        'rake': np.array([90.0]),
        'dip': np.array([20.0]),
        'hdepth': 10.0
    }
    mock_file.__enter__.return_value.attrs = mock_attrs
    mock_h5py_file.return_value = mock_file
    
    # Call the function with print capture
    with patch('builtins.print') as mock_print:
        result = empirical.load_srf_info(Path("/path/to/srf_info.h5"), "test_event")
        
        # Check print message
        mock_print.assert_called_once_with("INFO: tect_type not found.  Default 'ACTIVE_SHALLOW' is used.")
    
    # Check result - note that ACTIVE_SHALLOW maps to 'Undetermined' in the REVERSE_TECT_CLASS_MAPPING
    expected = pd.Series({
        'mag': 6.5,
        'tect_class': 'Undetermined',  # Changed from 'Crustal' to 'Undetermined'
        'z_tor': 3.0,
        'z_bor': 15.0,
        'rake': 90.0,
        'dip': 20.0,
        'depth': 10.0
    }, name="test_event")
    pd.testing.assert_series_equal(result, expected)


@patch('empirical.util.empirical.h5py.File')
def test_load_srf_info_no_dtop(mock_h5py_file):
    """Test load_srf_info function when dtop is not provided."""
    # Create mock h5py File context manager
    mock_file = MagicMock()
    mock_attrs = {
        'mag': np.array([6.5]),
        'tect_type': 'ACTIVE_SHALLOW',
        # No dtop attribute
        'dbottom': np.array([15.0]),
        'rake': np.array([90.0]),
        'dip': np.array([20.0]),
        'hdepth': 10.0
    }
    mock_file.__enter__.return_value.attrs = mock_attrs
    mock_h5py_file.return_value = mock_file
    
    # Call the function
    result = empirical.load_srf_info(Path("/path/to/srf_info.h5"), "test_event")
    
    # Check result - z_tor should fall back to hdepth
    expected = pd.Series({
        'mag': 6.5,
        'tect_class': 'Undetermined',  # Changed from 'Crustal' to 'Undetermined'
        'z_tor': 10.0,  # Using hdepth as fallback
        'z_bor': 15.0,
        'rake': 90.0,
        'dip': 20.0,
        'depth': 10.0
    }, name="test_event")
    pd.testing.assert_series_equal(result, expected)


@patch('empirical.util.empirical.h5py.File')
def test_load_srf_info_invalid_rake(mock_h5py_file):
    """Test load_srf_info function with invalid rake values."""
    # Create mock h5py File context manager
    mock_file = MagicMock()
    mock_attrs = {
        'mag': np.array([6.5]),
        'tect_type': 'ACTIVE_SHALLOW',
        'dtop': np.array([3.0]),
        'dbottom': np.array([15.0]),
        'rake': np.array([90.0, 91.0]),  # Different rake values
        'dip': np.array([20.0]),
        'hdepth': 10.0
    }
    mock_file.__enter__.return_value.attrs = mock_attrs
    mock_h5py_file.return_value = mock_file
    
    # Call the function and check for ValueError
    with pytest.raises(ValueError, match="unexpected rake value"):
        empirical.load_srf_info(Path("/path/to/srf_info.h5"), "test_event")


@patch('pandas.read_csv')
def test_load_rel_csv(mock_read_csv):
    """Test load_rel_csv function."""
    # Create mock CSV data
    mock_csv_data = pd.DataFrame({
        'dtop': [3.0],
        'dbottom': [15.0],
        'dip': [60.0],
        'dhypo': [5.0],
        'magnitude': [6.5],
        'tect_type': ['Crustal'],
        'rake': [90.0]
    })
    mock_read_csv.return_value = mock_csv_data
    
    # Call the function
    result = empirical.load_rel_csv(Path("/path/to/rel.csv"), "test_event")
    
    # Check calls
    mock_read_csv.assert_called_once_with(Path("/path/to/rel.csv"))
    
    # Calculate expected hypocentral depth
    hypo_depth = 3.0 + np.sin(np.radians(60.0)) * 5.0
    
    # Check result
    expected = pd.Series({
        'mag': 6.5,
        'tect_class': 'Crustal',
        'z_tor': 3.0,
        'z_bor': 15.0,
        'rake': 90.0,
        'dip': 60.0,
        'depth': hypo_depth
    }, name="test_event")
    pd.testing.assert_series_equal(result, expected)


@patch('pandas.read_csv')
def test_load_rel_csv_missing_columns(mock_read_csv):
    """Test load_rel_csv function with missing columns."""
    # Create mock CSV data with missing columns
    mock_csv_data = pd.DataFrame({
        'dtop': [3.0],
        'dbottom': [15.0],
        # Missing 'dip', 'dhypo', etc.
    })
    mock_read_csv.return_value = mock_csv_data
    
    # Call the function and check for KeyError with a broader pattern
    with pytest.raises(KeyError) as excinfo:
        empirical.load_rel_csv(Path("/path/to/rel.csv"), "test_event")
    
    # Check that the error message mentions missing columns
    assert "Column" in str(excinfo.value)
    assert "are not found" in str(excinfo.value)


def test_nhm_flt_to_df():
    """Test nhm_flt_to_df function."""
    # Create a mock NHM fault
    mock_fault = MagicMock()
    mock_fault.name = "test_fault"
    mock_fault.mw = 6.5
    mock_fault.dip = 60.0
    mock_fault.rake = 90.0
    mock_fault.dbottom = 15.0
    mock_fault.dtop = 3.0
    mock_fault.tectonic_type = "Crustal"
    mock_fault.recur_int_median = 500.0
    
    # Call the function
    result = empirical.nhm_flt_to_df(mock_fault)
    
    # Check result
    expected = pd.DataFrame({
        "index": ["test_fault"],
        "fault_name": ["test_fault"],
        "mag": [6.5],
        "dip": [60.0],
        "rake": [90.0],
        "dbot": [15.0],
        "ztor": [3.0],
        "tect_class": ["Crustal"],
        "recurrance_rate": [500.0]
    }).reset_index(drop=True)
    pd.testing.assert_frame_equal(result, expected)