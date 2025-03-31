import numpy as np
import pandas as pd
import pytest

from empirical.util.estimations import estimate_width_ASK14

def test_estimate_width_ASK14():
    """Test the estimate_width_ASK14 function correctly calculates widths."""
    # Create some test data
    dips = [30, 45, 60, 75, 90]
    mags = [5.0, 6.0, 7.0, 8.0, 9.0]
    
    # Convert lists to pandas Series
    dip_series = pd.Series(dips)
    mag_series = pd.Series(mags)
    
    # Calculate widths
    widths = estimate_width_ASK14(dip_series, mag_series)
    
    # Verify widths are correctly calculated
    for i in range(len(dips)):
        dip = dips[i]
        mag = mags[i]
        
        expected_width = min(18 / np.sin(np.radians(dip)), 10 ** (-1.75 + 0.45 * mag))
        # Using explicit relative tolerance of 1e-6 (default value)
        assert widths[i] == pytest.approx(expected_width, rel=1e-6)

    # Test that all widths are positive
    assert all(widths > 0)
    
    # Test that shape of output matches input
    assert len(widths) == len(dips)


def test_estimate_width_ASK14_different_lengths():
    """Test that estimate_width_ASK14 raises ValueError when input series have different lengths."""
    # Create series with different lengths
    dip_series = pd.Series([30, 45, 60])
    mag_series = pd.Series([5.0, 6.0])
    
    # Check that a ValueError is raised
    with pytest.raises(ValueError, match="Input series 'dip' and 'mag' must have the same length"):
        estimate_width_ASK14(dip_series, mag_series)