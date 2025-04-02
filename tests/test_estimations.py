from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from conftest import MockPlane, openquake_test_wrapper

from empirical.util.estimations import (
    calculate_avg_strike_dip_rake,
    estimate_width_ASK14,
)


@pytest.mark.parametrize(
    "dip, mag, expected_width",
    [
        (30, 5.0, 3.162278),
        (45, 6.0, 8.912509),
        (60, 7.0, 20.784610),
        (75, 8.0, 18.634971),
        (90, 9.0, 18.000000),
    ],
)
def test_estimate_width_ASK14_values(
    dip: float, mag: float, expected_width: float
) -> None:
    """Test the estimate_width_ASK14 function correctly calculates individual widths.

    This parameterized test checks multiple combinations of dip and magnitude values
    against their expected width results.

    Parameters
    ----------
    dip : float
        Fault dip angle in degrees
    mag : float
        Earthquake magnitude
    expected_width : float
        Expected fault width value
    """
    # Convert input to pandas Series with single values
    dip_series = pd.Series([dip])
    mag_series = pd.Series([mag])

    # Calculate width
    width = estimate_width_ASK14(dip_series, mag_series)[0]

    # Verify width is correctly calculated
    assert width == pytest.approx(expected_width, rel=1e-2)

    # Verify width is positive
    assert width > 0




def test_estimate_width_ASK14_different_lengths() -> None:
    """Test that estimate_width_ASK14 raises ValueError when input series have different lengths."""
    # Create series with different lengths
    dip_series = pd.Series([30, 45, 60])
    mag_series = pd.Series([5.0, 6.0])

    # Check that a ValueError is raised
    with pytest.raises(
        ValueError, match="Input series 'dip' and 'mag' must have the same length"
    ):
        estimate_width_ASK14(dip_series, mag_series)


def test_calculate_avg_strike_dip_rake() -> None:
    """Test the calculate_avg_strike_dip_rake function correctly calculates average values."""
    # Create test data using our MockPlane instead of real Plane
    planes = [MockPlane(strike=0, dip=45), MockPlane(strike=90, dip=60)]
    rakes = [0, 90]  # First plane has a strike-slip, second has a dip-slip
    slips = [2, 1]  # First plane has twice the slip of the second

    # Calculate average values
    avg_strike, avg_dip, avg_rake = calculate_avg_strike_dip_rake(planes, rakes, slips)

    # Expected weighted averages (2/3 of first value + 1/3 of second value)
    expected_strike = 0 * 2 / 3 + 90 * 1 / 3
    expected_dip = 45 * 2 / 3 + 60 * 1 / 3
    expected_rake = 0 * 2 / 3 + 90 * 1 / 3

    # Verify results
    assert avg_strike == pytest.approx(expected_strike)
    assert avg_dip == pytest.approx(expected_dip)
    assert avg_rake == pytest.approx(expected_rake)


@openquake_test_wrapper
@pytest.mark.parametrize(
    "planes, rakes, slips, expected",
    [
        # Simple case: single plane
        ([MockPlane(strike=10, dip=20)], [30], [1.0], (10, 20, 30)),
        # Equal weight case: two planes with equal slip
        (
            [MockPlane(strike=0, dip=45), MockPlane(strike=90, dip=60)],
            [0, 90],
            [1, 1],
            (45, 52.5, 45),
        ),
        # Complex case: three planes with different slips
        (
            [
                MockPlane(strike=0, dip=30),
                MockPlane(strike=45, dip=45),
                MockPlane(strike=90, dip=60),
            ],
            [0, 45, 90],
            [3, 2, 1],
            (30, 40, 30),  # Corrected expected values based on actual calculation
        ),
    ],
    ids=["single-plane", "equal-weight", "weighted-average"],
)
def test_calculate_avg_strike_dip_rake_parametrized(
    planes: List[MockPlane],
    rakes: List[float],
    slips: List[float],
    expected: Tuple[float, float, float],
) -> None:
    """Test calculate_avg_strike_dip_rake with various input combinations.

    Parameters
    ----------
    planes : List[MockPlane]
        List of mock fault planes
    rakes : List[float]
        List of rake values for each plane
    slips : List[float]
        List of slip values for each plane
    expected : Tuple[float, float, float]
        Expected (strike, dip, rake) result
    """
    # Calculate average values
    result = calculate_avg_strike_dip_rake(planes, rakes, slips)

    # Verify results
    assert result[0] == pytest.approx(expected[0])
    assert result[1] == pytest.approx(expected[1])
    assert result[2] == pytest.approx(expected[2])

    return result  # Return for wrapper validation
