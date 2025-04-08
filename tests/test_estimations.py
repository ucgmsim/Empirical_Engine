from typing import Optional

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from empirical.util import estimations
from source_modelling import sources


def coordinate(lat: float, lon: float, depth: Optional[float] = None) -> np.ndarray:
    """Create a coordinate array from latitude, longitude, and optional depth."""
    if depth is not None:
        return np.array([lat, lon, depth])
    return np.array([lat, lon])


# Define strategy for dip Series (values between 0.1-90 degrees)
# Avoiding 0 degrees to prevent division by zero in sin calculation
@st.composite
def dip_series_strategy(draw: st.DrawFn) -> pd.Series:
    # Generate a list of floats between 0.1 and 90 degrees
    dip_values = draw(st.lists(st.floats(min_value=0.1, max_value=90), min_size=1))
    return pd.Series(dip_values)


# Define strategy for magnitude Series (values between 0-10)
@st.composite
def mag_series_strategy(draw: st.DrawFn) -> pd.Series:
    # Generate a list of floats between 0 and 10
    mag_values = draw(st.lists(st.floats(min_value=0, max_value=10), min_size=1))
    return pd.Series(mag_values)


@given(dip=dip_series_strategy(), mag=mag_series_strategy())
def test_estimate_width_ASK14(dip: pd.Series, mag: pd.Series) -> None:  # noqa: N802
    # Calculate the expected result using the formula directly
    expected = np.minimum(18 / np.sin(np.radians(dip)), 10 ** (-1.75 + 0.45 * mag))

    # Get the result from the function
    result = estimations.estimate_width_ASK14(dip, mag)

    # Check that the results match
    pd.testing.assert_series_equal(result, expected)


@st.composite
def rupture_data_strategy(draw: st.DrawFn) -> tuple:
    # First, determine how many planes we'll have
    n_planes = draw(st.integers(min_value=1, max_value=100))

    # Generate n_planes of each component
    planes = draw(
        st.lists(
            st.builds(
                sources.Plane.from_centroid_strike_dip,
                length=st.floats(0.1, 1000),
                width=st.floats(0.1, 1000),
                strike=st.floats(0, 179),
                dip_dir=st.floats(0, 179),
                dip=st.floats(0.1, 90),
                centroid=st.builds(
                    coordinate,
                    lat=st.floats(-50, -31),
                    lon=st.floats(160, 180),
                    depth=st.floats(1, 10),
                ),
            ),
            min_size=n_planes,
            max_size=n_planes,  # Exact size
        )
    )

    # Generate exactly n_planes rake values
    plane_avg_rake = draw(
        st.lists(st.floats(-180, 180), min_size=n_planes, max_size=n_planes)
    )

    # Generate exactly n_planes slip values (in m)
    plane_total_slip = draw(
        st.lists(st.floats(0.1, 10), min_size=n_planes, max_size=n_planes)
    )

    return planes, plane_avg_rake, plane_total_slip


@given(rupture_data=rupture_data_strategy())
def test_calculate_avg_strike_dip_rake(
    rupture_data: tuple[list[sources.Plane], list[float], list[float]],
) -> None:
    planes, plane_avg_rake, plane_total_slip = rupture_data

    # Now call the function and test the results
    avg_strike, avg_dip, avg_rake = estimations.calculate_avg_strike_dip_rake(
        planes, plane_avg_rake, plane_total_slip
    )

    # Calculate the expected result using the formula directly
    slip_weights = np.asarray(plane_total_slip) / sum(plane_total_slip)
    avg_strike = np.average([plane.strike for plane in planes], weights=slip_weights)
    avg_dip = np.average([plane.dip for plane in planes], weights=slip_weights)
    avg_rake = np.average(plane_avg_rake, weights=slip_weights)

    assert avg_strike == pytest.approx(avg_strike, rel=1e-3)
    assert avg_dip == pytest.approx(avg_dip, rel=1e-3)
    assert avg_rake == pytest.approx(avg_rake, rel=1e-3)