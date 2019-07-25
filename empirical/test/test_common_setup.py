import pytest

from qcore import testing

FOLDERS = [
    (
        "Empirical_Engine",
        "https://qc-s3-autotest.s3-ap-southeast-2.amazonaws.com/testing/Empirical_Engine/Empirical_Engine.zip",
    )
]


@pytest.yield_fixture(scope="session", autouse=True)
def set_up(request):
    data_locations = testing.test_set_up(FOLDERS)
    yield data_locations[0]
    testing.test_tear_down(data_locations)
