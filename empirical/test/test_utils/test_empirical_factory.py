import pytest

from empirical.util import empirical_factory
from empirical.util.classdef import SiteClass


VS30S = [100, 200, 300, 600, 1100]
EXPECTED_RESULTS = [
    (100, SiteClass.SOFTSOIL),
    (200, SiteClass.MEDIUMSOIL),
    (300, SiteClass.HARDSOIL),
    (600, SiteClass.ROCK),
    (1100, SiteClass.HARDROCK),
]


@pytest.mark.parametrize("test_vs30, expected_siteclass", EXPECTED_RESULTS)
def test_determine_siteclass(test_vs30, expected_siteclass):
    assert empirical_factory.determine_siteclass(test_vs30) == expected_siteclass
