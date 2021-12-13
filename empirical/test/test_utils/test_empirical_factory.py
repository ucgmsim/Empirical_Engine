import pytest

from empirical.util import empirical_factory
from empirical.util.classdef import SiteClass


VS30S = [100, 201, 301, 601, 1101]
EXPECTED_RESULTS = [
    (100, SiteClass.SOFTSOIL),
    (201, SiteClass.MEDIUMSOIL),
    (301, SiteClass.HARDSOIL),
    (601, SiteClass.ROCK),
    (1101, SiteClass.HARDROCK),
]


@pytest.mark.parametrize("test_vs30, expected_siteclass", EXPECTED_RESULTS)
def test_determine_siteclass(test_vs30, expected_siteclass):
    print(test_vs30)
    assert empirical_factory.determine_siteclass(test_vs30) == expected_siteclass
