import os
import pytest
import pickle

from empirical.test.test_common_setup import set_up
from empirical.util import empirical_factory
from empirical.util.classdef import Site, Fault, GMM

IM = "pSA"
RRUPS = [10, 70, 200]
PERIODS = [0, 0.01, 0.40370172586, 0.5, 3.0, 8.6974900]

FAULT = Fault()
FAULT.Mw = 10.5
FAULT.ztor = 0
FAULT.rake = 180
FAULT.dip = 45

SITE = Site()
SITE.Rjb = 10
SITE.vs30 = 500
SITE.Rx = -1
SITE.Rtvz = 50


@pytest.mark.parametrize("test_rrup", RRUPS)
def test_Bradley_2013_Sa(set_up, test_rrup):
    SITE.Rrup = test_rrup
    test_results = empirical_factory.compute_gmm(FAULT, SITE, GMM.Br_13, IM, PERIODS)

    with open(
        os.path.join(
            set_up, "output", "Bradley_2013_Sa_ret_val_rrup_{}.P".format(test_rrup)
        ),
        "rb",
    ) as f:
        expected_results = pickle.load(f)

    assert test_results == expected_results
