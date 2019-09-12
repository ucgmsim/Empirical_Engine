import os
import pickle
import pytest

from empirical.test.test_common_setup import set_up
from empirical.util.empirical_factory import compute_gmm
from empirical.util.classdef import Site, Fault, GMM

RRUPS = [10, 70, 200]

CB_M = [4.0, 5.4, 7.8]
CB_IMS = ["CAV", "AI"]

TEST_PARAMS = [(rrup, mag, im) for rrup in RRUPS for mag in CB_M for im in CB_IMS]

FAULT = Fault()
FAULT.ztor = 0
FAULT.rake = 180
FAULT.dip = 45

SITE = Site()
SITE.Rjb = 10
SITE.vs30 = 500
SITE.Rx = -1
SITE.Rtvz = 50
SITE.z2p5 = 0.9186718412435146


@pytest.mark.parametrize("test_rrup, test_mag, test_im", TEST_PARAMS)
def test_cb_2012(set_up, test_rrup, test_mag, test_im):
    SITE.Rrup = test_rrup
    FAULT.Mw = test_mag
    test_results = compute_gmm(FAULT, SITE, GMM.CB_12, test_im)

    with open(
        os.path.join(
            set_up,
            "output",
            "cb_2012_ret_val_rrup_{}_mag_{}_{}.P".format(test_rrup, test_mag, test_im),
        ),
        "rb",
    ) as f:
        expected_results = pickle.load(f)

    assert test_results == expected_results
