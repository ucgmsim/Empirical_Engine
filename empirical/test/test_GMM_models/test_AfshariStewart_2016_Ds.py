import os
import pickle
import pytest

from empirical.util.empirical_factory import compute_gmm
from empirical.util.classdef import Site, Fault, GMM

FAULT = Fault()
FAULT.faultstyle = 'SHALLOWCRUSTAL'
FAULT.ztor = 0
FAULT.Ztor = 0
FAULT.rake = 180
FAULT.dip = 45

SITE = Site()
SITE.Rjb = 10
SITE.vs30 = 500
SITE.V30 = 500
SITE.V30measured = None
SITE.Rx = -1

SITE.Rtvz = 50
SITE.z2p5 = 0.9186718412435146

RRUPS = [10, 70, 200]

BENCHMARK_DIR = '/home/melody/Empirical_Engine/pickled/as_2016/output'

AS_M = [5, 6.25, 7.5]
AS_IMS = ['Ds575', 'Ds595', 'Ds2080']

TEST_PARAMS = [(rrup, mag, im) for rrup in RRUPS for mag in AS_M for im in AS_IMS]


@pytest.mark.parametrize("test_rrup, test_mag, test_im", TEST_PARAMS)
def test_as_2016(test_rrup, test_mag, test_im):
    FAULT.Mw = test_mag
    SITE.Rrup = test_rrup
    test_results = compute_gmm(FAULT, SITE, GMM.AS_16, test_im)

    with open(os.path.join(BENCHMARK_DIR, 'as_2016_ret_val_rrup_{}_mag_{}_{}.P'.format(test_rrup, test_mag, test_im)), 'rb') as f:
        expected_results = pickle.load(f)

    assert test_results == expected_results
