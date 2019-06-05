from empirical.util import empirical_factory
from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM
import pytest
import pickle

import os
from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM
import glob

RRUPS = [10, 70, 200]

TEST_DATA_SAVE_DIR = '/home/melody/Empirical_Engine/pickled/bradley_2013_sa/'
INPUT = "input"
OUTPUT = "output"
TEST_INPUT = os.path.join(TEST_DATA_SAVE_DIR, INPUT)
TEST_OUTPUT = os.path.join(TEST_DATA_SAVE_DIR, OUTPUT)

with open(os.path.join(TEST_OUTPUT, 'Bradley_2013_Sa_ret_val.P'), 'rb') as f:
    EXPECTED_RESULTS = pickle.load(f)

TEST_PARAMS = list(zip(RRUPS, EXPECTED_RESULTS))

FAULT = Fault()
FAULT.Mw = 10.5
FAULT.faultstyle = 'interface'
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

PERIODS = [0, 0.01, 0.40370172586, 0.5, 3.0, 8.6974900]
IM = 'pSA'


@pytest.mark.parametrize("test_rrup, expected_results",TEST_PARAMS)
def test_Bradley_2013_Sa(test_rrup, expected_results):
    SITE.Rrup = test_rrup
    test_results = empirical_factory.compute_gmm(FAULT, SITE, GMM.Br_13, IM, PERIODS)
    assert test_results == expected_results

