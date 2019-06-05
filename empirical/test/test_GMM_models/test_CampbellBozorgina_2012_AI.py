from empirical.util.empirical_factory import compute_gmm
from empirical.util.classdef import Site, Fault, GMM
import pickle
import os

fault = Fault()
fault.faultstyle = 'SHALLOWCRUSTAL'
fault.ztor = 0
fault.Ztor = 0
fault.rake = 180
fault.dip = 45

site = Site()
site.Rjb = 10
site.vs30 = 500
site.V30 = 500
site.V30measured = None
site.Rx = -1

site.Rtvz = 50
site.z2p5 = 0.9186718412435146

RRUP = [10, 70, 200]

# TEST FOR CB
CB_M = [4.0, 5.4, 7.8]
CB_IMS = ['CAV', 'AI']



TEST_DATA_SAVE_DIR = '/home/melody/Empirical_Engine/pickled/cb_2012/'
INPUT = "input"
OUTPUT = "output"
TEST_INPUT = os.path.join(TEST_DATA_SAVE_DIR, INPUT)
TEST_OUTPUT = os.path.join(TEST_DATA_SAVE_DIR, OUTPUT)

for im in CB_IMS:
    with open(os.path.join(TEST_OUTPUT, 'cb_2012_{}_ret_val.P'.format(im)), 'rb') as f:
        result = pickle.load(f)

TEST_PARAMS = list(zip(RRUPS, EXPECTED_RESULTS))
@pytest.mark.parametrize("test_rrup, expected_results", TEST_PARAMS)
def test_Bradley_2013_Sa(test_rrup, expected_results):
    SITE.Rrup = test_rrup
    test_results = empirical_factory.compute_gmm(FAULT, SITE, GMM.Br_13, IM, PERIODS)
    assert test_results == expected_results


def test_cb_2012():
for im in CB_IMS:
    print(im)
    all = []
    for rrup in RRUP:
        site.Rrup = rrup
        for mag in CB_M:
            fault.Mw = mag
            results = compute_gmm(fault, site, GMM.CB_12, im)
            print(results)
            print('\n')
            all.append(results)
    with open('/home/melody/Empirical_Engine/pickled/cb_2012/output/cb_2012_{}_ret_val.P'.format(im), 'wb') as f:
        pickle.dump(all, f)