import os
import pandas as pd
import pickle

from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM
from empirical.util.empirical_factory import compute_gmm

TECT_TYPES = {'TectType.SUBDUCTION_SLAB':TectType.SUBDUCTION_SLAB, 'TectType.SUBDUCTION_INTERFACE':TectType.SUBDUCTION_INTERFACE,'TectType.ACTIVE_SHALLOW':TectType.ACTIVE_SHALLOW}
SITE_CLASSES = {'SiteClass.SOFTSOIL':SiteClass.SOFTSOIL, 'SiteClass.MEDIUMSOIL':SiteClass.MEDIUMSOIL, 'SiteClass.HARDSOIL':SiteClass.HARDSOIL,'SiteClass.ROCK':SiteClass.ROCK, 'SiteClass.HARDROCK':SiteClass.HARDROCK}

SITE = Site()

FAULT = Fault()
FAULT.Mw = 7.2
FAULT.faultstyle = 'interface'
FAULT.hdepth = 0
FAULT.ztor = 0
FAULT.rake = 0
FAULT.dip = 0


TEST_DATA_SAVE_DIR = '/home/melody/Empirical_Engine/pickled/zhao_2006/'
INPUT = "input"
OUTPUT = "output"
BENCHMARK = 'Zhao_Test_Cases_2.xlsx'
TEST_INPUT = os.path.join(TEST_DATA_SAVE_DIR, INPUT, BENCHMARK)
TEST_OUTPUT = os.path.join(TEST_DATA_SAVE_DIR, OUTPUT)

IM = 'pSA'

with open(os.path.join(TEST_OUTPUT, 'zhao_2006_ret_val.P'), 'rb') as f:
    EXPECTED_RESULTS = pickle.load(f)


def test_zhao_2006():
    df = pd.read_excel(TEST_INPUT)
    all_results = []
    for index, row in df.iterrows():
        if row.name == 194:
            FAULT.hdepth = 10
        if not row.isnull().any():
            FAULT.tect_type = TECT_TYPES[row['tect_type']]
            SITE.Rrup = float(row['rrup'])
            SITE.siteclass = SITE_CLASSES[row['siteclass']]
            results = compute_gmm(FAULT, SITE, GMM.ZA_06, IM, [row['period']])
            all_results.append(results)
    assert all_results == EXPECTED_RESULTS
