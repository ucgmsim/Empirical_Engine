import os
import pandas as pd
import pickle

from empirical.test.test_common_setup import set_up
from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM
from empirical.util.empirical_factory import compute_gmm

IM = 'pSA'
TECT_TYPES = {'TectType.SUBDUCTION_SLAB':TectType.SUBDUCTION_SLAB, 'TectType.SUBDUCTION_INTERFACE':TectType.SUBDUCTION_INTERFACE,'TectType.ACTIVE_SHALLOW': TectType.ACTIVE_SHALLOW}
SITE_CLASSES = {'SiteClass.SOFTSOIL':SiteClass.SOFTSOIL, 'SiteClass.MEDIUMSOIL': SiteClass.MEDIUMSOIL, 'SiteClass.HARDSOIL':SiteClass.HARDSOIL,'SiteClass.ROCK':SiteClass.ROCK, 'SiteClass.HARDROCK':SiteClass.HARDROCK}

SITE = Site()

FAULT = Fault()
FAULT.Mw = 7.2
FAULT.faultstyle = 'interface'
FAULT.hdepth = 0
FAULT.ztor = 0
FAULT.rake = 0
FAULT.dip = 0


def test_zhao_2006(set_up):
    with open(os.path.join(set_up, 'output', 'zhao_2006_ret_val.P'), 'rb') as f:
        expected_results = pickle.load(f)

    df = pd.read_excel(os.path.join(set_up, 'input', 'Zhao_Test_Cases_2.xlsx'))
    all_results = []
    for index, row in df.iterrows():
        # hdepth change at row 194 (row 199 in xlsx file
        if row.name == 194:
            FAULT.hdepth = 10
        # if not an empty row
        if not row.isnull().any():
            FAULT.tect_type = TECT_TYPES[row['tect_type']]
            SITE.Rrup = float(row['rrup'])
            SITE.siteclass = SITE_CLASSES[row['siteclass']]
            results = compute_gmm(FAULT, SITE, GMM.ZA_06, IM, [row['period']])
            all_results.append(results)
    assert all_results == expected_results
