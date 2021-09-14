from pathlib import Path
import itertools

import pandas as pd
import numpy as np

from empirical.test.test_common_setup import set_up
from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM
from empirical.util.empirical_factory import compute_gmm

IM = "pSA"
TECT_TYPES = [
    TectType.SUBDUCTION_SLAB,
    TectType.SUBDUCTION_INTERFACE,
    TectType.ACTIVE_SHALLOW,
]
SITE_CLASSES = [
    SiteClass.SOFTSOIL,
    SiteClass.MEDIUMSOIL,
    SiteClass.HARDSOIL,
    SiteClass.ROCK,
    SiteClass.HARDROCK,
]


SITE = Site()

FAULT = Fault()
FAULT.Mw = 7.2
FAULT.faultstyle = "interface"
FAULT.hdepth = 0
FAULT.ztor = 0
FAULT.rake = 0
FAULT.dip = 0

periods = [0, 0.01, 0.5, 3]
rrups = [10, 70, 200]
hdepths = [0, 10]


def test_zhao_2006(set_up):
    set_up = Path(set_up)
    expected_results = pd.read_csv(set_up / "output" / "zhao_output_21p8.csv")

    all_results = {}
    for i, (hdepth, tect_type, site_class, rrup, period) in enumerate(
        itertools.product(hdepths, TECT_TYPES, SITE_CLASSES, rrups, periods)
    ):
        FAULT.hdepth = hdepth
        FAULT.tect_type = tect_type
        SITE.Rrup = rrup
        SITE.siteclass = site_class
        mean, (std_total, std_inter, std_intra) = compute_gmm(
            FAULT, SITE, GMM.ZA_06, IM, period
        )
        all_results[i] = {
            "period": period,
            "rrup": rrup,
            "siteclass": site_class,
            "tect_type": tect_type,
            "hdepth": hdepth,
            "mean": mean,
            "std_total": std_total,
            "std_inter": std_inter,
            "std_intra": std_intra,
        }
    results_df = pd.DataFrame.from_dict(all_results, orient="index")
    assert np.all(np.isclose(expected_results["mean"], results_df["mean"]))
    assert np.all(np.isclose(expected_results["std_total"], results_df["std_total"]))
    assert np.all(np.isclose(expected_results["std_inter"], results_df["std_inter"]))
    assert np.all(np.isclose(expected_results["std_intra"], results_df["std_intra"]))
