from empirical.util import empirical_factory
from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM
import pytest

FAULT1 = Fault()
FAULT1.Mw = 10.5
FAULT1.faultstyle = 'interface'
FAULT1.ztor = 0
FAULT1.Ztor = 0
FAULT1.rake = 180
FAULT1.dip = 45

SITE1 = Site()
#site.Rrup = 10
SITE1.Rjb = 10
SITE1.vs30 = 500
SITE1.V30 = 500
SITE1.V30measured = None
SITE1.Rx = -1
SITE1.Rtvz = 50

PERIOD = [0, 0.01, 0.40370172586, 0.5, 3.0, 8.6974900]


def compute_gmm(test_rrup, test_fault, test_site, test_im, expected_value):


