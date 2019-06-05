from empirical.GMM_models import Bradley_2013_Sa
from empirical.util.classdef import Site, Fault, TectType, SiteClass, GMM
import pytest

def test_Bradley_2013_Sa(siteprop, faultprop, im, periods):