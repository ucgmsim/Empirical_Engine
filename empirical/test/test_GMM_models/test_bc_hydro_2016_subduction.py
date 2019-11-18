#!/usr/bin/env python

import os
from subprocess import call

matlab_tests = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_matlab")

def test_bc_hydro_2016_subduction():
    call([os.path.join(matlab_tests, "bc_hydro_2016_subduction_verify.m")])
    call([os.path.join(matlab_tests, "bc_hydro_2016_subduction_verify.py")])
