#!/usr/bin/env python

import os
from subprocess import call

matlab_tests = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_matlab")

def test_Mcverryetal_2006_SAgm():
    #call([os.path.join(matlab_tests, "Mcverryetal_2006_SAgm_verify.m")], cwd=matlab_tests)
    call([os.path.join(matlab_tests, "Mcverryetal_2006_SAgm_verify.py")], cwd=matlab_tests)
