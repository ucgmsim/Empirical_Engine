#!/usr/bin/env python

import os
from subprocess import call

matlab_tests = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_matlab")

def test_ASK_2014_nga():
    call([os.path.join(matlab_tests, "ASK_2014_nga_verify.m")])
    call([os.path.join(matlab_tests, "ASK_2014_nga_verify.py")])
