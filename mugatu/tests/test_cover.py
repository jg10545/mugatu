#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:44:27 2021

@author: joe
"""
from mugatu._cover import _find_start_and_delta

def test_find_start_and_delta_no_overlap():
    s, d = _find_start_and_delta(0, 100, 2, 0)
    assert s == 50
    assert d == 50
    