#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 10:17:04 2021

@author: joe
"""
import numpy as np
import pandas as pd

from mugatu._util import _lens_dict

def test_lens_dict():
    N = 100
    data = pd.DataFrame({x:np.random.normal(0,1,N) for x in ["x","y","z"]})
    lens_dict = _lens_dict(data)
    assert len(lens_dict) == 4
    for l in lens_dict:
        assert isinstance(lens_dict[l], np.ndarray)
        assert lens_dict[l].shape == (N,)
    
    
def test_lens_dict_with_extra_lens_data():
    N = 100
    data = pd.DataFrame({x:np.random.normal(0,1,N) for x in ["x","y","z"]})
    lens_data = pd.DataFrame({x:np.random.normal(0,1,N) for x in ["lens1", "lens2"]})
    lens_dict = _lens_dict(data, lens_data=lens_data)
    assert len(lens_dict) == 6
    for l in lens_dict:
        assert isinstance(lens_dict[l], np.ndarray)
        assert lens_dict[l].shape == (N,)