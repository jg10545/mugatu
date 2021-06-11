#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:44:27 2021

@author: joe
"""
import numpy as np

from mugatu._cover import _find_start_and_delta, _compute_1D_cover_indices
from mugatu._cover import compute_cover_indices

def test_find_start_and_delta_no_overlap():
    s, d = _find_start_and_delta(0, 100, 2, 0)
    assert s == 50
    assert d == 50
    
    
    
def test_compute_1D_cover_indices():
    N = 100
    num_intervals = 5
    f = 0.1
    i = np.arange(N)
    l = np.random.normal(0,1,N)
    indices = _compute_1D_cover_indices(i, l, num_intervals, f, False)
    # correct number of intervals
    assert len(indices) == num_intervals
    # indices stores as arrays
    assert isinstance(indices[0], np.ndarray)
    # overlap > 0
    assert np.sum([len(x) for x in indices]) > N
    
def test_compute_1D_cover_indices_balanced():
    N = 100
    num_intervals = 5
    f = 0.1
    i = np.arange(N)
    l = np.random.normal(0,1,N)
    indices = _compute_1D_cover_indices(i, l, num_intervals, f, True)
    # correct number of intervals
    assert len(indices) == num_intervals
    # indices stores as arrays
    assert isinstance(indices[0], np.ndarray)
    # overlap > 0
    assert np.sum([len(x) for x in indices]) > N
    # balance
    assert np.abs(len(indices[0])-len(indices[1])) <= 1
    
    
def test_compute_cover_indices():
    N = 100
    num_intervals = 5
    f = 0.1
    i = np.arange(N) + 500
    l = np.random.normal(0,1,N)
    l2 = np.random.normal(0,1,N)
    indices = compute_cover_indices(i, l, l2, num_intervals, f, False)
    # correct number of intervals
    assert len(indices) == num_intervals**2
    # indices stores as arrays
    assert isinstance(indices[0], np.ndarray)
    # overlap > 0
    assert np.sum([len(x) for x in indices]) > N
    
def test_compute_cover_indices_balanced():
    N = 100
    num_intervals = 5
    f = 0.1
    i = np.arange(N)
    l = np.random.normal(0,1,N)
    l2 = np.random.normal(0,1,N)
    indices = compute_cover_indices(i, l, l2, num_intervals, f, True)
    # correct number of intervals
    assert len(indices) == num_intervals**2
    # indices stores as arrays
    assert isinstance(indices[0], np.ndarray)
    # overlap > 0
    assert np.sum([len(x) for x in indices]) > N
    # balance
    assert np.abs(len(indices[0])-len(indices[1])) <= 1