#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:52:23 2021

@author: joe
"""
import pandas as pd

def _find_start_and_delta(minval, maxval, num_intervals, f):
    """
    Compute the spacing between start values and length of each
    interval
    
    :minval: minimum value of lens
    :maxval: maximum value of lens
    :intervals: number of overlapping intervals
    :f: overlap fraction
    """
    start = (maxval-minval)/(num_intervals - 1 + 1/(1-f))
    delta = (maxval-minval)/(num_intervals + f - f*num_intervals)
    return start, delta


def _compute_1D_cover_indices(index, lens, num_intervals, f, balance=False):
    """
    Return the index set of the covering as a list of arrays; each array
    can be used to index the original dataframe
    """
    assert len(index) == len(lens)
    indices = []
    if balance:
        pass
        lens = lens.argsort()
        start, delta = _find_start_and_delta(lens.min(), lens.max(), num_intervals, f)
        for i in range(num_intervals):
            s = int(lens.min() + i*start)
            e = int(s+delta)
            indices.append(index[lens[s:e+1]])
    else:
        start, delta = _find_start_and_delta(lens.min(), lens.max(), num_intervals, f)
        for i in range(num_intervals):
            s = lens.min() + i*start
            e = s+delta
            indices.append(index[(lens >=s)&(lens < e)])
            
    return indices


def compute_cover_indices(index, lens1, lens2=None, num_intervals=5, f=0.1, balance=False):
    """
    Build a covering of a dataset as a list of arrays, where each array can be used to index
    the original dataframe to recover the subset corresponding to an element of the index set
    
    :index: (N,) array of data indices
    :lens1: (N,) array representing first lens
    :lens2: optional; (N,) array representing second lens
    :num_intervals: int; number of pieces to break the dataset into
    :f: float; overlap fraction between pieces
    :balance: bool; if False slice lens into equally-sized pieces; if True slice lens into
        equally-occupied pieces
    """
    indices_1D = _compute_1D_cover_indices(index, lens1, num_intervals, f, balance)
    if lens2 is None:
        return indices_1D
    indices_2D = []
    # rebuild second lens as a pandas series so that we can subset it using
    # the index
    lens2 = pd.Series(data=lens2, index=index)
    for i in indices_1D:
        indices_2D += _compute_1D_cover_indices(i, lens2[i].values, num_intervals, f, balance)
        
    return indices_2D