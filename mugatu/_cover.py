#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 11:52:23 2021

@author: joe
"""
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