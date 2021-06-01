#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:52:46 2021

@author: joe
"""
import numpy as np
import pandas as pd

from mugatu._cluster import reduce_and_cluster, compute_clusters

def test_reduce_and_cluster():
    N = 1000
    k = 10
    test_data = np.random.normal(0, 1, (N, 20))
    test_index = np.arange(N) + 13
    indices = reduce_and_cluster(test_data, test_index, 5, k)
    # check data types of index
    assert isinstance(indices, list)
    assert isinstance(indices[0], np.ndarray)
    # check correct number of clusters
    assert len(indices) == k
    # check that all records accounted for
    assert np.sum([len(x) for x in indices]) == N
    
    
    
def test_compute_clusters():
    N = 1000
    k = 5
    df = pd.DataFrame({"x":np.random.normal(0,1,N), "y":np.random.normal(0,1,N)}, index=np.arange(N)+N)
    cover = [np.arange(N,int(1.5*N)), np.arange(int(1.4*N), int(1.9*N))]
    indices = compute_clusters(df, cover, False, k)
    assert len(indices) == k*len(cover)
    
    