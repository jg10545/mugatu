#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:52:46 2021

@author: joe
"""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse

from mugatu._cluster import reduce_and_cluster, compute_clusters, reduce_and_cluster_optics
from mugatu._cluster import sparse_corr

def test_reduce_and_cluster_kmeans():
    N = 1000
    k = 10
    test_data = np.random.normal(0, 1, (N, 20))
    test_index = np.arange(N) + 13
    indices = reduce_and_cluster(test_data, test_index, 5, k, xmeans=False)
    # check data types of index
    assert isinstance(indices, list)
    assert isinstance(indices[0], np.ndarray)
    # check correct number of clusters
    assert len(indices) == k
    # check that all records accounted for
    assert np.sum([len(x) for x in indices]) == N
    


def test_reduce_and_cluster_xmeans():
    N = 1000
    test_data = np.random.normal(0, 1, (N, 20))
    test_index = np.arange(N) 
    indices = reduce_and_cluster(test_data, test_index, 5,  xmeans=True)
    # check data types of index
    assert isinstance(indices, list)
    assert isinstance(indices[0], np.ndarray)
    # check correct number of clusters
    # check that all records accounted for
    assert np.sum([len(x) for x in indices]) == N
    

def test_reduce_and_cluster_empty_bin():
    k = 10
    test_data = np.random.normal(0,1,size=(0,10))
    test_index = np.arange(0) 
    indices = reduce_and_cluster(test_data, test_index, 5, k)
    # check data types of index
    assert isinstance(indices, list)
    assert len(indices) == 0
    
"""
def test_reduce_and_cluster_way_too_few_in_bin():
    N = 9
    k = 10
    test_data = np.random.normal(0, 1, (N, 20))
    test_index = np.arange(N) + 13
    indices = reduce_and_cluster(test_data, test_index, 5, k,
                                 min_points_per_cluster=10)
    # check data types of index
    assert isinstance(indices, list)
    assert isinstance(indices[0], np.ndarray)
    assert len(indices) == 1
    assert (indices[0] == test_index).all()

    
    
def test_reduce_and_cluster_too_few_in_bin():
    N = 24
    k = 5
    test_data = np.random.normal(0, 1, (N, 20))
    test_index = np.arange(N) + 13
    indices = reduce_and_cluster(test_data, test_index, 5, k,
                                 min_points_per_cluster=5)
    # check data types of index
    assert isinstance(indices, list)
    assert isinstance(indices[0], np.ndarray)
    assert len(indices) == 4

    
def test_reduce_and_cluster_optics():
    N = 1000
    min_samples = 5
    test_data = np.random.normal(0, 1, (N, 20))
    test_index = np.arange(N) + 13
    indices = reduce_and_cluster_optics(test_data, test_index, 5, min_samples)
    # check data types of index
    assert isinstance(indices, list)
    assert isinstance(indices[0], np.ndarray)
    # check correct number of clusters
    assert len(indices) >= 1 #== k
    # check that all records accounted for
    assert np.sum([len(x) for x in indices]) == N
    
    
def test_reduce_and_cluster_optics_empty_bin():
    min_samples = 5
    test_data = np.random.normal(0,1,size=(0,10))
    test_index = np.arange(0) 
    indices = reduce_and_cluster_optics(test_data, test_index, 5, min_samples)
    # check data types of index
    assert isinstance(indices, list)
    assert len(indices) == 0
"""
      
    
def test_compute_clusters_kmeans():
    N = 1000
    k = 5
    df = pd.DataFrame({"x":np.random.normal(0,1,N), "y":np.random.normal(0,1,N)}, index=np.arange(N)+N)
    cover = [np.arange(0,int(0.5*N)), np.arange(int(0.4*N), int(N))]
    indices = compute_clusters(df.values, cover, svd_dim=False, k=k, xmeans=False)
    assert len(indices) == k*len(cover)
    
    
    
def test_compute_clusters_conflicting_kwargs():
    N = 1000
    df = pd.DataFrame({"x":np.random.normal(0,1,N), "y":np.random.normal(0,1,N)}, index=np.arange(N)+N)
    cover = [np.arange(N,int(1.5*N)), np.arange(int(1.4*N), int(1.9*N))]
    with pytest.raises(UnboundLocalError):
        indices = compute_clusters(df, cover, svd_dim=False,
                                   k=0, min_samples=None)
    
    
def test_sparse_corr():
    r = 100
    c = 1000
    a = scipy.sparse.rand(r, c, density=0.01, format='csr')
    
    coeffs1 = sparse_corr(a.T)
    coeffs2 = np.corrcoef(a.todense())
    assert coeffs1.shape == (r,r)
    assert np.allclose(coeffs1, coeffs2)
    