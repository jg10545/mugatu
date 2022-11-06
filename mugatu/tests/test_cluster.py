"""
Created on Sun May 30 13:52:46 2021

@author: joe
"""
import numpy as np
import pandas as pd
import pytest

from mugatu._cluster import reduce_and_cluster, compute_clusters, reduce_and_cluster_optics

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
"""

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


def test_compute_clusters_kmeans():
    N = 1000
    k = 5
    df = pd.DataFrame({"x":np.random.normal(0,1,N), "y":np.random.normal(0,1,N)}, index=np.arange(N)+N)
    cover = [np.arange(N,int(1.5*N)), np.arange(int(1.4*N), int(1.9*N))]
    indices = compute_clusters(df, cover, pca_dim=False, k=k)
    assert len(indices) == k*len(cover)

def test_compute_clusters_optics():
    N = 1000
    min_samples = 5
    df = pd.DataFrame({"x":np.random.normal(0,1,N), "y":np.random.normal(0,1,N)}, index=np.arange(N)+N)
    cover = [np.arange(N,int(1.5*N)), np.arange(int(1.4*N), int(1.9*N))]
    indices = compute_clusters(df, cover, pca_dim=False, min_samples=min_samples)
    assert len(indices) >= 1


def test_compute_clusters_conflicting_kwargs():
    N = 1000
    df = pd.DataFrame({"x":np.random.normal(0,1,N), "y":np.random.normal(0,1,N)}, index=np.arange(N)+N)
    cover = [np.arange(N,int(1.5*N)), np.arange(int(1.4*N), int(1.9*N))]
    with pytest.raises(UnboundLocalError):
        indices = compute_clusters(df, cover, pca_dim=False,
                                   k=0, min_samples=None)
