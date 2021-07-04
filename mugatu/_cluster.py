#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:51:31 2021

@author: joe
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import sklearn.cluster
import faiss
import dask
import logging

def reduce_and_cluster(X, index, pca_dim=4, k=5, min_points_per_cluster=1, **kwargs):
    """
    Reduce the dimension of a dataset with PCA, cluster with
    k-means, and return a list of indices assigned to each cluster
    
    :X: (N,d) array of raw data
    :index: (N,) array of indices used to identify data in original dataframe
    :pca_dim: int; dimension < d to reduce data to. 0 to disable.
    :k: number of clusters to look for
    :min_points_per_cluster:
    :kwargs: keyword arguments to pass to faiss.Kmeans()
    """
    N = X.shape[0]
    # in case the lens generates an empty segment
    if N == 0:
        return []
    # check to see if we have more than zero points, but fewer than
    # min_points_per_cluster- return a single cluster
    elif N < min_points_per_cluster:
        return [index]
    # similarly make adjust k if we don't have enough data
    elif N < k*min_points_per_cluster:
        k = N//min_points_per_cluster
    
    # make sure data is on a common scale and float-32 (to work with FAISS)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    # if using PCA, reduce dimension
    if pca_dim:
        # only reduce dimension if we have enough data points
        if N > pca_dim:
            mat = faiss.PCAMatrix(X.shape[1], pca_dim)
            mat.train(X)
            X = mat.apply_py(X)
            
    # Cluster and get indices. 
    kmeans = faiss.Kmeans(X.shape[1] ,k, **kwargs)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.ravel()
    indices = [index[I == i] for i in range(k)]
    # filter out empty clusters
    indices = [i for i in indices if len(i) > 0]
    return indices

def reduce_and_cluster_optics(X, index, pca_dim=4, min_samples=5):
    """
    Reduce the dimension of a dataset with PCA, cluster with
    OPTICS, and return a list of indices assigned to each cluster
    
    :X: (N,d) array of raw data
    :index: (N,) array of indices used to identify data in original dataframe
    :pca_dim: int; dimension < d to reduce data to. 0 to disable.
    :min_samples: min_samples parameter for OPTICS
    """
    N = X.shape[0]
    # in case the lens generates an empty segment
    if N == 0:
        return []
    # check to see if we have more than zero points, but fewer than
    # min_points_per_cluster- return a single cluster
    elif N < min_samples:
        return [index]
    
    # make sure data is on a common scale and float-32 (to work with FAISS)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    # if using PCA, reduce dimension
    if pca_dim:
        # only reduce dimension if we have enough data points
        if N > pca_dim:
            mat = faiss.PCAMatrix(X.shape[1], pca_dim)
            mat.train(X)
            X = mat.apply_py(X)
            
    # Cluster and get indices. 
    I = sklearn.cluster.OPTICS(min_samples=min_samples).fit_predict(X)
    k = I.max() + 1
    if I.min() < 0:
        logging.info("including an outlier cluster")
    
    #indices = [index[I == i] for i in range(k)]
    indices = [index[I == i] for i in range(-1, k)]
    # filter out empty clusters
    indices = [i for i in indices if len(i) > 0]
    # add noise points
    return indices



def compute_clusters(df, cover, pca_dim=4, min_samples=None, k=None, **kwargs):
    """
    Input a dataset and cover, run k-means or OPTICS against every index set in the
    cover, and return a list containing the indices assigned to each cluster
    
    :df: pandas DataFrame containing the data
    :cover: a list of arrays indexing the DataFrame, each corresponding to a different
        index set from the cover
    :pca_dim: number of dimensions to reduce to with PCA before clustering. 0 to skip this step
    :k: number of clusters for k-means
    :kwargs: additional keyword arguments to pass to faiss.Kmeans()
    """
    # build a dask delayed task for every filtered region of the data, 
    # so that they can be computed in parallel
    # K-MEANS CASE
    if (min_samples is None)&(k is not None):
        tasks = [dask.delayed(reduce_and_cluster)(np.ascontiguousarray(df.loc[c,:].values), 
                                              c, pca_dim=pca_dim, k=k, **kwargs) for c in cover]
    # OPTICS CASE
    elif (min_samples is not None)&(k is None):
        tasks = [dask.delayed(reduce_and_cluster_optics)(np.ascontiguousarray(df.loc[c,:].values), 
                                              c, pca_dim=pca_dim, min_samples=min_samples) for c in cover]
    else:
        logging.critical("i don't know what to do with these clustering hyperparameters")
    results = dask.compute(tasks)
    
    output_indices = []
    for r in results:
        for i in r:
            output_indices += i
    return output_indices