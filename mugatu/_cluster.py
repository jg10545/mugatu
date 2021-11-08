#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:51:31 2021

@author: joe
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn.decomposition
import sklearn.cluster
import dask
import logging

from mugatu._xmeans import _compute_xmeans, _compute_kmeans, _pca_reduce

def reduce_and_cluster(X, index, pca_dim=4, k=5, min_points_per_cluster=1, 
                       xmeans=False, aic=False, sparse_data=None,
                       **kwargs):
    """
    Reduce the dimension of a dataset with PCA, cluster with
    k-means or x-means, and return a list of indices assigned to each cluster
    
    :X: (N,d) array of raw data
    :index: (N,) array of indices used to identify data in original dataframe
    :pca_dim: int; dimension < d to reduce data to. 0 to disable.
    :k: number of clusters to look for
    :min_points_per_cluster: int; guardrail to avoid finding clusters below
        this size
    :xmeans: if True, run x-means clustering with k initial clusters
    :aic: if True and xmeans == True, use AIC instead of BIC with x-means
    :kwargs: keyword arguments to pass to faiss.Kmeans()
    """
    N = X.shape[0]
    d = X.shape[1]
    #if (sparse_data is not None)&(N > 0):
    #    densified = sklearn.decomposition.TruncatedSVD(pca_dim).fit_transform(sparse_data)
    #    X = np.concatenate([X, densified], 1)
    
    # in case the lens generates an empty segment
    if N == 0:
        return []
    # check to see if we have more than zero points, but fewer than
    # min_points_per_cluster- return a single cluster
    elif N < min_points_per_cluster*k:
        return [index]
    # similarly make adjust k if we don't have enough data
    elif N < k*min_points_per_cluster:
        k = N//min_points_per_cluster
    
    # make sure data is on a common scale and float-32 (to work with FAISS)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    # if using PCA, reduce dimension
    if pca_dim:
        X = _pca_reduce(X, pca_dim)
    # if there's a sparse matrix, reduce that with SVD and concatenate
    if sparse_data is not None:
        densified = sklearn.decomposition.TruncatedSVD(pca_dim).fit_transform(sparse_data)
        X = np.concatenate([X, densified], 1).astype(np.float32)
    """

=======
        # only reduce dimension if we have enough data points
        if N > pca_dim:
            # only apply PCA to dense data matrix if pca_dim is lower than data dim
            if pca_dim < d:
                mat = faiss.PCAMatrix(X.shape[1], pca_dim)
                mat.train(X)
                X = mat.apply_py(X)
            # if there's a sparse matrix, reduce that with SVD and concatenate
            if sparse_data is not None:
                densified = sklearn.decomposition.TruncatedSVD(pca_dim).fit_transform(sparse_data)
                X = np.concatenate([X, densified], 1).astype(np.float32)
>>>>>>> 4f666ccde29226dc888667e537e479a5a20dfab8
    """
    # Cluster and get indices. 
    if xmeans:
        I = _compute_xmeans(X, aic=aic, init_k=k, 
                            min_size=min_points_per_cluster, **kwargs)
    else:
        I, _ = _compute_kmeans(X, k, **kwargs)
    indices = [index[I == i] for i in range(I.max()+1)]
    # filter out empty clusters
    indices = [i for i in indices if len(i) > 0]
    return indices

def reduce_and_cluster_optics(X, index, pca_dim=4, min_samples=5, sparse_data=None):
    """
    Reduce the dimension of a dataset with PCA, cluster with
    OPTICS, and return a list of indices assigned to each cluster
    
    :X: (N,d) array of raw data
    :index: (N,) array of indices used to identify data in original dataframe
    :pca_dim: int; dimension < d to reduce data to. 0 to disable.
    :min_samples: min_samples parameter for OPTICS
    """
    N = X.shape[0]
    if (sparse_data is not None)&(N > 0):
        densified = sklearn.decomposition.TruncatedSVD(pca_dim).fit_transform(sparse_data)
        X = np.concatenate([X, densified], 1)
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
        X = _pca_reduce(X, pca_dim)
            
    # Cluster and get indices. 
    I = sklearn.cluster.OPTICS(min_samples=min_samples).fit_predict(X)
    k = I.max() + 1
    if I.min() < 0:
        logging.debug("including an outlier cluster")
    
    indices = [index[I == i] for i in range(-1, k)]
    # filter out empty clusters
    indices = [i for i in indices if len(i) > 0]
    # add noise points
    return indices



def compute_clusters(df, cover, pca_dim=4, min_samples=5, k=None, 
                     xmeans=False, aic=False, sparse_data=None, **kwargs):
    """
    Input a dataset and cover, run k-means or OPTICS against every index set in the
    cover, and return a list containing the indices assigned to each cluster
    
    :df: pandas DataFrame containing the data
    :cover: a list of arrays indexing the DataFrame, each corresponding to a different
        index set from the cover
    :pca_dim: number of dimensions to reduce to with PCA before clustering. 0 to skip this step
    :min_samples: int; guardrail to try to avoid returning clusters below this size
    :k: number of clusters for k-means and x-means. set to 0 to use OPTICS
    :xmeans: if True and k>0, use x-means clustering with k as the initial number
        of clustering
    :aic: if True and xmeans==True, use AIC instead of BIC for x-means clustering
    :kwargs: additional keyword arguments to pass to faiss.Kmeans()
    """
    if sparse_data is not None:
        N = len(df)
        sdf = pd.Series(data=np.arange(N), index=df.index.values)
        sparse = [sparse_data[sdf[c].values] for c in cover]
    else:
        sparse = [None for c in cover]
    
    # build a dask delayed task for every filtered region of the data, 
    # so that they can be computed in parallel
    # OPTICS CASE
    if (min_samples is not None)&((k is None)|(k == 0)):
        logging.debug("clustering with OPTICS")
        tasks = [dask.delayed(reduce_and_cluster_optics)(np.ascontiguousarray(df.loc[c,:].values), 
                                              c, pca_dim=pca_dim, min_samples=min_samples,
                                              sparse_data=s)
     for c,s in zip(cover, sparse)]
    # K-MEANS CASE
    elif (k is not None)&(k > 0):
        logging.debug("clustering with k-means")
        tasks = [dask.delayed(reduce_and_cluster)(np.ascontiguousarray(df.loc[c,:].values), 
                                              c, pca_dim=pca_dim, k=k,
                                              min_points_per_cluster=min_samples,
                                              xmeans=xmeans, aic=aic, 
                                              sparse_data=s, **kwargs) 
                 for c,s in zip(cover, sparse)]
    else:
        logging.critical("i don't know what to do with these clustering hyperparameters")
    results = dask.compute(tasks)
    
    output_indices = []
    for r in results:
        for i in r:
            output_indices += i
    return output_indices