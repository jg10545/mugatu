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
import scipy.cluster, scipy.sparse, scipy.spatial.distance

from mugatu._xmeans import _compute_xmeans, _compute_kmeans,  _svd_reduce

<<<<<<< HEAD
def reduce_and_cluster(X, index, pca_dim=4, k=5, min_points_per_cluster=1,
                       xmeans=False, aic=False, sparse_data=None,
                       **kwargs):
    """
    Reduce the dimension of a dataset with PCA, cluster with
    k-means or x-means, and return a list of indices assigned to each cluster

=======
def reduce_and_cluster(X, index, svd_dim=0, k=2, min_points_per_cluster=1, 
                       xmeans=True, aic=False, **kwargs):
    """
    Reduce the dimension of a dataset with SVD, cluster with
    x-means, and return a list of indices assigned to each cluster
    
>>>>>>> revisiting
    :X: (N,d) array of raw data
    :index: (N,) array of indices used to identify data in original dataframe
    :svd_dim: int; dimension < d to reduce data to. 0 to disable.
    :k: number of clusters to look for
    :min_points_per_cluster: int; guardrail to avoid finding clusters below
        this size
    :xmeans: if True, run x-means clustering with k initial clusters
    :aic: if True and xmeans == True, use AIC instead of BIC with x-means
    :kwargs: keyword arguments to pass to faiss.Kmeans()
    """
    N = X.shape[0]
<<<<<<< HEAD
    if (sparse_data is not None)&(N > 0):
        densified = sklearn.decomposition.TruncatedSVD(pca_dim).fit_transform(sparse_data)
        X = np.concatenate([X, densified], 1)

=======
    d = X.shape[1]
    
>>>>>>> revisiting
    # in case the lens generates an empty segment
    if N == 0:
        return []
    # check to see if we have more than zero points, but fewer than
    # min_points_per_cluster- return a single cluster 
    if xmeans:
        threshold = 2*min_points_per_cluster
    else:
        threshold = k*min_points_per_cluster
    if N < threshold:
        return [index]
    # similarly make adjust k if we don't have enough data
<<<<<<< HEAD
    elif N < k*min_points_per_cluster:
        k = N//min_points_per_cluster

    # make sure data is on a common scale and float-32 (to work with FAISS)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    # if using PCA, reduce dimension
    if pca_dim:
        X = _pca_reduce(X, pca_dim)

    # Cluster and get indices.
=======
    #elif N < k*min_points_per_cluster:
    #    k = N//min_points_per_cluster
    
    # if using SVD, reduce dimension
    X = _svd_reduce(X, svd_dim)
    # Cluster and get indices. 
    # X-MEANS CASE
>>>>>>> revisiting
    if xmeans:
        I = _compute_xmeans(X, aic=aic, init_k=k,
                            min_size=min_points_per_cluster, **kwargs)
    # K-MEANS CASE
    else:
        if N < k*min_points_per_cluster:
            k = N//min_points_per_cluster
        I, _ = _compute_kmeans(X, k, **kwargs)
    indices = [index[I == i] for i in range(I.max()+1)]
    # filter out empty clusters
    indices = [i for i in indices if len(i) > 0]
    return indices

def reduce_and_cluster_optics(X, index, svd_dim=4, min_samples=5, sparse_data=None):
    """
    Reduce the dimension of a dataset with svd, cluster with
    OPTICS, and return a list of indices assigned to each cluster

    :X: (N,d) array of raw data
    :index: (N,) array of indices used to identify data in original dataframe
    :pca_dim: int; dimension < d to reduce data to. 0 to disable.
    :min_samples: min_samples parameter for OPTICS
    """
    N = X.shape[0]
    if (sparse_data is not None)&(N > 0):
        densified = sklearn.decomposition.TruncatedSVD(svd_dim).fit_transform(sparse_data)
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
<<<<<<< HEAD
    # if using PCA, reduce dimension
    if pca_dim:
        X = _pca_reduce(X, pca_dim)

    # Cluster and get indices.
=======
    # if using SVD, reduce dimension
    if svd_dim:
        X = _svd_reduce(X, svd_dim)
            
    # Cluster and get indices. 
>>>>>>> revisiting
    I = sklearn.cluster.OPTICS(min_samples=min_samples).fit_predict(X)
    k = I.max() + 1
    if I.min() < 0:
        logging.debug("including an outlier cluster")

    indices = [index[I == i] for i in range(-1, k)]
    # filter out empty clusters
    indices = [i for i in indices if len(i) > 0]
    # add noise points
    return indices



<<<<<<< HEAD
def compute_clusters(df, cover, pca_dim=4, min_samples=5, k=None,
                     xmeans=False, aic=False, sparse_data=None, **kwargs):
=======



def sparse_corr(A):
    N = A.shape[0]
    C=((A.T*A -(sum(A).T*sum(A)/N))/(N-1)).todense()
    V=np.sqrt(np.mat(np.diag(C)).T*np.mat(np.diag(C)))
    COR = np.divide(C,V+1e-119)
    return np.array(COR)



def cluster_hierarchical(X, index, dist_thresh=1.):
    """
    Compute pearson distances for sparse matrix X and run single-linkage
    hierarchical clustering.
    
    :X: (N,d) scipy csr matrix of raw data
    :index: (N,) array of indices used to identify data in original dataframe
    :dist_thresh: pearson distance threshold for clustering cutoff
    """
    print("starting clustering")
    print(X.shape[0])
    if X.shape[0] == 0:
        print("why is this empty??")
        return []
    elif X.shape[0] == 1:
        return [index]
    # compute pearson distances
    if isinstance(X, scipy.sparse.csr.csr_matrix):
        print("sparse version")
        pearson_dists = 1-sparse_corr(X.T)
    else:
        print("dense version")
        pearson_dists = 1-np.corrcoef(X)
    # convert redundant distance matrix to reduced form
    try:
        pearson_dists = scipy.spatial.distance.squareform(pearson_dists)
    except:
        print([pearson_dists[i,i] for i in range(pearson_dists.shape[0])])
        assert False
        
    #dist_thresh = np.median(pearson_dists)
    quantile = 0.01
    N = X.shape[0]
    M = N**2
    # np.quantile returns the value of the first quantile*M values. but the
    # N lowest values should be zero (diagonal of the distance matrix) so
    # we really want the (quantile*(M-N) + N)th value.
    adjusted_quantile = (quantile*(M-N) + N)/M
    print("adjusted quantile:", adjusted_quantile)
    dist_thresh = np.quantile(pearson_dists, adjusted_quantile)
    #rms = np.sqrt(X.multiply(X).mean() - X.mean()**2)
    #distance_threshold = rms
    #distance_threshold = pearson_dists.sum()/(M-N)
    print("distance threshold:", dist_thresh)
    print("minimum distance:", pearson_dists.min())
    # cluster
    linkage = scipy.cluster.hierarchy.single(pearson_dists)
    clusters = scipy.cluster.hierarchy.fcluster(linkage, dist_thresh, criterion="distance")
    # find indices associated with each cluster
    cluster_indices = [index[clusters==c] for c in set(clusters)]
    for c in cluster_indices:
        if len(c) == 0:
            print(clusters)
            assert False
    print("%s clusters out of %s records"%(len(cluster_indices), X.shape[0]))
    return cluster_indices



def compute_clusters(X, cover, svd_dim=0, min_samples=5, k=2, 
                     xmeans=True, aic=False, **kwargs):
>>>>>>> revisiting
    """
    Input a dataset and cover, run k-means or OPTICS against every index set in the
    cover, and return a list containing the indices assigned to each cluster

    :df: pandas DataFrame containing the data
    :cover: a list of arrays indexing the DataFrame, each corresponding to a different
        index set from the cover
    :svd_dim: number of dimensions to reduce to with SVD before clustering. 0 to skip this step
    :min_samples: int; guardrail to try to avoid returning clusters below this size
    :k: number of clusters for k-means and x-means. 
    :xmeans: if True and k>0, use x-means clustering with k as the initial number
        of clustering
    :aic: if True and xmeans==True, use AIC instead of BIC for x-means clustering
    :kwargs: additional keyword arguments to pass to faiss.Kmeans()
    """
<<<<<<< HEAD
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
=======
    # build a dask delayed task for every filtered region of the data, 
    # so that they can be computed in parallel
    logging.debug("building tasks")
    # dense version- using numpy arrays
    if isinstance(X, np.ndarray):
        tasks = [dask.delayed(reduce_and_cluster)(np.ascontiguousarray(X[c,:]), 
                                              c, svd_dim=svd_dim, k=k,
                                              min_points_per_cluster=min_samples,
                                              xmeans=xmeans, aic=aic, 
                                               **kwargs)
                 for c in cover]
    # sparse version- scipy CSR matrix
    elif isinstance(X, scipy.sparse.csr_matrix):
        tasks = [dask.delayed(reduce_and_cluster)(X[c,:], 
                                              c, svd_dim=svd_dim, k=k,
                                              min_points_per_cluster=min_samples,
                                              xmeans=xmeans, aic=aic, 
                                               **kwargs) 
                 for c in cover]
    else:
        logging.critical("i don't know what to do with this data matrix")
    logging.debug("computing tasks")
    results = dask.compute(*tasks)
    logging.debug("done with tasks")
    output_indices = []
    for r in results:
        output_indices += r
    return output_indices
>>>>>>> revisiting
