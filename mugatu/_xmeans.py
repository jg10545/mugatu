#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 13:41:36 2021

@author: joe
"""
import numpy as np 
import logging

import sklearn.cluster, sklearn.decomposition
    
def _compute_kmeans(X,k, **kwargs):
    clus = sklearn.cluster.KMeans(k)
    clus = clus.fit(X)
    return clus.predict(X), clus.cluster_centers_
    
def _pca_reduce(X, pca_dim):
    N, d = X.shape
    # only reduce dimension if we have enough data of
    # high enough dimension
    if (d <= pca_dim)|(N <= pca_dim):
        return X
    else:
        return sklearn.decomposition.PCA(pca_dim).fit_transform(X)

def _svd_reduce(X, svd_dim):
    if svd_dim == 0:
        return X
    N, d = X.shape
    # only reduce dimension if we have enough data of
    # high enough dimension
    if (d <= svd_dim)|(N <= svd_dim):
        return X
    else:
        return sklearn.decomposition.TruncatedSVD(svd_dim).fit_transform(X)



def _compute_BIC(X, centroids, I, aic=False):
    """
    Compute Bayesian or Akaike Information Criterion for a k-means model.
    
    Using formalism where larger AIC/BIC is preferable to smaller
    
    :X: (N,d) numpy array of data
    :centroids: (k,d) numpy array of cluster centroids
    :I: (N,) numpy array mapping data points to the nearest cluster
    :aic: if True, compute AIC; if False, compute BIC
    """
    N, d = X.shape
    #k = centroids.shape[0]
    # We might include centroids that aren't represented in this subset of 
    # vectors- so compute k based on just the number represented here.
    k = len(set(I))
    eps = 1e-8
    # DENSE ARRAY CASE
    if isinstance(X, np.ndarray):
        variance = np.sum((X-centroids[I,:])**2)/(d*(N-k))
    # SPARSE MATRIX CASE
    else:
        variance = 0
        for i in range(N):
            variance += np.sum((np.array(X[i,:].todense())-centroids[I[i],:])**2)/(d*(N-k))
            
    if (N < 0)|(variance < 0):
        print (N, variance, k)
        assert False
    
    BIC = 0
    for i in set(I):
        Nk = np.sum(I == i)
        if Nk > 0:
            BIC += Nk * np.log(Nk)
            
    if aic:
        BIC += -1*N*np.log(N) - (N*d/2)*np.log(2*np.pi*variance+eps) - d*(N-k)/2 - k*(d+1)
        BIC *= 2
    else:
        BIC += -1*N*np.log(N) - (N*d/2)*np.log(2*np.pi*variance+eps) - d*(N-k)/2 - np.log(N)*k*(d+1)/2
    
    return BIC


def _compute_xmeans(X, aic=False, min_size=0, max_depth=8, **kwargs):
    """
    Compute cluster assignments using X-means clustering
    
    :X: (N,d) float32 numpy array containing data
    :aic: if True, use AIC instead of BIC
    :min_size: Guardrail to keep x-means from finding a bunch of teensy
        clusters- set this to a positive value to stop splitting whenever
        a cluster has fewer than this number of records
    :max_depth: another guardrail- max number of passes over the clusters to
        check for splitting
    """
    # set of indices that we already know don't improve with splitting
    stop_checking = set()
    # initialize with one big cluster
    I = np.zeros(X.shape[0], dtype=np.int64)
    if isinstance(X, np.ndarray):
        centroids = X.mean(0, keepdims=True)
    else:
        centroids = np.array(X.mean(0))

    # iterate over clusters, splitting if the BIC improves, up to
    # max_depth times
    for m in range(max_depth):
        logging.debug(f"xmeans depth: {m}")
    
        keep_going = False
        I_old = I.copy()
        # for each cluster
        for i in set(I_old)-stop_checking:

            X_subset = X[I_old==i,:]
            I_subset = I_old[I_old==i]
            initial_BIC = _compute_BIC(X_subset, centroids, I_subset, aic=aic)

            I_next, centroids_next = _compute_kmeans(X_subset,2, **kwargs)
            # how big is the smallest new cluster?
            smallest_cluster = np.min([(I_next == j).sum() for j in range(2)])
            # if the smallest cluster doesn't disqualify this split, 
            # compute BIC for split clusters
            if smallest_cluster >= min_size:
                split_BIC = _compute_BIC(X_subset, centroids_next, I_next, aic=aic)
            else:
                split_BIC = 1e6

            # keep split if BIC went up AND none of the clusters are too small
            if (initial_BIC < split_BIC)&(smallest_cluster >= min_size):
                logging.debug(f"smallest cluster: {smallest_cluster} compare {min_size}")
                logging.debug(f"xmeans splitting cluster{i}")
                keep_going = True
                I[I_old==i] += I_next*(I.max()+1-i)
                centroids[i,:] = centroids_next[0,:]
                centroids = np.concatenate([centroids, centroids_next[1,:].reshape(1,-1)],0)
            else:
                # if splitting didn't improve the BIC- add this cluster to stop_checking
                # so that we don't waste time recomputing
                stop_checking.add(i)
        if not keep_going:
            logging.debug(f"no clusters split; interrupting xmeans at depth {m}")
            break
    num_clusters = I.max()+1
    logging.debug(f"xmeans completed with {num_clusters} clusters found")
    return I