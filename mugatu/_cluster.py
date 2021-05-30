#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:51:31 2021

@author: joe
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
import faiss
import dask

def reduce_and_cluster(X, index, pca_dim=4, k=5, **kwargs):
    """
    Reduce the dimension of a dataset with PCA, cluster with
    k-means, and return a list of indices assigned to each cluster
    
    :X: (N,d) array of raw data
    :index: (N,) array of indices used to identify data in original dataframe
    :pca_dim: int; dimension < d to reduce data to. 0 to disable.
    :k: number of clusters to look for
    :kwargs: keyword arguments to pass to faiss.Kmeans()
    """
    # make sure data is on a common scale and float-32 (to work with FAISS)
    X = StandardScaler().fit_transform(X).astype(np.float32)
    # if using PCA, reduce dimension
    if pca_dim:
        mat = faiss.PCAMatrix(X.shape[1], pca_dim)
        mat.train(X)
        X = mat.apply_py(X)
    # Cluster and get indices
    kmeans = faiss.Kmeans(X.shape[1] ,k, **kwargs)
    kmeans.train(X)
    D, I = kmeans.index.search(X, 1)
    I = I.ravel()
    indices = [index[I == i] for i in range(k)]
    return indices


def compute_clusters(df, cover, pca_dim=4, k=5, **kwargs):
    """
    Input a dataset and cover, run k-means against every index set in the
    cover, and return a list containing the indices assigned to each cluster
    
    :df: pandas DataFrame containing the data
    :cover: a list of arrays indexing the DataFrame, each corresponding to a different
        index set from the cover
    :pca_dim: number of dimensions to reduce to with PCA before clustering. 0 to skip this step
    :k: number of clusters for k-means
    :kwargs: additional keyword arguments to pass to faiss.Kmeans()
    """
    tasks = [dask.delayed(reduce_and_cluster)(np.ascontiguousarray(df.loc[c,:].values), 
                                              c, pca_dim=pca_dim, k=k, **kwargs) for c in cover]
    results = dask.compute(tasks)
    
    output_indices = []
    for r in results:
        for i in r:
            output_indices += i
    return output_indices