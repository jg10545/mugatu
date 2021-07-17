#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:32:21 2021

@author: joe
"""
import mugatu._cover
import mugatu._cluster
import mugatu._graph


def build_mapper_graph(df, lens, lens2=None, num_intervals=5, f=0.1, balance=False,
                       pca_dim=4, k=None, min_samples=5, xmeans=False, aic=False, **kwargs):
    """
    Run the entire mapper pipeline!
    
    :df: dataframe containing raw data
    :lens: array containing the lens; same length as df
    :lens2: optional; array containing second lens
    :num_intervals: number of intervals to divide data into along the lens. if two lenses are used,
        total number of intervals will be num_intervals**2
    :f: float; overlap fraction between intervals
    :balance: bool; whether to adjust cover so that occupation is approximately equal
    :pca_dim: int; dimension to reduce data to within each index set. 0 to skip the PCA step.
    :k:
    :min_samples: int; min_samples parameter for OPTICS
    :xmeans:
    :aic:
    
    Returns:
    :cluster_indices: a list containing the raw-data indices associated with each cluster
    :g: NetworkX Graph object containing the Mapper graph
    """
    cover = mugatu._cover.compute_cover_indices(df.index.values, lens,
                                                lens2=lens2, num_intervals=num_intervals, 
                                                f=f, balance=balance)
    
    cluster_indices = mugatu._cluster.compute_clusters(df, cover, pca_dim=pca_dim, 
                                                       min_samples=min_samples, k=k,
                                                       xmeans=xmeans, aic=aic,
                                                       **kwargs)
    g = mugatu._graph.build_graph_from_indices(cluster_indices)
    return cluster_indices, g