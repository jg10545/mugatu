#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:03:24 2021

@author: joe
"""
import numpy as np
import pandas as pd
import networkx as nx
import holoviews as hv

from mugatu._mapper import build_mapper_graph
from mugatu._viz import _build_node_dataset
from mugatu._viz import _build_holoviews_fig, mapper_fig


N = 100
data = pd.DataFrame({"a":np.random.normal(0,1,N), "b":np.random.normal(0,1,N)})
lens = data.a.values
lens2 = data.b.values
num_intervals = 2
f = 0.1
balance = False
pca_dim = 0
k = 2
clust_ind, g = build_mapper_graph(data, lens, num_intervals=num_intervals, f=f, balance=balance,
                                     pca_dim=pca_dim, k=k)

def test_build_node_dataset():
    ds = _build_node_dataset(data, clust_ind, lenses={}, include_indices=True)
    assert isinstance(ds, pd.DataFrame)
    assert len(ds) == len(clust_ind)
    
def test_build_node_dataset_without_indices():
    ds = _build_node_dataset(data, clust_ind, lenses={}, include_indices=False)
    assert isinstance(ds, pd.DataFrame)
    assert len(ds) == len(clust_ind)
    
def test_build_node_dataset_with_lenses():
    ds = _build_node_dataset(data, clust_ind, lenses={"a":lens, "b":lens2}, include_indices=True)
    assert isinstance(ds, pd.DataFrame)
    assert len(ds) == len(clust_ind)
    
"""
NEED TO IMPORT A BOKEH PLOTTING EXTENSION FIRST

def test_build_holoviews_fig():
    ds = _build_node_dataset(data, clust_ind, lenses={"lens":data.a.values}, include_indices=True)
    pos = nx.layout.fruchterman_reingold_layout(g, iterations=1)
    fig = _build_holoviews_fig(g, pos, node_df=ds, color="lens",width=800, 
                               height=600, node_size=20, cmap="plasma")
    
    assert isinstance(fig, hv.Graph)
    
def test_mapper_fig():
    ds = _build_node_dataset(data, clust_ind, lenses={"lens":data.a.values, "lens2":data.b.values}, 
                             include_indices=True)
    pos = nx.layout.fruchterman_reingold_layout(g, iterations=1)
    fig = mapper_fig(g, pos, node_df=ds, color="lens",width=800, 
                               height=600, node_size=20, cmap="plasma")
    assert isinstance(fig, hv.Graph)
    
def test_mapper_fig_holomap():
    ds = _build_node_dataset(data, clust_ind, lenses={"lens":data.a.values, "lens2":data.b.values}, 
                             include_indices=True)
    pos = nx.layout.fruchterman_reingold_layout(g, iterations=1)
    fig = mapper_fig(g, pos, node_df=ds, color=["lens", "lens2"],width=800, 
                               height=600, node_size=20, cmap="plasma")
    assert isinstance(fig, hv.HoloMap)
"""