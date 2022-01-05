#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 20:35:59 2021

@author: joe
"""
import numpy as np
import pandas as pd
import networkx as nx

from mugatu._mapper import build_mapper_graph



def test_build_mapper_graph_one_lens():
    N = 100
    data = pd.DataFrame({"a":np.random.normal(0,1,N), "b":np.random.normal(0,1,N)})
    lens = data.a.values
    num_intervals = 2
    f = 0.1
    balance = False
    pca_dim = 0
    min_samples = 5
    clust_ind, g = build_mapper_graph(data.values, lens, num_intervals=num_intervals, f=f, balance=balance,
                                     pca_dim=pca_dim, min_samples=min_samples)
    assert isinstance(clust_ind, list)
    assert isinstance(g, nx.Graph)
    assert len(g.nodes) >= num_intervals

def test_build_mapper_graph_two_lenses():
    N = 100
    data = pd.DataFrame({"a":np.random.normal(0,1,N), "b":np.random.normal(0,1,N)})
    lens = data.a.values
    lens2 = data.b.values
    num_intervals = 2
    f = 0.1
    balance = False
    pca_dim = 0
    #k = 2
    min_samples = 5
    clust_ind, g = build_mapper_graph(data.values, lens, lens2=lens2, num_intervals=num_intervals, f=f, balance=balance,
                                     pca_dim=pca_dim, min_samples=min_samples)
    assert isinstance(clust_ind, list)
    assert isinstance(g, nx.Graph)
    assert len(g.nodes) >= num_intervals