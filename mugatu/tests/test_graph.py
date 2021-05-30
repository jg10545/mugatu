#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:42:53 2021

@author: joe
"""
import networkx as nx

from mugatu._graph import build_graph_from_indices

def test_build_graph_from_indices():
    indices = [[0,1,2], [1,2,3], [0]]
    g = build_graph_from_indices(indices)
    assert isinstance(g, nx.Graph)