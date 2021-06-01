#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:42:22 2021

@author: joe
"""
import networkx as nx

def build_graph_from_indices(indices):
    g = nx.Graph()
    for i in range(len(indices)):
        g.add_node(i)
        set1 = set(indices[i])
        for j in range(i):
            set2 = set(indices[j])
            w = len(set1&set2)
            if w > 0:
                g.add_edge(i,j, weight=w)
    return g