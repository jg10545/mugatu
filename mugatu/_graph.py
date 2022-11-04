#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 30 13:42:22 2021

@author: joe
"""
import numpy as np
import networkx as nx

def build_graph_from_indices(indices):
    g = nx.Graph()
    sets = [set(i) for i in indices]
    for i in range(len(indices)):
        g.add_node(i)
        #set1 = set(indices[i])
        for j in range(i):
            #set2 = set(indices[j])
            #w = len(set1&set2)
            w = len(sets[i]&sets[j])
            if w > 0:
                g.add_edge(i,j, weight=np.log(w+1))
    return g