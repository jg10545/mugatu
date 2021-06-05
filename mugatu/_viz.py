#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:02:02 2021

@author: joe
"""
import numpy as np
import pandas as pd
import holoviews as hv


def _build_node_dataset(df, cluster_indices, lenses={}, include_indices=True):
    """
    Build a holoviews dataset object to contain summary stats on each node in the
    mapper graph
    """
    node_df = pd.DataFrame({
        "index":np.arange(len(cluster_indices)),
        "size":[len(c) for c in cluster_indices],
        })
    if include_indices:
        node_df["indices"] = [", ".join([str(i) for i in c]) for c in cluster_indices]
    
    if len(lenses) > 0:
        lens_df = pd.DataFrame(lenses, index=df.index)
        for l in lenses:
            node_df[l] = [np.mean(lens_df.loc[c,l]) for c in cluster_indices]
            
    for d in df.columns:
        # note: appending "data_" to column names is a workaround for an
        # annoying issue with HoloViews, where some column names appear
        # to be protexted in hv.Dataset (like "x")
        node_df["data_"+d] = [np.mean(df.loc[c,d]) for c in cluster_indices]
            
    return node_df


def _build_holoviews_fig(g, positions, node_df=None, color=[], width=800, 
                         height=600, node_size=20, cmap="plasma"):
    """
    
    """
    maxsize = np.max(node_df["size"])
    data = hv.Dataset(node_df, kdims=list(node_df.columns))
    fig = hv.Graph.from_networkx(g, positions=positions,
                                 nodes=data).opts(width=width,
                                            height=height, 
                                            xaxis=None, yaxis=None, edge_alpha=0.5,
                                            cmap=cmap, node_color=hv.dim(color),
                                            node_line_color="white",
                                            node_size=0.5*node_size*(1+hv.dim("size")/maxsize))
    return fig

def mapper_fig(g, positions, node_df=None, color=[], width=800, 
               height=600, node_size=20, cmap="plasma"):
    """
    
    """
    if isinstance(color, str):
        return _build_holoviews_fig(g, positions, node_df, color, width, height, node_size, cmap)
    elif isinstance(color, list):
        if (len(color) == 0)&(node_df is not None):
            color = [x for x in node_df.columns if x not in ["indices", "index"]]
        color_dict = {c:_build_holoviews_fig(g, positions, node_df, c, width, height, node_size, 
                                                    cmap) for c in color}
        return hv.HoloMap(color_dict, kdims="Color nodes by")
    else:
        assert False, "don't know what to do with the argument you passes to `color`"
    