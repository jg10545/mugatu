#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 31 21:02:02 2021

@author: joe
"""
import numpy as np
import pandas as pd
import networkx as nx
import holoviews as hv
from bokeh.models import HoverTool
import scipy.sparse

def _build_node_dataset(X, columns, rows, cluster_indices, lenses={}, 
                        include_indices=True, num=5):
    """
    
    """
    num = min(num, len(columns)-1)
    # compute means for each column over each cluster
    means = np.stack([np.array(X[c].mean(0)).ravel() for c in cluster_indices], 0)
    mean_cols = [f"mean_{c}" for c in columns]
    node_df = pd.DataFrame(means, columns=mean_cols)
    node_df["size"] = [len(c) for c in cluster_indices]
    node_df["index"] = np.arange(len(cluster_indices))
        
    # the above creates way too much information to be readable in the hover tool.
    # so for each node just find unusually high or low values. compute these by
    # looking at node means shifted by the mean of all node means, and scaled by
    # the standard deviation of node means.
    rel_means = (node_df[mean_cols] - node_df[mean_cols].mean())/node_df[mean_cols].std()
    sorted_means = rel_means.values.argsort(1)
    
    cols = [str(c) for c in list(columns)]
    highest = sorted_means[:,-num:][:,::-1]
    node_df["high"] = [", ".join([cols[i] for i in highest[j,:] if 
                                     rel_means.values[j,i] > 0]) for j in
                          range(len(highest))]
    lowest = sorted_means[:,:num]
    node_df["low"] = [", ".join([cols[i] for i in lowest[j,:] if
                                rel_means.values[j,i] < 0]) for j in 
                      range(len(lowest))]
    
    if len(lenses) > 0:
        lens_df = pd.DataFrame(lenses, index=np.arange(X.shape[0]))
        for l in lenses:
            node_df[l] = [np.mean(lens_df.loc[c,l]) for c in cluster_indices]
    
    if include_indices:
        node_df["row_indices"] = [", ".join([str(i) for i in np.array(rows)[c]]) for c in cluster_indices]
    return node_df

        

def _build_holoviews_fig(g, positions, node_df=None, color=[], width=900, 
                         height=700, node_size=20, cmap="plasma", title="",
                         tooltips=None, extra_tooltips=[]):
    """
    
    """
    cols = ["index", "size", "high", "low"]
    if isinstance(color, str):
        cols.append(color)
    maxsize = np.max(node_df["size"])
    if node_df is not None:
        if tooltips is None:
            tooltips = [
                ('Index', '@index'),
                ('Size', '@size'),
                ('High', '@high'),
                ('Low', '@low')
                ]  
        if "indices" in node_df.columns:
            tooltips.append(('Indices', '@indices'))
            cols.append("indices")
        for t in extra_tooltips:
            tooltips.append(t)
        if isinstance(color, str):
            tooltips.append((color, "@"+color))
        else:
            for c in color:
                tooltips.append((c, "@"+c))
        tools = [HoverTool(tooltips=tooltips)]
        #node_df = hv.Dataset(node_df, kdims=list(node_df.columns))
        node_df = hv.Dataset(node_df[cols], kdims=cols)
    else:
        tools = []
    fig = hv.Graph.from_networkx(g, positions=positions,
                                 nodes=node_df).opts(width=width,
                                            height=height, 
                                            xaxis=None, yaxis=None, edge_alpha=0.5,
                                            cmap=cmap, node_color=hv.dim(color),
                                            node_line_color="white", title=title,
                                            node_size=0.5*node_size*(1+hv.dim("size")/maxsize),
                                            colorbar=True, tools=tools)
    return fig

def mapper_fig(g, positions, node_df=None, color=[], width=800, 
               height=600, node_size=20, cmap="plasma", title="",
               extra_tooltips=[]):
    """
    Generate a holoviews figure displaying a mapper graph
    
    :g: networkx Graph object representing the mapper graph
    :positions: positions of nodes in the graph- dictionary of tuples (output
                    of any of the nx.layout functions)
    :node_df: dataframe mapping node indices to summary statistics
    :color: string or list of strings; column names from node_df to use for 
        coloring graph nodes. if an empty list, use all columns
    :width: width of figure
    :height: height of figure
    :node_size: global scale of nodes in graph (individual nodes will be bigger
            or smaller depending on the number of constituent data points)
    :cmap: colormap to use.
        
        
    Returns
    A Holoviews Graph or HoloMap object if color is a string or list, respectively
    """
    if isinstance(color, str):
        return _build_holoviews_fig(g, positions, node_df, color, width, 
                                    height, node_size, cmap, title,
                                    extra_tooltips=extra_tooltips)
    elif isinstance(color, list):
        if (len(color) == 0)&(node_df is not None):
            color = [x for x in node_df.columns if x not in ["indices", "index", 
                                                             "high", "low"]]
        color_dict = {c:_build_holoviews_fig(g, positions, node_df, c, width, 
                                             height, node_size,
                                             cmap, title, extra_tooltips=extra_tooltips)
                      for c in color}
        return hv.HoloMap(color_dict, kdims=["Color nodes by"])
    else:
        assert False, "don't know what to do with the argument you passes to `color`"
    
    
    
def _compute_node_positions(node_df, g, lens1, lens2=None):
    """
    Heuristics to choose reasonable initialization for Mapper node layouts
    
    :node_df: pandas dataframe containing averaged lens values for each node
    :g: networkx graph object containing mapper graph
    :lens1: string; name of first lens
    :lens2: string; name of second lens if using
    """
    # 1D lens: use lens and first singular value
    if (lens2 is None)&(lens1 != "svd_1")&("svd_1" in node_df.columns):
        pos_priors = {i:(node_df[lens1].values[i], 
                             node_df["svd_1"].values[i]) for i in
                      range(len(node_df))}
    # 2D lens: use the lenses
    elif lens2 is not None:
        pos_priors = {i:(node_df[lens1].values[i], 
                             node_df[lens2].values[i]) for i in
                      range(len(node_df))}
    elif ("svd_1" in node_df.columns)&("svd_2" in node_df.columns):
        pos_priors = {i:(node_df["svd_1"].values[i], 
                             node_df["svd_2"].values[i]) for i in
                      range(len(node_df))}
    else:
        pos_priors = None
        
    k = 0.01/np.sqrt(len(g.nodes))
    pos = nx.layout.fruchterman_reingold_layout(g, k=k, pos=pos_priors)
    return pos