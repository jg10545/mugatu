"""
Created on Mon May 31 21:02:02 2021

@author: joe
"""
import numpy as np
import pandas as pd
import holoviews as hv
from bokeh.models import HoverTool, LassoSelectTool, BoxSelectTool, TapTool


def _build_node_dataset_DEPRECATED(df, cluster_indices, lenses={}, include_indices=True, num=3):
    """

    """
    num = min(num, len(df.columns)-1)
    # for each node- record the number of constituent data points
    node_df = pd.DataFrame({
        "index":np.arange(len(cluster_indices)),
        "size":[len(c) for c in cluster_indices],
        })
    # next- for each node, compute the average value of every covariate.
    mean_cols = []
    for d in df.columns:
        # skip if all values are the same
        if df[d].std() > 0:
            node_df[f"mean_{d}"] = [np.mean(df.loc[c,d]) for c in cluster_indices]
            mean_cols.append(f"mean_{d}")

    # the above creates way too much information to be readable in the hover tool.
    # so for each node just find unusually high or low values. compute these by
    # looking at node means shifted by the mean of all node means, and scaled by
    # the standard deviation of node means.
    rel_means = (node_df[mean_cols] - node_df[mean_cols].mean())/node_df[mean_cols].std()
    sorted_means = rel_means.values.argsort(1)

    cols = [str(c) for c in list(df.columns)]
    highest = sorted_means[:,-num:][:,::-1]
    node_df["high"] = [", ".join([cols[i] for i in highest[j,:] if
                                     rel_means.values[j,i] > 0]) for j in
                          range(len(highest))]
    lowest = sorted_means[:,:num]
    node_df["low"] = [", ".join([cols[i] for i in lowest[j,:] if
                                rel_means.values[j,i] < 0]) for j in
                      range(len(lowest))]

    #node_df = node_df.drop(mean_cols,1)

    if len(lenses) > 0:
        lens_df = pd.DataFrame(lenses, index=df.index)
        for l in lenses:
            node_df[l] = [np.mean(lens_df.loc[c,l]) for c in cluster_indices]

    if include_indices:
        node_df["indices"] = [", ".join([str(i) for i in c]) for c in cluster_indices]
    return node_df

def _build_node_dataset(df, cluster_indices, lenses={}, include_indices=True, num=3):
    """
    """
    num = min(num, len(df.columns)-1)
    # for each node- record the number of constituent data points
    node_df = pd.DataFrame({
        "index":np.arange(len(cluster_indices)),
        "size":[len(c) for c in cluster_indices],
        })
    # next- for each node, compute the average value of every covariate.
    mean_cols = []
    for d in df.columns:
        # skip if all values are the same
        if df[d].std() > 0:
            node_df[f"mean_{d}"] = [np.mean(df.loc[c,d]) for c in cluster_indices]
            mean_cols.append(f"mean_{d}")

    # the above creates way too much information to be readable in the hover tool.
    # so for each node just find unusually high or low values. compute these by
    # looking at node means shifted by the mean of all node means, and scaled by
    # the standard deviation of node means.
    rel_means = (node_df[mean_cols] - node_df[mean_cols].mean())/node_df[mean_cols].std()
    sorted_means = rel_means.values.argsort(1)

    cols = [str(c) for c in list(df.columns)]
    highest = sorted_means[:,-num:][:,::-1]
    node_df["high"] = [", ".join([cols[i] for i in highest[j,:] if
                                     rel_means.values[j,i] > 0]) for j in
                          range(len(highest))]
    lowest = sorted_means[:,:num]
    node_df["low"] = [", ".join([cols[i] for i in lowest[j,:] if
                                rel_means.values[j,i] < 0]) for j in
                      range(len(lowest))]

    #node_df = node_df.drop(mean_cols,1)

    if len(lenses) > 0:
        lens_df = pd.DataFrame(lenses, index=df.index)
        for l in lenses:
            node_df[l] = [np.mean(lens_df.loc[c,l]) for c in cluster_indices]

    if include_indices:
        node_df["indices"] = [", ".join([str(i) for i in c]) for c in cluster_indices]
    return node_df

def _build_holoviews_fig(g, positions, node_df=None, color=[], width=800,
                         height=600, node_size=20, cmap="plasma", title="",
                         tooltips=None, extra_tooltips=[]):
    """

    """
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
        for t in extra_tooltips:
            tooltips.append(t)
        if isinstance(color, str):
            tooltips.append((color, "@"+color))
        else:
            for c in color:
                tooltips.append((c, "@"+c))
        tools = [HoverTool(tooltips=tooltips),
                 LassoSelectTool(), BoxSelectTool()]
        node_df = hv.Dataset(node_df, kdims=list(node_df.columns))
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


def _prep_linked_fig(g, node_df, pos=None):
    """

    """
    if pos is None:
        pos = nx.layout.fruchterman_reingold_layout
    graphfig = hv.Graph.from_networkx(g, pos)
    edgefig = graphfig.edgepaths.opts(line_alpha=0.5)
    nodedata = graphfig.nodes.data

    cols = [c for c in node_df.columns if c not in ["low", "high"]]
    merged = nodedata.merge(node_df[cols], on="index", how="inner")

    return merged, edgefig


def _build_linked_holoviews_fig(df, edgefig, x, y, colors=[], width=600,
                                height=300, node_size=20, cmap="plasma", title=""):
    """

    """
    maxsize = df.size.max()
    tooltips = [
        ('Index', '@index'),
        ('Size', '@size'),
        ('High', '@high'),
        ('Low', '@low')
    ]
    tools = [HoverTool(tooltips=tooltips),
             LassoSelectTool(), BoxSelectTool(), TapTool()]
    #tools = ["hover", "box_select", "lasso_select", "tap"]

    def _genmap(color, x, y):
        opts = {
            "colorbar":True,
            "tools":tools,
            "size":0.5 * node_size * (1 + hv.dim("size") / maxsize),
            "color":hv.dim(color),
            "cmap":cmap,
            "nonselection_fill_alpha":0.,
            "selection_fill_alpha":0.75
        }
        nodefig = hv.Points(df, kdims=["x", "y"]).opts(xaxis=None, yaxis=None,
                                                       title="mapper graph", **opts)
        featurefig = hv.Points(df, kdims=[x, y]).opts(show_grid=True,
                                                      title="node attributes", **opts)

        return ((edgefig * nodefig).opts(width=width, height=height) + featurefig.opts(width=width, height=height)).opts(
            hv.opts.Layout(shared_datasource=True)).cols(1)

    if isinstance(colors, list):
        hmdict = {c: _genmap(c, x, y)
              for c in colors}
        return hv.HoloMap(hmdict, kdims=["Color nodes by"]).collate()
    else:
        return _genmap(colors,x,y)
