#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 20:59:35 2021

@author: joe
"""
import pandas as pd
import networkx as nx
import panel as pn
import holoviews as hv
import io

from mugatu._util import _lens_dict
from mugatu._mapper import build_mapper_graph
from mugatu._viz import mapper_fig, _build_node_dataset

def _build_widgets(colnames, lenses, title=""):
    """
    generate all the Panel widgets
    """
    if len(title) > 0:
        filename = title.replace(" ", "_") + ".html"
    else:
        filename = "mugatu.html"
    # MODELING WIDGETS
    cross_selector = pn.widgets.CrossSelector(name='Inputs', value=colnames, options=colnames)
    lens1 = pn.widgets.Select(name="Lens 1", options=lenses)
    lens2 = pn.widgets.Select(name="Lens 2", options=["None"]+lenses)
    go_button = pn.widgets.Button(name="PUNCH IT, CHEWIE", button_type="success")
    progress = pn.indicators.Progress(name='Progress', value=100, width=600, active=False)
    pca_dim = pn.widgets.IntInput(name="PCA dimension (0 to disable)", value=max(int(len(colnames)/2),2))
    k = pn.widgets.IntInput(name="k", value=5)
    num_intervals = pn.widgets.IntInput(name="Number of intervals", value=5)
    overlap_frac = pn.widgets.FloatInput(name="Overlap fraction", value=0.25)
    balance = pn.widgets.Checkbox(name="Balance intervals", value=False)
    include_indices = pn.widgets.Checkbox(name="Include indices in visualization", value=False)
    model_layout = pn.Column(
    pn.Row(
        pn.Column(
            pn.pane.Markdown("### Variables to Include"),
            cross_selector,
        ),
        pn.Column(
            pn.pane.Markdown("### Lens"),
            lens1, lens2
        ),
    ), 
    pn.layout.Divider(),
    pn.Column(
        pn.Row(
            pca_dim,
            k,
            include_indices
        ),
        pn.Row(
            num_intervals,
            overlap_frac,
            balance
        )
    ),
    pn.layout.Divider(),
    pn.Row(go_button, progress))
    
    # DISPLAY WIDGETS
    pos_button = pn.widgets.Button(name="Reset layout", button_type="primary")
    hmap = hv.HoloMap({c:hv.Points(data=pd.DataFrame({"x":[0,1], "y":[1,2]})).opts(color=c) for c in ["red","blue"]})
    fig_panel = pn.panel(hmap)
    
    def _save_callback(*events):
        fig = fig_panel[0]
        bytesio = io.BytesIO()
        hv.save(fig, bytesio, backend="bokeh", resources="inline", title=title)
        bytesio.seek(0)
        return bytesio
    sav_button = pn.widgets.FileDownload(name="Download figure (may take a few minutes)",
                                         filename=filename,
                                         callback=_save_callback)
    layout = pn.layout.Tabs(("Modeling", model_layout), 
                            ("Visualization", pn.Column(fig_panel, sav_button)))
                          #("Visualization", pn.Column(pos_button, fig_panel)))
    return {"cross_selector":cross_selector, 
           "lens1":lens1,
           "lens2":lens2,
           "go_button":go_button,
           "progress":progress,
           "layout":layout,
           "k":k, 
           "pca_dim":pca_dim,
           "include_indices":include_indices,
           "fig_panel":fig_panel,
           "pos_button":pos_button,
           "num_intervals":num_intervals,
           "overlap_frac":overlap_frac,
           "balance":balance,
           "sav_button":sav_button}






def _compute_lenses(df, variables_to_include, lens_data=None,
                    old_variables_to_include=None, old_lenses=None,
                    compute=["svd", "isolation_forest"]):
    """
    Only recompute lenses if necessary
    """
    # check to see if we can skip the computation
    if old_variables_to_include is not None:
        if set(variables_to_include) == set(old_variables_to_include):
            return old_lenses
    else:
        return _lens_dict(df[variables_to_include], lens_data, compute=compute)


def _combine_dictionaries(d1,d2):
    d = {}
    if d1 is not None:
        for k in d1:
            d[k] = d1[k]
    if d2 is not None:
        for k in d2:
            d[k] = d2[k]
    if len(d) == 0:
        d = None
    return d

class Mapperator(object):
    
    def __init__(self, df, lens_data=None, compute=["svd", "isolation_forest"],
                 color_data=None, title=""):
        """
        
        """
        # store data and precompute lenses
        self.df = df
        self.lens_data = lens_data
        self.color_data = color_data
        self._node_df = None
        self._title = title
        include = df.columns.to_list()
        self._old_variables_to_include = include
        self._compute = compute
        self.lens_dict = _compute_lenses(df, include, lens_data, compute=compute)
        # set up gui
        self._widgets = _build_widgets(list(df.columns), list(self.lens_dict.keys()),
                                       title)
        self._widgets["go_button"].on_click(self.build_mapper_model)
        self._widgets["pos_button"].on_click(self.update_node_positions)
        
    def _update_lens(self):
        include = self._widgets["cross_selector"].values
        self.lens_dict = _compute_lenses(self.df, include, self.lens_data,
                                          self._old_variables_to_include, self.lens_dict,
                                          compute=self._compute)
        self._old_variables_to_include = include
        
    def _build_mapper_graph(self):
        w = self._widgets
        lens = self.lens_dict[w["lens1"].value]
        lens2 = w["lens2"].value
        if lens2 == "None":
            lens2 = None
        else:
            lens2 = self.lens_dict[lens2]
        
        cluster_indices, g = build_mapper_graph(self.df, lens, lens2, 
                                        num_intervals = w["num_intervals"].value,
                                        f = w["overlap_frac"].value, 
                                        balance = w["balance"].value,
                                        pca_dim = w["pca_dim"].value,
                                        k = w["k"].value)
        self._cluster_indices = cluster_indices
        self._g = g
        
    def _build_node_df(self):
        # if we have any exogenous information we'd like to color the nodes
        # by, combine that with the lens dict. The visualization will 
        # automatically add all of them as coloring options.
        exog = _combine_dictionaries(self.lens_dict, self.color_data)
        self._node_df = _build_node_dataset(self.df, 
                                            self._cluster_indices, 
                                            lenses=exog, 
                                            include_indices=self._widgets["include_indices"].value)
        
    def _update_fig(self):
        fig = mapper_fig(self._g, self._pos, node_df=self._node_df, width=600,
                         title=self._title)
        fig = pn.panel(fig)
        self._widgets["fig_panel"][0] = fig[0]
        self._widgets["fig_panel"][1] = fig[1]
        
    def _compute_node_positions(self):
        # see if we can start the nodes in reasonable positions using the singular 
        # values of the data
        if (self._node_df is not None)&("svd_1" in self._node_df.columns):
            pos_priors = {i:(self._node_df["svd_1"].values[i], 
                             self._node_df["svd_2"].values[i]) for i in range(len(self._node_df))}
        else:
            pos_priors = None
        self._pos = nx.layout.fruchterman_reingold_layout(self._g, pos=pos_priors)
        
    def update_node_positions(self, *events):
        """
        try layout out the mapper graph again
        """
        self._compute_node_positions()
        self._update_fig()
        
    def build_mapper_model(self, *events):
        """
        Pull parameters from the GUI, run the mapper algorithm, and 
        update the HoloViews display
        """
        self._widgets["progress"].value = 0
        self._widgets["progress"].active = True
        # update lenses if necessary
        self._update_lens()
        self._widgets["progress"].value = 20
        
        # build mapper graph
        self._build_mapper_graph()
        self._widgets["progress"].value = 40
        
        # build node dataframe
        self._build_node_df()
        self._widgets["progress"].value = 60
        
        # compute layout for visualization
        self._compute_node_positions()
        self._widgets["progress"].value = 80
        
        # build holoviews figure
        self._update_fig()
        self._widgets["progress"].value = 100
        # DONE
        self._widgets["progress"].active = False
        
    def panel(self):
        return self._widgets["layout"]


