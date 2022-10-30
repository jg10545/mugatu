#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun  6 20:59:35 2021

@author: joe
"""
import numpy as np
import pandas as pd
import networkx as nx
import panel as pn
import holoviews as hv
import io
import logging
import mlflow

from mugatu._util import _lens_dict
from mugatu._mapper import build_mapper_graph
from mugatu._viz import mapper_fig, _build_node_dataset
from mugatu._metrics import _compute_node_measure_concentrations

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
    cluster_select = pn.widgets.Select(name="Clustering method", 
                                       options=["k-means", "x-means (AIC)", 
                                                "x-means (BIC)", "OPTICS"],
                                       value="x-means (BIC)")
    k = pn.widgets.IntInput(name="k (k-means and x-means only)", value=2)
    min_samples = pn.widgets.IntInput(name="Minimum cluster size (OPTICS and x-means only)",
                                      value=5)
    num_intervals = pn.widgets.IntInput(name="Number of intervals", value=5)
    overlap_frac = pn.widgets.FloatInput(name="Overlap fraction", value=0.25)
    balance = pn.widgets.Checkbox(name="Balance intervals", value=False)
    include_indices = pn.widgets.Checkbox(name="Include indices in visualization", value=False)
    experiment_name = pn.widgets.TextInput(name="MLflow experiment", value="derelicte")
    log_button = pn.widgets.Button(name="Log to MLflow for all posterity",
                                   button_type="primary", align="end")
    
    mlflow_layout = pn.Row(experiment_name, log_button)
    status = pn.pane.Markdown("*waiting*", width=200) 
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
            num_intervals,
            overlap_frac,
            pca_dim
        ),
        pn.Row(
            cluster_select,
            k,
            min_samples
            ),
        pn.Row(
            balance, include_indices
            )
    ),
    pn.layout.Divider(),
    pn.Row(go_button, progress))
    
    # DISPLAY WIDGETS
    #pos_button = pn.widgets.Button(name="Reset layout", button_type="primary")
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
                            ("Visualization", pn.Column(fig_panel, sav_button,
                                                        mlflow_layout)))
    return {"cross_selector":cross_selector, 
           "lens1":lens1,
           "lens2":lens2,
           "go_button":go_button,
           "progress":progress,
           "layout":layout,
           "k":k, 
           "min_samples":min_samples,
           "pca_dim":pca_dim,
           "include_indices":include_indices,
           "fig_panel":fig_panel,
           "num_intervals":num_intervals,
           "overlap_frac":overlap_frac,
           "balance":balance,
           "sav_button":sav_button,
           "experiment_name":experiment_name,
           "log_button":log_button,
           "cluster_select":cluster_select,
           "status":status}






def _compute_lenses(df, variables_to_include, lens_data=None,
                    old_variables_to_include=None, old_lenses=None,
                    compute=["svd", "isolation_forest", "l2"]):
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
    """
    Class for interactive modeling with Mapper
    """
    
    def __init__(self, df, lens_data=None, compute=["svd", "isolation_forest", "l2"],
                 color_data=None, title="", mlflow_uri=None, sparse_data=None):
        """
        :df: pandas DataFrame raw data to cluster on. The dataframe index
            will be used to relate nodes on the Mapper graph back to data
            points.
        :lens_data: pandas DataFrame (or dictionary of arrays); additional lenses that can be used
            to filter your data
        :compute: listof strings; generic lenses to precompute. can include:
            -"svd" computes first and second singular value decomposition vectors
            -"isolation_forest" assigns an anomaly score from an Isolation Forest
            -"l2" rescales the data so each column has zero mean and unit 
                variance, then records the L2-norm of each data point
            -"kde" estimates local density of each record using a kernel density
                estimator. Very slow on large datasets!
        :color_data: pandas DataFrame (or dictionary of arrays); additional data to use for coloring
            nodes in the Mapper graph
        :title: string; title for the figure
        :mlflow_uri: string; location of MLflow server for logging results
        """
        # store data and precompute lenses
        self.df = df
        self._sparse_data = sparse_data
        # if lens_data or color_data are dataframes- turn them
        # into dictionaries
        if isinstance(lens_data, pd.DataFrame):
            lens_data = {c:lens_data[c].values for c in lens_data.columns}
        if isinstance(color_data, pd.DataFrame):
            color_data = {c:color_data[c].values for c in color_data.columns}
        self.lens_data = lens_data
        self.color_data = color_data
        # what color options do we have for the figure?
        self._color_names = [x for x in compute if x != "svd"]
        if "svd" in compute:
            self._color_names += ["svd_1", "svd_2"]
        if lens_data is not None:
            self._color_names += list(lens_data.keys())
        if color_data is not None:
            self._color_names += list(color_data.keys())
        self._node_df = None
        self._title = title
        include = df.columns.to_list()
        self._old_variables_to_include = include
        self._compute = compute
        self.lens_dict = _compute_lenses(df, include, lens_data, compute=compute)
        self.mlflow_uri = mlflow_uri
        if mlflow_uri is not None:
            mlflow.set_tracking_uri(mlflow_uri)
        # set up gui
        self._widgets = _build_widgets(list(df.columns), list(self.lens_dict.keys()),
                                       title)
        self._widgets["go_button"].on_click(self.build_mapper_model)
        self._widgets["log_button"].on_click(self._mlflow_callback)
        
    def _update_lens(self):
        p = self._params
        self.lens_dict = _compute_lenses(self.df, p["include"], self.lens_data,
                                          self._old_variables_to_include, self.lens_dict,
                                          compute=self._compute)
        self._old_variables_to_include = p["include"]
        
    def _build_mapper_graph(self):
        p = self._params
        lens = self.lens_dict[p["lens1"]]
        lens2 = p["lens2"]
        if lens2 is not None:
            lens2 = self.lens_dict[lens2]
            
        # parse the clustering algorithm selection
        c = p["cluster_select"]
        k = p["k"]
        xmeans = "x-means" in c
        aic = "AIC" in c
        if c == "OPTICS":
            k = 0

        cluster_indices, g = build_mapper_graph(self.df, lens, lens2, 
                                        num_intervals = p["num_intervals"],
                                        f = p["overlap_frac"], 
                                        balance = p["balance"],
                                        pca_dim = p["pca_dim"],
                                        min_samples=p["min_samples"],
                                        k=k, xmeans=xmeans, aic=aic,
                                        sparse_data=self._sparse_data)
        self._cluster_indices = cluster_indices
        self._g = g
        
    def _build_node_df(self):
        # if we have any exogenous information we'd like to color the nodes
        # by, combine that with the lens dict. The visualization will 
        # automatically add all of them as coloring options.
        exog = _combine_dictionaries(self.lens_dict, self.color_data)
        p = self._params
        self._node_df = _build_node_dataset(self.df, 
                                            self._cluster_indices, 
                                            lenses=exog, 
                                            include_indices=p["include_indices"])
        
    def _update_fig(self):
        fig = mapper_fig(self._g, self._pos, node_df=self._node_df, width=600,
                         color=self._color_names,
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
        k = 0.01/np.sqrt(len(self._g.nodes))
        self._pos = nx.layout.fruchterman_reingold_layout(self._g, k=k, pos=pos_priors)
        
    def update_node_positions(self, *events):
        """
        try layout out the mapper graph again
        """
        self._compute_node_positions()
        self._update_fig()
        
    def _collect_params(self):
        """
        Get any important parameters from the GUI
        """
        w = self._widgets
        params = {
            "include":w["cross_selector"].values,
            "lens1":w["lens1"].value,
            "lens2":w["lens2"].value,
            "num_intervals":w["num_intervals"].value,
            "overlap_frac":w["overlap_frac"].value, 
            "balance":w["balance"].value,
            "pca_dim":w["pca_dim"].value,
            "min_samples":w["min_samples"].value,
            "k":w["k"].value,
            "include_indices":w["include_indices"].value,
            "cluster_select":w["cluster_select"].value
            }
        if params["lens2"] == "None":
            params["lens2"] = None
        self._params = params
        
    def _update_filename(self):
        p = self._params
        alg = p["cluster_select"].replace(" ","_").replace("(","").replace(")","")
            
        if p["lens2"] is None:
            lens = p["lens1"]
        else:
            lens = p["lens1"] + "+" + p["lens2"]
            
        filename = f"mugatu_{alg}_{lens}.html"
        self._widgets["sav_button"].filename = filename
        
    def _mlflow_callback(self, record_fig=True, *events):
        """
        
        """
        if self.mlflow_uri is not None:
            mlflow.set_experiment(self._widgets["experiment_name"].value)
            p = self._params
            params = {k:p[k] for k in p if k not in ["include"]}

            # save out the figure so we can include it as an artifact
            if record_fig:
                # get holoviews html file as a BytesIO object
                bio = self._widgets["sav_button"].callback()
                # write to file
                with open("figure.html", "wb") as f:
                    f.write(bio.read())
                    
            # compute concentration values for graph colorings
            if self.color_data is not None:
                coldict = {c:self._node_df[c].values for c in self.color_data}
                concentrations = _compute_node_measure_concentrations(coldict, self._g)
                
            with mlflow.start_run():
                mlflow.log_params(params)
                if record_fig:
                    mlflow.log_artifact("figure.html")
                if self.color_data is not None:
                    mlflow.log_metrics(concentrations)
        
        
    def build_mapper_model(self, *events):
        """
        Pull parameters from the GUI, run the mapper algorithm, and 
        update the HoloViews display
        """
        self._widgets["progress"].value = 0
        self._widgets["progress"].active = True
        self._collect_params()
        self._update_filename()
        # update lenses if necessary
        logging.info("updating lenses")
        self._update_lens()
        self._widgets["progress"].value = 20
        
        # build mapper graph
        logging.info("building mapper graph")
        self._build_mapper_graph()
        self._widgets["progress"].value = 40
        
        # build node dataframe
        logging.info("computing node statistics")
        self._build_node_df()
        self._widgets["progress"].value = 60
        
        # compute layout for visualization
        logging.info("computing graph layout")
        self._compute_node_positions()
        self._widgets["progress"].value = 80
        
        # build holoviews figure
        logging.info("building figure")
        self._update_fig()
        self._widgets["progress"].value = 100
        # DONE
        self._widgets["progress"].active = False
        
    def panel(self):
        """
        Return the panel object for the GUI.
        """
        return self._widgets["layout"]
    
    def template(self):
        """
        Return app as a panel template. Call template.show() to
        open interface in new window
        """
        vanilla = pn.template.VanillaTemplate(title="mugatu")

        for c in ["lens1", "lens2", "num_intervals", "overlap_frac", "pca_dim",
                  "cluster_select", "min_samples", "balance", "include_indices", 
                  "go_button", "progress", "status", "sav_button", 
                  "experiment_name", "log_button"]:
            vanilla.sidebar.append(self._widgets[c])
    
        vanilla.main.append(self._widgets["fig_panel"])
        return vanilla
    
    def show(self):
        if not hasattr(self, "_template"):
            self._template = self.template()
            
        self._template.show()


