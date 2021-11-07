#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 13 16:37:57 2021

@author: joe
"""
import numpy as np
import pandas as pd
import holoviews as hv
from mugatu._gui import _build_widgets, Mapperator

hv.extension("bokeh")

import dask
dask.config.set(scheduler="processes")

def test_build_widgets():
    # just make sure it runs i guess?
    colnames = ["foo", "bar"]
    lenses = ["lens_foo", "lens_bar"]
    widgets = _build_widgets(colnames, lenses)
    assert isinstance(widgets, dict)
    
N = 100
d = 4  
df = pd.DataFrame(np.random.normal(0, 1, (N,d)),
                  columns=[str(i) for i in range(d)])
lenses = {"foo":np.random.normal(0,1,N), "bar":np.random.normal(0,1,N)}
colors = {"foo_color":np.random.normal(0,1,N)}

def test_build_and_run_mapperator():
    mapper = Mapperator(df, lens_data=lenses, color_data=colors)
    mapper.build_mapper_model()
    mapper.update_node_positions()
    assert hasattr(mapper, "_node_df")
    assert hasattr(mapper, "_pos")