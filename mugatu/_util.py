#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 09:53:01 2021

@author: joe
"""
import numpy as np
import pandas as pd
import sklearn.preprocessing, sklearn.decomposition
import sklearn.neighbors, sklearn.ensemble
import logging

def _lens_dict(data, lens_data=None, compute=["svd", "isolation_forest", "kde"]):
    """
    Generate a dictionary of arrays that can be used as lenses
    
    :data: the pandas DataFrame of original data
    :lens_data: optional; pandas DataFrame of precomputed lenses
    :compute: optional; list of generic lenses we can precompute
    
    Returns
    :lens_dict: dictionary of numpy arrays of same length as data
    """
    lenses = {}
            
    if lens_data is not None:
        if isinstance(lens_data, dict):
            for d in lens_data:
                assert len(lens_data[d]) == len(data), f"lens data for {d} appears to be the wrong shape"
                lenses[d] = lens_data[d]
        elif isinstance(lens_data, pd.DataFrame):
            assert len(lens_data) == len(data), "lens_data appears to be the wrong length"
            for d in lens_data.columns:
                lenses[d] = lens_data[d].values
            
    data_rescaled = sklearn.preprocessing.StandardScaler().fit_transform(data.values)
    
    if "svd" in compute:
        # SVD
        logging.info("precomputing SVD lenses")
        svd = sklearn.decomposition.TruncatedSVD(2).fit_transform(data_rescaled)
        lenses["svd_1"] = svd[:,0]
        lenses["svd_2"] = svd[:,1]

    if "isolation_forest" in compute:
        logging.info("precomputing IsolationForest lense")
        isoforest = sklearn.ensemble.IsolationForest(n_jobs=-1).fit(data_rescaled)
        lenses["isolation_forest"] = isoforest.decision_function(data_rescaled)
        
    if "kde" in compute:
        logging.info("precomputing kernel density estimate lense")
        kde = sklearn.neighbors.KernelDensity(kernel="gaussian").fit(data_rescaled)
        lenses["kernel_density_estimate"] = kde.score_samples(data_rescaled)
    return lenses