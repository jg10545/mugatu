#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 29 20:40:27 2021

@author: joe
"""
import numpy as np
import pandas as pd
import sklearn.feature_extraction
import panel as pn
import re
from mugatu._rake import fasterrake

from mugatu._viz import mapper_fig, _build_node_dataset
from mugatu._gui import _combine_dictionaries, Mapperator

def build_rake_tdm(corpus, max_words=5, min_characters=1, min_frequency=1, stopwords=None):
    """
    
    """
    sklearn_pattern = re.compile('(?u)\\b\\w\\w+\\b')
    splitter = re.compile('(?u)\W+') # from RAKE.RAKE.separate_words
    corpus = [re.sub(splitter, " ", c) for c in corpus]
    keywords = fasterrake("\n".join(corpus), max_words=max_words, 
                           min_characters=min_characters, min_frequency=min_frequency, 
                           stopwords=stopwords)
    keyword_vocab = [k[0] for k in keywords if bool(re.match(sklearn_pattern, k[0]))&("\n" not in k[0])]
    vec = sklearn.feature_extraction.text.TfidfVectorizer(vocabulary=keyword_vocab, ngram_range=(1, max_words))
    tdm = vec.fit_transform(corpus)
    return keyword_vocab, tdm  


def find_high_tfidf_tokens(raw_text, cluster_indices, num_tokens=25, ngram_range=(2,2),
                          min_df=2, max_df=0.5):
    """
    
    """
    # aggregate text by Mapper graph node
    node_docs = [
        "\n\n".join([raw_text[i] for i in c])
        for c in cluster_indices
        ]
    # vectorize the text and get a term-document matrix
    vectorizer = sklearn.feature_extraction.text.TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, 
                                                                 max_df=max_df)
    node_tdm = vectorizer.fit_transform(node_docs)
    voc_lookup = {vectorizer.vocabulary_[k]:k for k in vectorizer.vocabulary_}
    
    # now find high-tfidf tokens for each node, aggregated to a single 
    # comma-delimited string
    def _ind_to_string(indices, i):
        return ", ".join([voc_lookup[k] for k in indices if node_tdm[i,k] > 0])
    token_list = [
        _ind_to_string(node_tdm[i,:].toarray().ravel().argsort()[::-1][:num_tokens],i)
        for i in range(len(cluster_indices))
    ]
    return token_list


def find_high_tfidf_keywords(df, keywords, tdm, cluster_indices, num_tokens=10):
    """
    :df:
    :keywords
    :tdm: a (num_documents, vocab_size) sparse matrix
    :cluster_indices:
    """
    # get the indices for each cluster
    sdf = pd.Series(data=np.arange(len(df)), index=df.index.values)
    indices = [sdf[c].values for c in cluster_indices]

    means = np.zeros(tdm.shape[1])
    stddevs = np.zeros(tdm.shape[1])
    for i in indices:
        means += np.array(tdm[i].mean(0)).ravel()/len(indices)
        
    for i in indices:
        stddevs += ((np.array(tdm[i].mean(0)).ravel()-means)**2)/len(indices)
    stddevs = np.sqrt(stddevs)
    token_list = []
    for i in indices:
        dense = (np.array(tdm[i].mean(0)).ravel() - means)/(stddevs+1)
        ordering = dense.argsort()[::-1][:num_tokens]
        tokenstring = ", ".join([keywords[k] for k in ordering if dense[k] > 0])
        token_list.append(tokenstring)

    return token_list


class TextMapperator(Mapperator):
    """
    """
    
    def __init__(self, df, tdm, vocabulary, lens_data=None, compute=["svd", "isolation_forest", "l2"],
                 color_data=None, title="", mlflow_uri=None, num_tokens=10):
        """
        :df: pandas DataFrame raw data to cluster on. The dataframe index
            will be used to relate nodes on the Mapper graph back to data
            points.
        :text:
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
        #self._text = {}#text
        #for i,t in zip(df.index.values, text):
        #    self._text[i] = t
            
        self._token_params = {"num_tokens":num_tokens}
        
        self._meta_df = df
        self._vocab = vocabulary
        #vecs_df = pd.DataFrame(vecs, index=df.index)
        #full_df = pd.concat([df, vecs_df], 1)        
            
        super().__init__(df, lens_data, compute, color_data, title, mlflow_uri, sparse_data=tdm)
        
    def _build_node_df(self):
        # if we have any exogenous information we'd like to color the nodes
        # by, combine that with the lens dict. The visualization will 
        # automatically add all of them as coloring options.
        exog = _combine_dictionaries(self.lens_dict, self.color_data)
        p = self._params
        self._node_df = _build_node_dataset(self._meta_df, 
                                            self._cluster_indices, 
                                            lenses=exog, 
                                            include_indices=p["include_indices"])
        #self._node_df["tokens"] = find_high_tfidf_tokens(self._text, self._cluster_indices,
        #                                                **self._token_params)
        self._node_df["tokens"] = find_high_tfidf_keywords(self.df, self._vocab, self._sparse_data, 
                                   self._cluster_indices, **self._token_params)
        
    def _update_fig(self):
        fig = mapper_fig(self._g, self._pos, node_df=self._node_df, width=600,
                         color=self._color_names,
                         title=self._title, extra_tooltips=[("Tokens", "@tokens")])
        fig = pn.panel(fig)
        self._widgets["fig_panel"][0] = fig[0]
        self._widgets["fig_panel"][1] = fig[1]