import numpy as np
import pandas as pd
import panel as pn


from mugatu._viz import mapper_fig, _build_node_dataset
from mugatu._gui import _combine_dictionaries, Mapperator


def find_high_tfidf_keywords(df, keywords, tdm, cluster_indices, num_tokens=10):
    """
    :df:
    :keywords
    :tdm: a (num_documents, vocab_size) sparse matrix
    :cluster_indices:
    """
    # get the indices for each cluster (since pandas can have arbitrary indices
    # but our sparse matrix has to be indexed by 0, 1, 2, ...N-1)
    sdf = pd.Series(data=np.arange(len(df)), index=df.index.values)
    indices = [sdf[c].values for c in cluster_indices]

    # Compute the mean and standard deviation of the CLUSTER AVERAGES of
    # each token. we'll use these to identify tokens that are interesting
    # for a CLUSTER compared to other clusters
    means = np.zeros(tdm.shape[1])
    stddevs = np.zeros(tdm.shape[1])
    for i in indices:
        means += np.array(tdm[i].mean(0)).ravel()/len(indices)
    for i in indices:
        stddevs += ((np.array(tdm[i].mean(0)).ravel()-means)**2)/len(indices)
    stddevs = np.sqrt(stddevs)

    # Now build a string describing interesting tokens for each cluster, by averaging
    # the rows of the term-document matrix for that cluster, shifting by the means and
    # scaling by the standard deviations we computed above, then finding the highest
    # remaining values (and requiring that they be above zero)
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

    def __init__(self, df, tdm, vocabulary, lens_data=None,
                 compute=["svd", "isolation_forest", "l2"],
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
        self._token_params = {"num_tokens":num_tokens}

        self._meta_df = df
        self._vocab = vocabulary

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

        self._node_df["tokens"] = find_high_tfidf_keywords(self.df, self._vocab, self._sparse_data,
                                   self._cluster_indices, **self._token_params)

    def _update_fig(self):
        fig = mapper_fig(self._g, self._pos, node_df=self._node_df, width=600,
                         color=self._color_names,
                         title=self._title, extra_tooltips=[("Tokens", "@tokens")])
        fig = pn.panel(fig)
        self._widgets["fig_panel"][0] = fig[0]
        self._widgets["fig_panel"][1] = fig[1]
