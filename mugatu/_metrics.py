"""
Created on Mon Jul 19 21:07:05 2021

@author: joe
"""
import numpy as np
import networkx as nx

def _concentration(B, A, norm, balance=0.5):
    """

    """
    # compute the RMS value of the measure across nodes
    self_rms = np.sqrt(np.sum(B**2))

    # now compute the square root of the mean product of each nodes'
    # measure and the average of its neighbor's measures
    Bmat = np.matrix(B.reshape(1,-1))
    BAB = np.array(Bmat*A).ravel()*B
    other_rms = np.sqrt(np.sum(BAB/norm))

    return balance*self_rms + (1-balance)*other_rms


def _compute_node_measure_concentrations(coldict, g, balance=0.5):
    """

    """
    outdict = {}
    A = nx.adjacency_matrix(g)
    norm = np.array(A.sum(1)).ravel() + 1e-6

    # compute concentration for each color
    for c in coldict:
        outdict[c] = _concentration(coldict[c], A, norm, balance)

    return outdict
