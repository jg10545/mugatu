"""
Created on Fri Jul 16 13:41:49 2021

@author: joe
"""
import numpy as np

from mugatu._xmeans import _compute_kmeans, _compute_BIC, _compute_xmeans



# generate some test data to use for unit tests
N = 1000
X = np.random.normal(0, 1, (N,2)).astype(np.float32)

X[:250,0] += 5

X[250:500,0] -= 4
X[250:500,1] -= 2

X[500:750,0] -= 1
X[500:750,1] += 4


def test_kmeans():
    k = 3
    I, centroids = _compute_kmeans(X, k)

    assert len(I) == X.shape[0]
    assert centroids.shape == (k, X.shape[1])
    assert I.max() == k-1
    assert I.min() == 0

def test_bic():
    k = 3
    I, centroids = _compute_kmeans(X, k)
    # compute BIC for the kmeans model
    BIC_kmeans = _compute_BIC(X, centroids, I, False)
    # and let's compare to scrambled centroids
    centroids_scrambled = np.random.normal(0, 1, size=centroids.shape)
    BIC_scrambled = _compute_BIC(X, centroids_scrambled, I, False)
    assert BIC_kmeans > BIC_scrambled

def test_aic():
    k = 3
    I, centroids = _compute_kmeans(X, k)
    # compute BIC for the kmeans model
    AIC_kmeans = _compute_BIC(X, centroids, I, True)
    # and let's compare to scrambled centroids
    centroids_scrambled = np.random.normal(0, 1, size=centroids.shape)
    AIC_scrambled = _compute_BIC(X, centroids_scrambled, I, True)
    assert AIC_kmeans > AIC_scrambled


def test_compute_xmeans():
    init_k = 3

    for aic in [True, False]:
        for min_size in [0,100]:
            I = _compute_xmeans(X, aic=aic, init_k=init_k, min_size=min_size)

            assert I.shape == (X.shape[0],)
            assert I.max() >= init_k

            cluster_sizes = [(I == i).sum() for i in set(I)]
            assert np.min(cluster_sizes) >= min_size
