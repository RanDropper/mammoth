import numpy as np
from sklearn.cluster import KMeans
import copy

def ts_kmeans(X, nc, autocorr=None):
    if autocorr is None:
        x = copy.deepcopy(X)
        x -= x.mean(axis=1, keepdims=True)
        x /= x.std(axis=1, keepdims=True)
        corr = np.matmul(x, x.T)
    else:
        corr = autocorr

    cluster = KMeans(n_clusters=nc).fit(corr)
    return cluster.labels_