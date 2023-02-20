import numpy as np
from sklearn.cluster import KMeans
import copy

def ts_kmeans(X, nc):
    x = copy.deepcopy(X)
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True)

    cluster = KMeans(n_clusters=nc).fit(x)
    return cluster.labels_