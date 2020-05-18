import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from Utilities.metric import euclidean


def mst(arr):
    """ Returns the Minimum Spanning Tree of a List of Points (Fully-Connected)

    Args:
        arr (np.ndarray): A list of points.

    Returns:
        (np.ndarray) A 2D-distance matrix connecting all points
    """
    dist = np.zeros((arr.shape[0], arr.shape[0]))
    for i in range(arr.shape[0]):
        for k in range(i+1, arr.shape[0]):
            dist[i, k] = euclidean(arr[i], arr[k])
    mst = minimum_spanning_tree(dist)
    mst.toarray().astype(float)
    return mst


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib import collections as mc

    X = np.random.uniform(-50, 50, (12, 2))

    dist = mst(X)
    lines = []
    for i in range(dist.shape[0]):
        for k in range(i+1, dist.shape[0]):
            if dist[i, k] != 0:
                lines.append([X[i], X[k]])

    lc = mc.LineCollection(lines, linewidth=2)
    fig, ax = plt.subplots()
    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    plt.scatter(X[:, 0], X[:, 1])

    plt.show()
