"""N-Dimensional Distance

Defines the distance of 2 points in N dimensional space with different norms.
"""

import numpy as np


def metric(a, b, pow=2):
    """2-D Distance of Lp Norm

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,). Defines a point in 2-D space.
        b (numpy.ndarray): A numpy array of shape (2,). Defines a point in 2-D space.
        pow (float): The norm used for distance (Default: 2)

    Returns:
        (float) The distance between point a and point b with Lp Norm.
    """
    if a.shape != b.shape:
        raise ValueError("Points a and b are not the same shape.")
    return np.float_power(np.sum([np.float_power(np.abs(a[i] - b[i]), pow) for i in range(len(a))]), 1./pow)


def manhattan(a, b):
    """Manhattan Distance

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.
        b (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.

    Returns:
        The distance between points a and b with L1 norm.
    """
    return metric(a, b, pow=1)


def euclidean(a, b):
    """Euclidean Distance

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.
        b (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.

    Returns:
        The distance between points a and b with L2 norm.
    """
    return metric(a, b, pow=2)


def minkowski(a, b):
    """Minkowski Distance

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.
        b (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.

    Returns:
        The distance between points a and b with L3 norm.
    """
    return metric(a, b, pow=3)


def chebyshev(a, b):
    """Chebyshev Distance

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.
        b (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.

    Returns:
        The distance between points a and b with L-Infinity norm.
    """
    return np.max([np.abs(a[i] - b[i]) for i in range(len(a))])
