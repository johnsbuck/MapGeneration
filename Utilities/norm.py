"""N-Dimensional Norm

Defines the norm of 2 points in N dimensional space with different norms.
"""
import numpy as np


def norm(a, pow=2):
    """ Lp Norm

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,). Defines a point in 2-D space.
        pow (float): The norm used for distance (Default: 2)

    Returns:
        (float) The distance between point a and point b with Lp Norm.
    """
    return np.float_power(np.sum([np.float_power(np.abs(a[i]), pow) for i in range(len(a))]), 1./pow)


def manhattan(a):
    """Manhattan Norm

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.

    Returns:
        The distance between points a and b with L1 norm.
    """
    return norm(a, pow=1)


def euclidean(a):
    """Euclidean Norm

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.

    Returns:
        The distance between points a and b with L2 norm.
    """
    return norm(a, pow=2)


def minkowski(a):
    """Minkowski Norm

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.

    Returns:
        The distance between points a and b with L3 norm.
    """
    return norm(a, pow=3)


def chebyshev(a):
    """Chebyshev Norm

    Arguments:
        a (numpy.ndarray): A numpy array of shape (2,) or (2, 1). Defines a point in 2-D space.

    Returns:
        The distance between points a and b with L-Infinity norm.
    """
    return np.max([np.abs(a[i]) for i in range(len(a))])
